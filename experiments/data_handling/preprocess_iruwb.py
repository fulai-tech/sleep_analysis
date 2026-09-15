"""IR-UWB 20 Hz 雷达数据预处理管线（多进程并行, 断点续跑）。

输入布局（``--raw-dir``, 由 ``simulate_iruwb.py`` 生成, 真实采集系统按同格式交付）::

    <raw_dir>/Vp_<ID>/physio_<ID>.csv     # 列: time_offset_s, phase
    <raw_dir>/Vp_<ID>/labels_<ID>.csv     # 列: epoch, sleep, 5stage, 4stage, 3stage

产出布局（与 MESA/SHHS/MrOS 一致, 供训练管线直接消费）::

    <processed_dir>/
      run_config.json                              # 模式锁 (causal 等)
      checkpoint.json                              # 断点续跑
      features_full_combined/features_combined<ID>.csv
      sleep_stages/sleep_stages<ID>.csv

因果模式由环境变量 ``SLEEP_CAUSAL`` 控制（见 ``processing_config.py``）。
⚠️ **换模式必须用新的 --output-dir** —— 输出目录有模式锁, 跨模式续跑会被拒绝。

用法
----
    # 因果模式 (实时分期用)
    SLEEP_CAUSAL=1 python experiments/data_handling/preprocess_iruwb.py \\
        --raw-dir /tmp/iruwb_raw --output-dir /tmp/iruwb_processed_causal --n-workers 4

    # 非因果模式 (复现作者口径, 特征含未来信息)
    python experiments/data_handling/preprocess_iruwb.py \\
        --raw-dir /tmp/iruwb_raw --output-dir /tmp/iruwb_processed_leak --n-workers 4
"""

# 必须在 import numpy 之前设置，防止每个子进程内部的多线程争抢
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import re
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

import sleep_analysis.processing_config as pc  # noqa: E402

EPOCH_S = 30.0


def process_one_subject(cfg: dict):
    """对单个被试跑完整管线, 返回 (subj, ok, info)。"""
    subj = cfg["subj"]
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])

    from sleep_analysis.preprocessing.iruwb.pipeline import process_signal

    t0 = time.time()
    try:
        subj_dir = raw_dir / f"Vp_{subj}"
        physio_path = subj_dir / f"physio_{subj}.csv"
        labels_path = subj_dir / f"labels_{subj}.csv"
        if not physio_path.exists():
            return (subj, False, {"reason": "no physio file"})

        physio = pd.read_csv(physio_path)
        signal = physio["phase"].to_numpy(dtype=float)
        if len(signal) < 2:
            return (subj, False, {"reason": "signal too short"})

        # 采样率由时间轴推断, 不硬编码 —— 将来换硬件只需换数据
        if "time_offset_s" in physio.columns and len(physio) > 1:
            dt = np.median(np.diff(physio["time_offset_s"].to_numpy(dtype=float)))
            fs = float(round(1.0 / dt, 6)) if dt > 0 else 20.0
        else:
            fs = 20.0

        # 起始时刻: 仿真数据写在 sim_config, 真实数据应从采集日志取
        start_time = "2026-01-01 22:00:00"
        cfg_file = subj_dir / f"sim_config_{subj}.json"
        if cfg_file.exists():
            start_time = json.loads(cfg_file.read_text()).get("start_time", start_time)

        features, diag = process_signal(signal, fs, start_time)

        if not labels_path.exists():
            return (subj, False, {"reason": "no labels file"})
        labels = pd.read_csv(labels_path)

        # 标签按 **epoch 序号** 对齐, 不按行位置 —— 管线会剔除拍数不足的 epoch,
        # 两边行数不再相等, 按位置截断会错位。
        # features 的索引是 epoch 时间轴, 起点 = floor(录制起点, 30s),
        # 而 labels 的 epoch 列是从录制起点开始的 0-based 序号。
        ep0 = pd.Timestamp(start_time).floor("30s")
        ordinals = ((features.index - ep0).total_seconds() / 30.0).round().astype(int)
        oob = (ordinals < 0) | (ordinals >= len(labels))
        if oob.any():
            return (subj, False,
                    {"reason": f"{int(oob.sum())} 个 epoch 序号超出标签范围 "
                               f"(0..{len(labels)-1}), 特征与标签时间轴不匹配"})
        labels = labels.iloc[ordinals].reset_index(drop=True)

        n = len(features)
        if n == 0:
            return (subj, False, {"reason": "no epochs after filtering"})
        # 睡眠时长门槛与 MESA/SHHS 保持一致 (默认 >2h 睡眠 = 120 epoch)
        sleep_n = int(pd.to_numeric(labels["sleep"], errors="coerce").fillna(0).to_numpy()[:n].sum())
        if sleep_n <= cfg["min_sleep_epochs"]:
            return (subj, False,
                    {"reason": f"sleep {sleep_n} <= {cfg['min_sleep_epochs']} epochs"})

        out_feat = processed_dir / "features_full_combined" / f"features_combined{subj}.csv"
        out_lab = processed_dir / "sleep_stages" / f"sleep_stages{subj}.csv"
        # index=True 与 MESA/SHHS 的约定一致: 训练端用 index_col=0 消费
        features.to_csv(out_feat, index=True)
        labels.to_csv(out_lab, index=False)

        elapsed = time.time() - t0
        return (subj, True, {
            "elapsed": elapsed,
            "n_epochs": n,
            "n_beats": diag.beat_result.n_detected,
            "n_beats_removed": diag.beat_result.n_removed_outlier,
        })
    except Exception as e:
        return (subj, False, {"reason": f"{type(e).__name__}: {str(e)[:180]}"})


def main() -> None:
    p = argparse.ArgumentParser(description="IR-UWB 20 Hz 预处理管线")
    p.add_argument("--raw-dir", type=Path, required=True, help="原始数据目录")
    p.add_argument("--output-dir", type=Path, required=True, help="产出目录")
    p.add_argument("--n-workers", type=int, default=4, help="并行 worker 数")
    p.add_argument("--n-subjects", type=int, default=99999, help="限制处理数量 (调试用)")
    p.add_argument("--min-sleep-epochs", type=int, default=120,
                   help="睡眠时长门槛 (epoch, 30s 一个)。默认 120 = 2h, 与 MESA/SHHS 一致；\n"
                        "仿真短录制调试时可调低")
    args = p.parse_args()

    processed_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise SystemExit(f"原始数据目录不存在: {raw_dir}")

    # 目录级模式锁: 跨模式重跑已有目录直接拒绝, 防止静默复用旧模式文件
    pc.check_run_mode(processed_dir, {
        "causal": pc.causal, "no_edr": True, "script": "preprocess_iruwb.py",
    })

    for sub in ["features_full_combined", "sleep_stages"]:
        (processed_dir / sub).mkdir(parents=True, exist_ok=True)

    subjects = sorted(m.group(1) for d in raw_dir.glob("Vp_*")
                      if (m := re.match(r"Vp_(\w+)", d.name)))
    subjects = subjects[:args.n_subjects]
    if not subjects:
        raise SystemExit(f"在 {raw_dir} 下没找到 Vp_* 被试目录")

    mode = "因果 (SLEEP_CAUSAL=1)" if pc.causal else "非因果 (含未来信息)"
    print(f"处理模式: {mode}")
    print(f"被试 {len(subjects)} 个, {args.n_workers} worker")

    cp_file = processed_dir / "checkpoint.json"
    completed = set()
    if cp_file.exists():
        completed = set(json.loads(cp_file.read_text()))
    pending = [s for s in subjects if s not in completed]
    if completed:
        print(f"Checkpoint: 已完成 {len(completed)}, 待处理 {len(pending)}")

    base_cfg = {"raw_dir": str(raw_dir), "processed_dir": str(processed_dir),
                "min_sleep_epochs": args.min_sleep_epochs}
    executor = None

    def _cleanup():
        nonlocal executor
        if executor is not None:
            for pid in list(getattr(executor, "_processes", {})):
                try:
                    executor._processes[pid].kill()
                except Exception:
                    pass
            executor.shutdown(wait=False, cancel_futures=True)

    def _on_signal(signum, _frame):
        print(f"\n[INTERRUPTED] 信号 {signum}, 清理 worker...", flush=True)
        _cleanup()
        print("[INTERRUPTED] 已终止。重跑可从 checkpoint 续。", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    success = fail = 0
    t_start = time.time()
    try:
        if args.n_workers > 1:
            executor = ProcessPoolExecutor(max_workers=args.n_workers,
                                           mp_context=mp.get_context("spawn"))
            futures = {executor.submit(process_one_subject, {**base_cfg, "subj": s}): s
                       for s in pending}
            for fut in as_completed(futures):
                subj = futures[fut]
                try:
                    subj_id, ok, info = fut.result()
                except Exception as e:
                    print(f"  [{subj}] CRASH: {e}")
                    fail += 1
                    continue
                _report(subj_id, ok, info, completed, cp_file)
                success += ok
                fail += (not ok)
        else:
            for s in pending:
                subj_id, ok, info = process_one_subject({**base_cfg, "subj": s})
                _report(subj_id, ok, info, completed, cp_file)
                success += ok
                fail += (not ok)
    finally:
        _cleanup()

    print(f"\n{'=' * 70}")
    print(f"完成于 {datetime.now():%Y-%m-%d %H:%M:%S}, 用时 {(time.time() - t_start) / 60:.1f} min")
    print(f"成功 {success}, 失败 {fail}, 累计完成 {len(completed)}")
    print(f"产出: {processed_dir}")
    for d in sorted(processed_dir.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.csv")))
            if n:
                mb = sum(f.stat().st_size for f in d.glob("*.csv")) / 1e6
                print(f"  {d.name}: {n} 文件, {mb:.1f} MB")


def _report(subj_id, ok, info, completed, cp_file):
    if ok:
        completed.add(subj_id)
        cp_file.write_text(json.dumps(sorted(completed)))
        print(f"[{datetime.now():%H:%M:%S}] [{subj_id}] OK "
              f"({info.get('elapsed', 0):.1f}s, {info.get('n_epochs', 0)} epoch, "
              f"检出 {info.get('n_beats', 0)} 拍/剔除 {info.get('n_beats_removed', 0)})",
              flush=True)
    else:
        print(f"  [{subj_id}] FAILED: {info.get('reason', '?')}", flush=True)


if __name__ == "__main__":
    main()
