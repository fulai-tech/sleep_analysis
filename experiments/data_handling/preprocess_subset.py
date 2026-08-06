"""
MESA 数据预处理管线 — 按指定被试数量运行（多进程并行, 断点续跑）
=========================================
与同目录下 data_handling.py 功能完全一致（EDR 提取 → RRV 提取 → 清洗对齐 → 特征提取 → 合并），
唯一区别是可以通过命令行参数限制处理的被试数量，而不是默认跑全量。

用法:
    python experiments/data_handling/preprocess_subset.py 5                  # 只处理前5个被试 (串行)
    python experiments/data_handling/preprocess_subset.py 50 --n-workers 10 # 50个被试, 10 worker 并行
    python experiments/data_handling/preprocess_subset.py 2056 --n-workers 10 --no-edr \
        --output-dir /srv/shared/psgdata/xxx/mesa_processed               # 全量 (因果模式示例)

预处理管线（6 步）:
    1. EDR 特征提取 — 从 ECG 心电信号提取呼吸波形，计算 RRV 特征（--no-edr 时跳过）
    2. RRV 特征提取 — 从 EDF 胸腔呼吸带信号提取真实呼吸特征
    3. MESA 数据预处理 — 加载体动/ECG/PSG/呼吸数据，清洗、时间对齐、睡眠分期标注转换
    4. 体动特征提取 — 对活动计数序列用不同窗口大小计算统计量，产出 370 维特征
    5. HRV 特征提取 — 从 RR 间期计算时域/频域/非线性心率变异性特征，产出 31 维
    6. 特征合并 — 将 ACT/HRV/RRV/EDR 四模态特征按 epoch 对齐，产出 460 维特征表

Step 1-5 由一个 worker 对被试完整串行处理（每被试独立, 互不依赖），Step 6 在主进程最后统一执行。
支持断点续跑（checkpoint.json）与中断清理（SIGINT/SIGTERM 时终止 worker, 不留孤儿进程）。

输入路径（在 study_data.json 中配置）:
    - mesa_path: MESA 原始数据目录 (EDF、体动 CSV、PSG XML 标注等)
    - mesa_path_edf: EDF 文件目录
    - processed_mesa_path: 预处理产出目录
"""

# 必须在 import numpy / mne 之前设置，防止每个子进程内部的多线程争抢
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

import sleep_analysis.processing_config as pc


# ---------------------------------------------------------------------------
# 单被试处理 (多进程 worker) — Step 1~5 顺序执行
# ---------------------------------------------------------------------------

def process_one_subject(cfg: dict):
    """对单个被试执行 Step 1(可选 EDR) → 2(RRV) → 3(清洗对齐) → 4(体动特征) → 5(HRV 特征)。

    返回 (subj, ok, info): ok=True 表示 Step 3 通过且 Step 4/5 完成;
    info 为 dict, 含 reason / timings 等。
    """
    subj = cfg["subj"]
    edf_dir = Path(cfg["edf_dir"])
    mesa_path = Path(cfg["mesa_path"])
    processed_path = Path(cfg["processed_path"])
    no_edr = cfg["no_edr"]
    output_dir = cfg.get("output_dir")
    overlap = cfg["overlap"]          # DataFrame, spawn pickle 传入
    dataset_info = cfg["dataset_info"]  # DataFrame

    # 若指定 --output-dir, 子进程 import 后模块级路径被重置为 study_data.json 的值,
    # 需要在 worker 内重新覆盖 (clean_data_to_csv / merge_features 使用模块级路径)
    if output_dir:
        from sleep_analysis.preprocessing.mesa_dataset import utils as _p_utils
        from sleep_analysis.feature_extraction.mesa_datasst import utils as _f_utils
        _p_utils.processed_mesa_path = processed_path
        _f_utils.processed_mesa_path = processed_path

    from sleep_analysis.preprocessing.utils import extract_edf_channel
    from sleep_analysis.preprocessing.mesa_dataset.edr import _extract_edr, process_resp
    from sleep_analysis.feature_extraction.mesa_datasst.rrv import extract_rrv_features_helper
    from sleep_analysis.preprocessing.mesa_dataset.edr_placeholder import make_edr_placeholder
    from sleep_analysis.preprocessing.mesa_dataset.preprocess_mesa import _clean_data_helper
    from sleep_analysis.preprocessing.mesa_dataset.utils import match_exclusion_criteria
    from sleep_analysis.feature_extraction.mesa_datasst.actigraphy import calc_actigraph_features
    from sleep_analysis.feature_extraction.mesa_datasst.hrv import calc_hrv_features

    t0 = time.time()
    timings = {}
    out_edr = out_rrv = out_ecg = out_act = out_hrv = None
    try:
        # --- Step 1: EDR (--no-edr 时跳过, 下游用全 0 占位表) ---
        out_edr = processed_path / f"edr_respiration_features_raw/edr_respiration{subj}.csv"
        if not no_edr and not out_edr.exists():
            raw_ecg, epochs = extract_edf_channel(edf_dir, subj_id=int(subj), channel="EKG")
            edr_signal = _extract_edr(raw_ecg, sampling_rate=256)
            resp_df, epochs = process_resp(edr_signal.respiratory_signal, epochs)
            features = extract_rrv_features_helper(resp_df, nan_pad=0.0, sampling_rate=32)
            features.to_csv(out_edr)
        timings["edr"] = time.time() - t0
        t1 = time.time()

        # --- Step 2: RRV 特征提取 ---
        out_rrv = processed_path / f"respiration_features_raw/respiration{subj}.csv"
        if not out_rrv.exists():
            resp_df, epochs = extract_edf_channel(edf_dir, subj_id=int(subj), channel="Thor")
            resp_df, epochs = process_resp(resp_df, epochs)
            features = extract_rrv_features_helper(resp_df)
            features.to_csv(out_rrv)
        timings["rrv"] = time.time() - t1
        t2 = time.time()

        # --- Step 3: MESA 数据预处理 (清洗/对齐/标注) ---
        # 质量筛选 (人口学数据)
        if match_exclusion_criteria(dataset_info, subj):
            return (subj, False, {"reason": "excluded_quality"})

        import mesa_data_importer as importer
        df_act = importer.load_single_actigraphy(mesa_path, int(subj))
        df_rpt = importer.load_single_r_point(mesa_path, int(subj))
        df_psg = importer.load_single_psg(mesa_path, int(subj))
        df_resp = importer.load_single_resp_features(processed_path, int(subj))
        if no_edr:
            # EDR 屏蔽: 用 RRV 特征表生成全 0 占位表, 保持下游对齐/合并/训练代码不变
            df_edr = make_edr_placeholder(df_resp)
        else:
            df_edr = importer.load_single_edr_feature(processed_path, int(subj))
        _clean_data_helper(df_act, df_rpt, df_psg, df_resp, df_edr, overlap, int(subj))
        timings["prep"] = time.time() - t2
        t3 = time.time()

        # --- Step 4: 体动特征提取 ---
        out_act = processed_path / f"actigraph_features/actigraph_features{subj}.csv"
        if not out_act.exists():
            act = pd.read_csv(processed_path / f"actigraph_data_clean/actigraph_data_clean{subj}.csv")
            feats = calc_actigraph_features(act["activity"])
            feats.to_csv(out_act, index=False)
        timings["act"] = time.time() - t3
        t4 = time.time()

        # --- Step 5: HRV 特征提取 ---
        out_hrv = processed_path / f"hrv_features/hrv_features{subj}.csv"
        if not out_hrv.exists():
            hr = pd.read_csv(processed_path / f"ecg_data_clean/ecg_data_clean{subj}.csv")
            feats = calc_hrv_features(hr)
            feats.to_csv(out_hrv, index=False)
        timings["hrv"] = time.time() - t4

        return (subj, True, {"timings": timings})

    except Exception as e:
        # 20260806: 不再清理"半成品" — 原逻辑会把 Step 2 已成功生成的 RRV
        # raw 文件也删掉 (Step 3 失败连带误删, 重跑时白白重算)。
        # 失败被试不进 checkpoint, 重跑会重新生成所有文件, 半成品自然被覆盖, 无需清理。
        return (subj, False, {"reason": str(e)[:200]})


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MESA 数据预处理管线（按被试数量运行, 多进程并行）")
    parser.add_argument("n_subjects", type=int, nargs="?", default=5, help="处理的被试数量（默认 5）")
    parser.add_argument("--no-edr", action="store_true",
                        help="屏蔽 EDR 处理: 跳过 Step 1, EDR 特征用全 0 占位表替入 (下游代码不变)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认 study_data.json 的 processed_mesa_path)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="并行 worker 数 (默认 1 = 串行)")
    args = parser.parse_args()

    N = args.n_subjects

    # 读取 study_data.json，获取原始数据和输出目录的路径
    cfg_path = Path(__file__).parents[2] / "study_data.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    # EDF 目录: 包含 2000+ 个 mesa-sleep-XXXX.edf 文件
    edf_dir = Path(cfg["mesa_path_edf"])

    # 从 EDF 文件名中提取被试 ID (如 mesa-sleep-0001.edf → "0001")
    # 按文件名排序取前 N 个，确保每次运行选择相同的被试
    all_edfs = sorted(edf_dir.glob("*.edf"))
    subjects = []
    for edf in all_edfs:
        m = re.findall(r"(\d{4})", edf.name)    # 匹配4位数字作为被试ID
        if m and m[0] not in subjects:
            subjects.append(m[0])
        if len(subjects) >= N:
            break
    print(f"Selected {len(subjects)} subjects: {subjects[:5]}{'...' if len(subjects) > 5 else ''}")

    # 处理后的数据统一存放在这个目录下
    processed_path = Path(args.output_dir) if args.output_dir else Path(cfg["processed_mesa_path"])

    # 目录级模式锁: 跨模式重跑已有目录直接拒绝, 防止静默复用旧模式文件 (reviewer High/Medium)
    pc.check_run_mode(
        processed_path,
        {"causal": pc.causal, "no_edr": args.no_edr, "script": "preprocess_subset.py"},
    )

    # 若指定 --output-dir, 同步覆盖子模块的模块级输出路径 (主进程的 Step 6 merge 需要;
    # worker 内会再覆盖一次, 见 process_one_subject)
    if args.output_dir:
        from sleep_analysis.preprocessing.mesa_dataset import utils as _p_utils
        from sleep_analysis.feature_extraction.mesa_datasst import utils as _f_utils
        _p_utils.processed_mesa_path = processed_path
        _f_utils.processed_mesa_path = processed_path

    # 确保所有输出子目录存在
    for subdir in [
        "edr_respiration_features_raw", "respiration_features_raw",
        "actigraph_data_clean", "ecg_data_clean",
        "respiration_features_clean", "edr_features_clean",
        "actigraph_features", "hrv_features", "features_full_combined",
    ]:
        (processed_path / subdir).mkdir(parents=True, exist_ok=True)

    if pc.causal:
        print("SLEEP_CAUSAL=1: RRV 特征因果处理（呼吸滤波/降采样仅使用已见数据）")
    if args.no_edr:
        print("--no-edr: EDR 特征提取跳过, 下游使用全 0 占位表")

    # PSG-体动重叠时间表 / 数据集信息表 (质量筛选) — 经 spawn pickle 传入 worker
    mesa_path = Path(cfg["mesa_path"])
    overlap = pd.read_csv(mesa_path / "overlap/mesa-actigraphy-psg-overlap.csv")
    dataset_info = pd.read_csv(mesa_path / "datasets/mesa-sleep-dataset-0.5.0.csv").set_index("mesaid")

    # --- 断点续跑 ---
    cp_file = processed_path / "checkpoint.json"
    completed = set()
    if cp_file.exists():
        with open(cp_file) as f:
            completed = set(json.load(f))
    pending = [s for s in subjects if s not in completed]
    if completed:
        print(f"Checkpoint: {len(completed)} done, {len(pending)} remaining")

    # --- 构建配置 ---
    base_cfg = {
        "edf_dir": str(edf_dir),
        "mesa_path": str(mesa_path),
        "processed_path": str(processed_path),
        "no_edr": args.no_edr,
        "output_dir": args.output_dir,
        "overlap": overlap,
        "dataset_info": dataset_info,
    }

    # --- Step 1~5: 多进程并行处理 ---
    t_start = time.time()
    success = 0
    excluded = 0
    fail = 0
    total_elapsed = 0.0
    executor = None

    def _cleanup_executor():
        nonlocal executor
        if executor is not None:
            for pid in list(executor._processes.keys()):
                try:
                    p = executor._processes[pid]
                    p.kill()
                    p.join(timeout=2)
                except Exception:
                    pass
            executor.shutdown(wait=False, cancel_futures=True)

    def _handle_signal(signum, frame):
        print(f"\n[INTERRUPTED] Signal {signum} received, cleaning up workers...", flush=True)
        _cleanup_executor()
        print("[INTERRUPTED] Workers terminated. Re-run to resume from checkpoint.", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    last_report = time.time()

    def save_cp(data):
        with open(cp_file, "w") as f:
            json.dump(sorted(data), f)

    try:
        executor = ProcessPoolExecutor(max_workers=args.n_workers,
                                       mp_context=mp.get_context("spawn"))
        futures = {}
        for subj in pending:
            task_cfg = {**base_cfg, "subj": subj}
            futures[executor.submit(process_one_subject, task_cfg)] = subj

        for i, future in enumerate(as_completed(futures)):
            subj = futures[future]
            try:
                subj_id, ok, info = future.result()
            except Exception as e:
                print(f"  [{subj}] CRASH: {e}")
                fail += 1
                continue

            if ok:
                success += 1
                t = info.get("timings", {})
                elapsed = sum(t.values())
                total_elapsed += elapsed
                completed.add(subj_id)
                save_cp(completed)
                print(f"[{datetime.now():%H:%M:%S}] [{subj_id}] OK "
                      f"({elapsed:.0f}s: EDR={t.get('edr',0):.0f}s RRV={t.get('rrv',0):.0f}s "
                      f"Prep={t.get('prep',0):.0f}s ACT={t.get('act',0):.0f}s HRV={t.get('hrv',0):.0f}s)")
            else:
                reason = info.get("reason", "?") if info else "?"
                if reason == "excluded_quality":
                    excluded += 1
                    print(f"  [{subj_id}] excluded (quality)")
                else:
                    fail += 1
                    print(f"  [{subj_id}] FAILED: {reason}")

            # 进度概要
            now = time.time()
            if now - last_report > 30:
                done = success + excluded + fail
                remaining = len(pending) - done
                avg = total_elapsed / success if success > 0 else 0
                eta_s = avg * remaining
                eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.0f}min"
                elapsed_s = now - t_start
                print(f"  ── Progress: {done}/{len(pending)} ({done/len(pending)*100:.1f}%) "
                      f"| avg={avg:.0f}s/subj | elapsed={elapsed_s/3600:.1f}h | ETA={eta_str}", flush=True)
                last_report = now
    finally:
        _cleanup_executor()

    # --- Step 6: 特征合并 (主进程, 全部 worker 完成后) ---
    print("\n=== Step 6: Merge features ===")
    if success > 0 or len(completed) > 0:
        from sleep_analysis.feature_extraction.mesa_datasst.utils import merge_features
        merge_features(overwrite=True)
        print("merge finished")
    else:
        print("No subjects processed, skipping merge.")

    # --- 完成 ---
    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed_total/3600:.1f}h")
    print(f"Results: {success} ok, {excluded} excluded (quality), {fail} failed, {len(completed)} total done")
    if success > 0:
        print(f"Avg: {total_elapsed/success:.0f}s/subject (sum of steps)")
    print(f"Output: {processed_path}")
    for d in sorted(processed_path.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.csv")))
            size = sum(f.stat().st_size for f in d.glob("*.csv"))
            print(f"  {d.name}: {n} files, {size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
