"""
SHHS 数据预处理管线 (多进程并行，断点续跑)
=========================================
适配 SHHS1 / SHHS2，无体动数据，R-point 从 ECG 自动生成。

用法:
    python experiments/data_handling/preprocess_shhs.py --study shhs1 --n-subjects 99999
    python experiments/data_handling/preprocess_shhs.py --study shhs2 --n-subjects 99999
    python experiments/data_handling/preprocess_shhs.py --study shhs1 --n-subjects 3   # 试跑
"""

# 必须在 import numpy / mne 之前设置，防止每个子进程内部的多线程争抢
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

import sleep_analysis.processing_config as pc

# ---------------------------------------------------------------------------
# 纯函数 — 无模块级副作用，可供多进程安全调用
# ---------------------------------------------------------------------------

def parse_shhs_xml(xml_path):
    """解析 NSRR XML，返回逐 epoch 睡眠分期 DataFrame。"""
    tree = ElementTree.parse(str(xml_path))
    root = tree.getroot()
    time_list, sleep_list = [], []
    j = 0
    for elem in root:
        for subelem in elem:
            if subelem[0].text == "Stages|Stages":
                duration = float(subelem[3].text)
                for _ in range(int(duration / 30)):
                    time_list.append(j)
                    sleep_list.append(subelem[1].text)
                    j += 30
    return pd.DataFrame({"time": time_list, "sleep": sleep_list})


def generate_r_points_from_ecg(raw_ecg_values, sampling_rate):
    """R 峰自动检测（原版 neurokit2）。sampling_rate 由 EDF 动态读取，不做硬编码。

    当前决策（2026-08-05）: 回退原版 — HRV 相关处理保持与原版一致
    （HRV 特征实测差异为 0% 量级, 且保持与原版已训练模型的可比性）。

    历史说明: 此前实现了 causal 分支（processing_config.causal 为 True 时改用
    preprocessing/ecg_rpeaks.rpeaks_causal, 自写因果 Pan-Tompkins）— neurokit2 的
    ecg_clean (sosfiltfilt 零相位双向滤波, 毫秒级延迟) 与 ecg_peaks
    (全局梯度/长度阈值) 均非因果。如需启用, 取消下方注释并改为:

        if pc.causal:
            from sleep_analysis.preprocessing.ecg_rpeaks import rpeaks_causal
            rpeaks = rpeaks_causal(np.asarray(raw_ecg_values, dtype=float), sampling_rate)
        else:
            ...下方原版...
    """
    import neurokit2 as nk
    ecg_cleaned = nk.ecg_clean(raw_ecg_values, sampling_rate=sampling_rate)
    _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    rpeaks = rpeaks_info["ECG_R_Peaks"]
    seconds = rpeaks / sampling_rate
    epoch = (seconds / 30 + 1).astype(int)
    return pd.DataFrame({
        "RPoint": rpeaks, "seconds": seconds, "epoch": epoch, "TPoint": 1,
    })


def sleep_stage_map(df_psg):
    """NSRR 分期字符串 → 数字标签 (Wake=0, N1=1, N2=2, N3=3, REM=4)。

    SHHS 使用 Rechtschaffen & Kales 标准，包含 MESA 没有的额外阶段:
      - Stage 4 sleep|4 → 映射为 N3 (3), 与 Stage 3 合并
      - Unscored|9    → NaN, 后续从交集中剔除
      - Movement|6     → NaN, 同上
    """
    mapping = {
        "Wake|0": 0,
        "Stage 1 sleep|1": 1,
        "Stage 2 sleep|2": 2,
        "Stage 3 sleep|3": 3,
        "Stage 4 sleep|4": 3,      # R&K Stage 4 = AASM N3
        "REM sleep|5": 4,           # NSRR REM=5 → AASM REM=4 (与 MESA 对齐)
        "Unscored|9": np.nan,      # 未评分 — 剔除
        "Movement|6": np.nan,       # 体动伪迹 — 剔除
    }
    ss = df_psg[["sleep"]].copy()
    ss["5stage"] = ss["sleep"].map(mapping)
    ss["4stage"] = ss["5stage"].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 3})  # Wake/N1+N2/N3/REM
    ss["3stage"] = ss["5stage"].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 2})  # Wake/NREM/REM
    # binary sleep: 仅 5stage!=0 且非 NaN 的 epoch 为 Sleep=1
    ss["sleep"] = ss["5stage"].apply(lambda x: 1 if pd.notna(x) and x != 0 else (0 if pd.notna(x) else np.nan))
    return ss


def process_rpoint(ecg_df):
    """R-point → RR 间期 → 去异常 → HR。

    已抽到 preprocessing/rr_utils.py（与 MESA 共用, 含 causal 分支）。
    """
    from sleep_analysis.preprocessing.rr_utils import process_rpoint as _process_rpoint

    return _process_rpoint(ecg_df)


def extract_edf_channel(edf_dir, subj, channel, study_prefix):
    """从 SHHS EDF 提取指定通道，返回原始数据 + 实际采样率。

    采样率随数据集/文件而变，因此按 EDF 实际 sfreq 动态读取，不在此处重采样，
    由下游 process_resp 按实际采样率对齐到 32 Hz：
        SHHS1: ECG ~125 Hz
        SHHS2: ECG 既有 250 Hz 也有 256 Hz（不同站点/设备），不是固定 256 Hz
    注意：MNE 读取混合采样率的 EDF 时会把所有通道上采样到文件内最高采样率，
    因此这里的 native_rate 是文件级公共采样率（如 THOR RES 原生 8-10 Hz 会被提升到 250/256）。
    """
    import mne
    from sleep_analysis.preprocessing.utils import _create_datetime_index

    edf_path = Path(edf_dir) / f"{study_prefix}-{subj}.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    edf = mne.io.read_raw_edf(edf_path, verbose=False)
    ch_data = edf.pick_channels([channel])
    native_rate = float(ch_data.info["sfreq"])
    data = ch_data.get_data()[0, :]

    time_idx, epochs = _create_datetime_index(ch_data.info["meas_date"],
                                              times_array=ch_data.times)
    col_name = "ecg" if channel in ("ECG", "EKG") else "resp"
    return pd.DataFrame(data, index=time_idx).rename(columns={0: col_name}), epochs, native_rate


# ---------------------------------------------------------------------------
# 单被试处理 (多进程 worker)
# ---------------------------------------------------------------------------

def process_one_subject(cfg):
    """对单个被试执行完整 5 步管线。返回 (subj, ok, timings) 或 (subj, False, None)。"""
    subj = cfg["subj"]
    study = cfg["study"]
    edf_dir = Path(cfg["edf_dir"])
    annot_dir = Path(cfg["annot_dir"])
    rpoint_dir = Path(cfg["rpoint_dir"])
    processed_dir = Path(cfg["processed_dir"])
    channel_ecg = cfg["channel_ecg"]
    channel_resp = cfg["channel_resp"]

    from sleep_analysis.preprocessing.mesa_dataset.edr import _extract_edr, process_resp
    from sleep_analysis.feature_extraction.mesa_datasst.rrv import extract_rrv_features_helper
    from sleep_analysis.feature_extraction.mesa_datasst.hrv import calc_hrv_features
    from sleep_analysis.preprocessing.mesa_dataset.respiration import check_resp_features
    from sleep_analysis.preprocessing.mesa_dataset.edr_placeholder import make_edr_placeholder

    t0 = time.time()
    timings = {}

    out_edr = out_rrv = out_ecg = out_hrv = out_merge = None
    try:
        # --- Step 1: EDR (--no-edr 时跳过, 下游用全 0 占位表) ---
        out_edr = processed_dir / f"edr_respiration_features_raw/edr_respiration{subj}.csv"
        if not cfg["no_edr"] and not out_edr.exists():
            raw_ecg, epochs, ecg_rate = extract_edf_channel(edf_dir, subj, channel_ecg, study)
            edr_signal = _extract_edr(raw_ecg, sampling_rate=ecg_rate)
            resp_df, epochs = process_resp(edr_signal.respiratory_signal, epochs,
                                           sampling_rate_in=ecg_rate)
            features = extract_rrv_features_helper(resp_df, nan_pad=0.0, sampling_rate=32)
            features.to_csv(out_edr)
        timings["edr"] = time.time() - t0
        t1 = time.time()

        # --- Step 2: RRV ---
        out_rrv = processed_dir / f"respiration_features_raw/respiration{subj}.csv"
        if not out_rrv.exists():
            resp_df, epochs, resp_rate = extract_edf_channel(edf_dir, subj, channel_resp, study)
            resp_df, epochs = process_resp(resp_df, epochs,
                                           sampling_rate_in=resp_rate)
            features = extract_rrv_features_helper(resp_df)
            features.to_csv(out_rrv)
        timings["rrv"] = time.time() - t1
        t2 = time.time()

        # --- Step 3: Preprocess ---
        out_ecg = processed_dir / f"ecg_data_clean/ecg_data_clean{subj}.csv"
        if not out_ecg.exists():
            xml_path = annot_dir / f"{study}-{subj}-nsrr.xml"
            if not xml_path.exists():
                return (subj, False, {"reason": "no PSG XML"})

            df_psg = parse_shhs_xml(xml_path)
            if len(df_psg) == 0:
                return (subj, False, {"reason": "empty PSG"})

            # R-point
            rp_file = rpoint_dir / f"{study}-{subj}-rpoint.csv"
            if rp_file.exists():
                df_rp = pd.read_csv(rp_file)
            else:
                raw_ecg, _, _ecg_rate = extract_edf_channel(edf_dir, subj, channel_ecg, study)
                df_rp = generate_r_points_from_ecg(raw_ecg.iloc[:, 0].values,
                                                    sampling_rate=_ecg_rate)
            df_hr = process_rpoint(df_rp)

            # Sleep stages
            df_psg["epoch"] = df_psg["time"] // 30 + 1
            sleep_st = df_psg[["epoch", "sleep"]].drop_duplicates(subset="epoch")
            labels = sleep_stage_map(sleep_st)
            labels["epoch"] = sleep_st["epoch"].values
            # 剔除 Unscored (NaN) 和 Movement (NaN) 的 epoch
            labels = labels.dropna(subset=["5stage"])

            # Respiration & EDR features
            df_resp = pd.read_csv(out_rrv, index_col=0)
            if cfg["no_edr"] or not out_edr.exists():
                # EDR 屏蔽: 用 RRV 特征表生成全 0 占位表, 保持下游对齐/合并/训练代码不变
                df_edr = make_edr_placeholder(df_resp)
            else:
                df_edr = pd.read_csv(out_edr, index_col=0)
            df_resp = check_resp_features(df_resp)
            df_edr = check_resp_features(df_edr)
            df_edr.columns = [c.replace("RRV", "EDR") for c in df_edr.columns]

            # Epoch intersection
            e_hr = set(df_hr["epoch"].values)
            e_psg = set(labels["epoch"].values)
            e_resp = set(df_resp["epoch"].values) if "epoch" in df_resp.columns else set()
            e_edr = set(df_edr["epoch"].values) if "epoch" in df_edr.columns else set()
            intersect = e_hr & e_psg
            if e_resp: intersect &= e_resp
            if e_edr: intersect &= e_edr
            if len(intersect) == 0:
                return (subj, False, {"reason": "no overlap"})

            df_hr = df_hr[df_hr["epoch"].isin(intersect)].copy()
            labels = labels[labels["epoch"].isin(intersect)].copy()

            # Sleep duration filter
            sleep_n = labels["sleep"].sum()  # sleep 列被 sleep_stage_map 转为 0=Wake 1=Sleep
            if sleep_n <= 120:
                return (subj, False, {"reason": f"sleep {sleep_n} ≤ 120"})

            # Save
            ecg_clean = df_hr.copy()
            smap = labels.set_index("epoch")["5stage"]
            ecg_clean["stage"] = ecg_clean["epoch"].map(smap).fillna(0)
            ecg_clean.to_csv(out_ecg, index=False)

            labels[labels["epoch"].isin(ecg_clean["epoch"])].to_csv(
                processed_dir / f"sleep_stages/sleep_stages{subj}.csv", index=False)
            if "epoch" in df_resp.columns:
                df_resp = df_resp[df_resp["epoch"].isin(intersect)]
            df_resp.to_csv(processed_dir / f"rrv_features/respiration_features_clean{subj}.csv", index=False)
            df_edr.to_csv(processed_dir / f"rrv_features/edr_features_clean{subj}.csv", index=False)
        timings["prep"] = time.time() - t2
        t3 = time.time()

        # --- Step 4: HRV ---
        out_hrv = processed_dir / f"hrv_features/hrv_features{subj}.csv"
        if not out_hrv.exists():
            hr = pd.read_csv(out_ecg)
            feats = calc_hrv_features(hr)
            feats.to_csv(out_hrv, index=False)
        timings["hrv"] = time.time() - t3
        t4 = time.time()

        # --- Step 5: Merge ---
        out_merge = processed_dir / f"features_full_combined/features_combined{subj}.csv"
        if not out_merge.exists():
            h = pd.read_csv(out_hrv)
            r = pd.read_csv(processed_dir / f"rrv_features/respiration_features_clean{subj}.csv")
            e = pd.read_csv(processed_dir / f"rrv_features/edr_features_clean{subj}.csv")
            ml = min(len(h), len(r), len(e))
            combined = pd.concat([
                h.iloc[:ml].reset_index(drop=True),
                r.iloc[:ml].reset_index(drop=True),
                e.iloc[:ml].reset_index(drop=True),
            ], axis=1)
            combined = combined.loc[:, ~combined.columns.duplicated()]
            combined.to_csv(out_merge, index=False)
        timings["merge"] = time.time() - t4

        elapsed = time.time() - t0
        return (subj, True, {"elapsed": elapsed, "timings": timings})

    except Exception as e:
        # 20260806: 不再清理"半成品" — 原逻辑会把 Step 2 已成功生成的 RRV
        # raw 文件也删掉 (Step 3 失败连带误删, 重跑时白白重算 ~26s)。
        # 失败被试不进 checkpoint, 重跑会重新生成所有文件, 半成品自然被覆盖, 无需清理。
        return (subj, False, {"reason": str(e)[:120]})


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["shhs1", "shhs2"], default="shhs1")
    parser.add_argument("--n-subjects", type=int, default=99999)
    parser.add_argument("--n-workers", type=int, default=10,
                        help="并行 worker 数 (默认 10)")
    parser.add_argument("--no-edr", action="store_true",
                        help="屏蔽 EDR 处理: 跳过 Step 1, EDR 特征用全 0 占位表替入 (下游代码不变)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认 /srv/shared/psgdata/{study}_processed)")
    args = parser.parse_args()

    STUDY = args.study
    N = args.n_subjects
    N_WORKERS = args.n_workers

    SHHS_ROOT = Path("/mnt/nas_data/psgdata/SHHS")
    EDF_DIR = SHHS_ROOT / "polysomnography/edfs" / STUDY
    ANNOT_DIR = SHHS_ROOT / "polysomnography/annotations-events-nsrr" / STUDY
    RPOINT_DIR = SHHS_ROOT / "polysomnography/annotations-rpoints" / STUDY
    DATASET_DIR = SHHS_ROOT / "datasets"

    if STUDY == "shhs1":
        DATASET_CSV = DATASET_DIR / "shhs1-dataset-0.13.0.csv"
        OVERALL_COL = "overall_shhs1"
    else:
        DATASET_CSV = DATASET_DIR / "shhs2-dataset-0.13.0-utf8.csv"
        OVERALL_COL = "overall_shhs2"

    PROCESSED_DIR = Path(args.output_dir) if args.output_dir else Path(f"/srv/shared/psgdata/{STUDY}_processed")

    # 目录级模式锁: 跨模式重跑已有目录直接拒绝, 防止静默复用旧模式文件 (reviewer High/Medium)
    pc.check_run_mode(
        PROCESSED_DIR,
        {"causal": pc.causal, "no_edr": args.no_edr, "script": f"preprocess_shhs.py ({STUDY})"},
    )

    print(f"Study: {STUDY}")
    print(f"Output: {PROCESSED_DIR}")
    print(f"Workers: {N_WORKERS}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 加载数据 ---
    df_info = pd.read_csv(DATASET_CSV).set_index("nsrrid")
    all_edfs = sorted(EDF_DIR.glob("*.edf"))
    all_subjects = []
    for f in all_edfs:
        m = re.findall(r"(\d{6})", f.name)
        if m and m[0] not in all_subjects:
            all_subjects.append(m[0])
    print(f"Total EDF files: {len(all_subjects)}")

    # --- 质量筛选 ---
    subjects = []
    excluded = 0
    for subj in all_subjects:
        sid = int(subj)
        if sid not in df_info.index or pd.isna(df_info.loc[sid, OVERALL_COL]):
            continue
        if int(df_info.loc[sid, OVERALL_COL]) >= 4:
            subjects.append(subj)
        else:
            excluded += 1
    print(f"Quality filter (≥4): {len(subjects)} passed, {excluded} excluded")

    subjects = subjects[:N]
    print(f"Processing up to {len(subjects)} subjects")

    # --- 断点续跑 ---
    cp_file = PROCESSED_DIR / "checkpoint.json"
    completed = set()
    if cp_file.exists():
        with open(cp_file) as f:
            completed = set(json.load(f))
    pending = [s for s in subjects if s not in completed]
    if completed:
        print(f"Checkpoint: {len(completed)} done, {len(pending)} remaining")

    # --- 创建输出目录 ---
    for subdir in ["ecg_data_clean", "sleep_stages",
                   "respiration_features_raw", "edr_respiration_features_raw",
                   "hrv_features", "rrv_features", "features_full_combined"]:
        (PROCESSED_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # --- 构建配置 ---
    base_cfg = {
        "study": STUDY,
        "edf_dir": str(EDF_DIR),
        "annot_dir": str(ANNOT_DIR),
        "rpoint_dir": str(RPOINT_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "channel_ecg": "ECG",
        "channel_resp": "THOR RES",
        "no_edr": args.no_edr,
    }

    # --- 多进程并行处理 ---
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import signal as _signal

    t_start = time.time()
    success = 0
    fail = 0
    total_elapsed = 0.0
    executor = None

    def _cleanup_executor():
        """确保主进程退出时所有 worker 被强制终止，不留孤儿进程。"""
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

    _signal.signal(_signal.SIGINT, _handle_signal)
    _signal.signal(_signal.SIGTERM, _handle_signal)

    last_report = time.time()

    def save_cp(data):
        with open(cp_file, "w") as f:
            json.dump(sorted(data), f)

    try:
        executor = ProcessPoolExecutor(max_workers=N_WORKERS,
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
                total_elapsed += info["elapsed"]
                completed.add(subj_id)
                save_cp(completed)
                t = info["timings"]
                print(f"[{datetime.now():%H:%M:%S}] [{subj_id}] OK "
                      f"({info['elapsed']:.0f}s: EDR={t['edr']:.0f}s RRV={t['rrv']:.0f}s "
                      f"Prep={t['prep']:.0f}s HRV={t['hrv']:.0f}s Merge={t['merge']:.0f}s)")
            else:
                fail += 1
                reason = info.get("reason", "?") if info else "?"
                print(f"  [{subj_id}] SKIP: {reason}")

            # 进度概要
            now = time.time()
            if now - last_report > 30:
                done = success + fail
                remaining = len(pending) - done
                avg = total_elapsed / success if success > 0 else 0
                eta_s = avg * remaining
                eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.0f}min"
                elapsed_s = now - t_start
                print(f"  ── Progress: {done}/{len(pending)} ({done/len(pending)*100:.1f}%) "
                      f"| avg={avg:.0f}s/subj | elapsed={elapsed_s/3600:.1f}h | ETA={eta_str}")
                last_report = now
    finally:
        _cleanup_executor()

    # --- 最终统计 ---
    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed_total/3600:.1f}h")
    print(f"Results: {success} ok, {fail} skipped/failed, {len(completed)} total done")
    if success > 0:
        print(f"Avg: {total_elapsed/success:.0f}s/subject (wall clock)")
    print(f"Output: {PROCESSED_DIR}")
    for d in sorted(PROCESSED_DIR.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.csv")))
            size = sum(f.stat().st_size for f in d.glob("*.csv"))
            print(f"  {d.name}: {n} files, {size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
