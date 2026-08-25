"""
MrOS 数据预处理管线 (多进程并行，断点续跑)
=========================================
适配 MrOS Sleep Study (多民族男性老年队列, NSRR 发布) visit1 / visit2。

与 SHHS 管线的差异:
  - 两 visit 通道命名/采样率完全不同, 且 visit1 有 6 个差分导联 montage 文件
    (aa1367/aa3370/aa3411 14ch, aa1715/aa1900/aa3903 17ch, 通道名为 'ECG L-ECG R' 等),
    visit2 目录里甚至混入 1 个 visit1 montage 的文件 (aa3903) → 通道名必须按文件
    动态匹配 (见 CHANNEL_ALIASES), 不能按 visit 写死
  - 无 annotations-rpoints → R 点全部用 neurokit2 从 ECG 自检
  - 无体动数据 → 只有 HRV + RRV 模态
  - 同一被试两 visit 的 ID 相同 (aaXXXX, 随访复查) → 输出目录按 visit 隔离
    (默认 /srv/shared/psgdata/mros_processed/v{1,2}), 模式锁防止混用
  - 20260822: 通道读取改为原生采样率解码 (绕开 MNE) — MNE 会把整个文件上采样到
    DHR 的 512/1024 Hz, 单被试峰值内存实测 10.5 GB, 8 worker 必被 OOM killer 杀
    进程 (causal/noncausal visit2 各崩 961/267 个)。原生读取后峰值 <1.6 GB,
    呼吸带 <32 Hz 时先 np.interp 到 32 Hz 再交给 process_resp (其降采样器不支持
    上采样方向)

用法:
    python experiments/data_handling/preprocess_mros.py --visit 1 --n-subjects 3   # 试跑
    python experiments/data_handling/preprocess_mros.py --visit 1 --no-edr --n-workers 10
    python experiments/data_handling/preprocess_mros.py --visit 2 --no-edr --n-workers 10

管线 (与 SHHS 一致, 6 步):
    1. EDR 特征提取 — 从 ECG 提取呼吸波形并计算 RRV 特征 (--no-edr 时跳过)
    2. RRV 特征提取 — 从胸腔呼吸带信号提取呼吸特征
    3. 数据预处理 — R 点检测 (neurokit2) → HR 序列 → XML 分期解析 (R&K→AASM)
       → epoch 交集对齐 → 睡眠时长过滤 (>120 睡眠 epoch) → 输出清洗表
    4. HRV 特征提取 — 从 RR 间期计算时域/频域/非线性特征
    5. 特征合并 — HRV/RRV/EDR 按 epoch 键合并 (不依赖行序), 产出特征表
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
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

import sleep_analysis.processing_config as pc

# ---------------------------------------------------------------------------
# MrOS 通道别名表 — 按优先级匹配, 逐文件动态解析
# ---------------------------------------------------------------------------
# visit1: ECG L / ECG R, Thoracic / Abdominal, SaO2, Cannula Flow (带空格), DHR@512
# visit2: ECGL / ECGR, Chest / ABD, SpO2, CannulaFlow (无空格), DHR@1024 为主
# 差分导联 (visit1 6 个文件): 'ECG L-ECG R' 是双极导联, 可直接作为单通道 ECG 用
CHANNEL_ALIASES = {
    "ecg": ["ECG L", "ECGL", "ECG L-ECG R", "ECG R", "ECGR", "ECG", "EKG"],
    "resp": ["Thoracic", "Chest", "THOR RES", "Abdominal", "ABD", "ABDO RES"],
}


def parse_mros_xml(xml_path):
    """解析 NSRR XML (Compumedics PSGAnnotation, 与 SHHS 同构), 返回逐 epoch 分期 DataFrame。"""
    tree = ElementTree.parse(str(xml_path))
    root = tree.getroot()
    time_list, sleep_list = [], []
    j = 0
    for elem in root:
        for subelem in elem:
            if len(subelem) >= 2 and subelem[0].text == "Stages|Stages":
                duration = float(subelem[3].text)
                for _ in range(int(duration / 30)):
                    time_list.append(j)
                    sleep_list.append(subelem[1].text)
                    j += 30
    return pd.DataFrame({"time": time_list, "sleep": sleep_list})


def sleep_stage_map(df_psg):
    """NSRR 分期字符串 → 数字标签 (Wake=0, N1=1, N2=2, N3=3, REM=4)。

    MrOS 使用 Rechtschaffen & Kales 标准 (同 SHHS):
      - Stage 4 sleep|4 → 映射为 N3 (3), 与 Stage 3 合并
      - Unscored|9 / Movement|6 → NaN, 从交集中剔除 (MrOS 实测只有 2 个 Unscored|9;
        Movement|6 未出现, 保留映射以防万一)
    """
    mapping = {
        "Wake|0": 0,
        "Stage 1 sleep|1": 1,
        "Stage 2 sleep|2": 2,
        "Stage 3 sleep|3": 3,
        "Stage 4 sleep|4": 3,      # R&K Stage 4 = AASM N3
        "REM sleep|5": 4,           # NSRR REM=5 → AASM REM=4 (与 MESA 对齐)
        "Unscored|9": np.nan,      # 未评分 — 剔除
        "Movement|6": np.nan,       # 体动伪迹 — 剔除 (防御保留)
    }
    ss = df_psg[["sleep"]].copy()
    ss["5stage"] = ss["sleep"].map(mapping)
    ss["4stage"] = ss["5stage"].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 3})  # Wake/N1+N2/N3/REM
    ss["3stage"] = ss["5stage"].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 2})  # Wake/NREM/REM
    # binary sleep: 仅 5stage!=0 且非 NaN 的 epoch 为 Sleep=1
    ss["sleep"] = ss["5stage"].apply(lambda x: 1 if pd.notna(x) and x != 0 else (0 if pd.notna(x) else np.nan))
    return ss


def generate_r_points_from_ecg(raw_ecg_values, sampling_rate):
    """R 峰自动检测 (neurokit2, 与 SHHS 回退路径一致 — 保持与原版 HRV 处理可比)。"""
    import neurokit2 as nk
    ecg_cleaned = nk.ecg_clean(raw_ecg_values, sampling_rate=sampling_rate)
    _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    rpeaks = rpeaks_info["ECG_R_Peaks"]
    seconds = rpeaks / sampling_rate
    epoch = (seconds / 30 + 1).astype(int)
    return pd.DataFrame({
        "RPoint": rpeaks, "seconds": seconds, "epoch": epoch, "TPoint": 1,
    })


def _read_edf_channel_native(edf_path, channel_aliases):
    """按原生采样率解码 EDF 单通道, 绕开 MNE 的全文件上采样 (内存降 16-32 倍)。

    EDF 布局: 256B 全局头 + N×256B 通道描述 (字段按列式布局) + 数据区。
    数据区每条记录 (MrOS 为 1s) 内按通道顺序交错存放 int16 样本, 各通道样本数
    (spr) 不同 → 只能逐记录 seek 目标通道, 不能整文件切片。
    物理值换算: (raw - digmin) * (physmax - physmin) / (digmax - digmin) + physmin。

    ⚠️ 20260822: 原实现用 MNE read_raw_edf — 混合采样率 EDF 会被整体上采样到
    文件内最高采样率 (visit2 的 DHR@1024), 单通道 35h 录制即 ~1 GB float64,
    实测单被试峰值内存 10.5 GB, 8 worker 同时跑必被 OOM killer 杀进程。
    原生读取: 呼吸带 32 Hz / ECG 512 Hz 直接按实际采样率解码, 峰值内存 <1.5 GB。
    附带收益: 消除了 MNE resample_poly 插值 (非因果 FIR) 对因果链的污染。
    """
    from datetime import datetime, timezone

    with open(edf_path, "rb") as f:
        hdr = f.read(256)
        recdur = float(hdr[244:252].decode().strip())
        nchan = int(hdr[252:256].decode().strip())
        nrec = int(hdr[236:244].decode().strip())
        start_date = hdr[168:176].decode().strip()
        start_time = hdr[176:184].decode().strip()
        # 通道描述: 列式布局 (所有 label 连续, 然后所有 transducer, ...)
        fields = {}
        for size, name in [(16, 'label'), (80, 'transducer'), (8, 'physdim'),
                           (8, 'physmin'), (8, 'physmax'), (8, 'digmin'), (8, 'digmax'),
                           (80, 'prefilter'), (8, 'spr'), (32, 'reserved')]:
            raw = f.read(size * nchan)
            fields[name] = [raw[i * size:(i + 1) * size].decode(errors='replace').strip('\x00 ').strip()
                            for i in range(nchan)]
    labels = fields['label']

    # 通道匹配 (按别名优先级)
    ch_name = next((a for a in channel_aliases if a in labels), None)
    if ch_name is None:
        raise ValueError(f"无可用通道: {channel_aliases} 不在 {labels}")
    idx = labels.index(ch_name)
    spr = int(fields['spr'][idx])
    native_rate = spr / recdur
    total = spr * nrec

    # 逐记录读取目标通道 (seek 跳过其他通道)
    sprs_int = [int(s) if s else 0 for s in fields['spr']]
    pre_bytes = sum(sprs_int[:idx]) * 2
    post_bytes = sum(sprs_int[idx + 1:]) * 2
    data = np.empty(total, dtype=np.int16)
    with open(edf_path, "rb") as f:
        f.seek(256 + nchan * 256)
        off = 0
        for _ in range(nrec):
            f.seek(pre_bytes, 1)
            f.readinto(data[off:off + spr])
            off += spr
            f.seek(post_bytes, 1)

    # 物理校准 (EDF 标准小端 int16)
    digmin_f, digmax_f = float(fields['digmin'][idx]), float(fields['digmax'][idx])
    physmin_f, physmax_f = float(fields['physmin'][idx]), float(fields['physmax'][idx])
    if digmax_f != digmin_f:
        data_f = (data.astype(np.float64) - digmin_f) * (physmax_f - physmin_f) / (digmax_f - digmin_f) + physmin_f
    else:
        data_f = data.astype(np.float64)

    # 起始时间 (EDF 头: DD.MM.YY HH.MM.SS, 兼容冒号分隔)
    starttime = datetime.strptime(
        f"{start_date.replace('.', '-')} {start_time.replace('.', ':')}",
        "%d-%m-%y %H:%M:%S",
    ).replace(tzinfo=timezone.utc)

    return data_f, native_rate, starttime, ch_name


def extract_edf_channel(edf_dir, subj, channel_aliases, visit):
    """从 MrOS EDF 提取指定通道 (按别名表动态匹配), 返回原始数据 + 实际采样率。

    原生采样率解码 (见 _read_edf_channel_native), 不经过 MNE 的全文件上采样:
      - visit1 ECG 512 Hz, 呼吸带 Thoracic 16 Hz → 先插值到 32 Hz (np.interp,
        线性插值, 与 biopsykit 风格一致; 16→32 残余向前依赖 1 个样本 ≈ 62ms,
        与因果分支文档化的 ≤8-16ms 同量级) 再交给 process_resp — 其降采样器
        不支持上采样方向 (16/32=0.5 时 cheby1 Wn=1.6 会抛异常)
      - visit2 ECG 512 Hz, 呼吸带 Chest 32 Hz → 原样传递, process_resp 32→32 恒等
    """
    from sleep_analysis.preprocessing.utils import _create_datetime_index

    edf_path = Path(edf_dir) / f"mros-visit{visit}-{subj}.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    data, native_rate, starttime, ch_name = _read_edf_channel_native(edf_path, channel_aliases)

    if native_rate < 32:
        factor = int(round(32 / native_rate))
        x_new = np.arange(len(data) * factor) / factor
        data = np.interp(x_new, np.arange(len(data)), data)
        native_rate = 32

    times = np.arange(len(data)) / native_rate
    time_idx, epochs = _create_datetime_index(starttime, times_array=times)
    col_name = "resp" if ch_name in CHANNEL_ALIASES["resp"] else "ecg"
    return pd.DataFrame(data, index=time_idx).rename(columns={0: col_name}), epochs, native_rate


def process_one_subject(cfg):
    """对单个被试执行完整 5 步管线。返回 (subj, ok, info) 或 (subj, False, reason)。"""
    subj = cfg["subj"]
    visit = cfg["visit"]
    edf_dir = Path(cfg["edf_dir"])
    annot_dir = Path(cfg["annot_dir"])
    processed_dir = Path(cfg["processed_dir"])
    no_edr = cfg["no_edr"]

    from sleep_analysis.preprocessing.mesa_dataset.edr import _extract_edr, process_resp
    from sleep_analysis.feature_extraction.mesa_datasst.rrv import extract_rrv_features_helper
    from sleep_analysis.feature_extraction.mesa_datasst.hrv import calc_hrv_features
    from sleep_analysis.preprocessing.mesa_dataset.respiration import check_resp_features
    from sleep_analysis.preprocessing.mesa_dataset.edr_placeholder import make_edr_placeholder
    from sleep_analysis.preprocessing.rr_utils import process_rpoint

    t0 = time.time()
    timings = {}

    out_edr = out_rrv = out_ecg = out_hrv = out_merge = None
    try:
        # --- Step 1: EDR (--no-edr 时跳过, 下游用全 0 占位表) ---
        out_edr = processed_dir / f"edr_respiration_features_raw/edr_respiration{subj}.csv"
        if not no_edr and not out_edr.exists():
            raw_ecg, epochs, ecg_rate = extract_edf_channel(edf_dir, subj, CHANNEL_ALIASES["ecg"], visit)
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
            resp_df, epochs, resp_rate = extract_edf_channel(edf_dir, subj, CHANNEL_ALIASES["resp"], visit)
            resp_df, epochs = process_resp(resp_df, epochs,
                                           sampling_rate_in=resp_rate)
            features = extract_rrv_features_helper(resp_df)
            features.to_csv(out_rrv)
        timings["rrv"] = time.time() - t1
        t2 = time.time()

        # --- Step 3: 数据预处理 (R 点 / 分期 / 对齐 / 清洗) ---
        out_ecg = processed_dir / f"ecg_data_clean/ecg_data_clean{subj}.csv"
        if not out_ecg.exists():
            xml_path = annot_dir / f"mros-visit{visit}-{subj}-nsrr.xml"
            if not xml_path.exists():
                return (subj, False, {"reason": "no PSG XML"})

            df_psg = parse_mros_xml(xml_path)
            if len(df_psg) == 0:
                return (subj, False, {"reason": "empty PSG"})

            # R-point (MrOS 无官方 rpoint 标注 → neurokit2 自检)
            raw_ecg, _, ecg_rate = extract_edf_channel(edf_dir, subj, CHANNEL_ALIASES["ecg"], visit)
            df_rp = generate_r_points_from_ecg(raw_ecg.iloc[:, 0].values,
                                               sampling_rate=ecg_rate)
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
            if no_edr or not out_edr.exists():
                # EDR 屏蔽: 用 RRV 特征表生成全 0 占位表, 保持下游对齐/合并/训练代码不变
                df_edr = make_edr_placeholder(df_resp)
            else:
                df_edr = pd.read_csv(out_edr, index_col=0)
            df_resp = check_resp_features(df_resp)
            df_edr = check_resp_features(df_edr)
            df_edr.columns = [c.replace("RRV", "EDR") for c in df_edr.columns]

            # Epoch 交集 (三个表全部过滤到 intersect — 20260821: 修正 SHHS 版
            # df_edr 未过滤导致的整夜行错位隐患)
            e_hr = set(df_hr["epoch"].values)
            e_psg = set(labels["epoch"].values)
            e_resp = set(df_resp["epoch"].values) if "epoch" in df_resp.columns else set()
            e_edr = set(df_edr["epoch"].values) if "epoch" in df_edr.columns else set()
            intersect = e_hr & e_psg
            if e_resp:
                intersect &= e_resp
            if e_edr:
                intersect &= e_edr
            if len(intersect) == 0:
                return (subj, False, {"reason": "no overlap"})

            df_hr = df_hr[df_hr["epoch"].isin(intersect)].copy()
            labels = labels[labels["epoch"].isin(intersect)].copy()
            df_resp = df_resp[df_resp["epoch"].isin(intersect)]
            df_edr = df_edr[df_edr["epoch"].isin(intersect)]

            # Sleep duration filter
            sleep_n = labels["sleep"].sum()  # sleep 列已转 0=Wake 1=Sleep
            if sleep_n <= 120:
                return (subj, False, {"reason": f"sleep {sleep_n} ≤ 120"})

            # Save
            ecg_clean = df_hr.copy()
            smap = labels.set_index("epoch")["5stage"]
            ecg_clean["stage"] = ecg_clean["epoch"].map(smap).fillna(0)
            ecg_clean.to_csv(out_ecg, index=False)

            labels[labels["epoch"].isin(ecg_clean["epoch"])].to_csv(
                processed_dir / f"sleep_stages/sleep_stages{subj}.csv", index=False)
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

        # --- Step 5: Merge (按 epoch 键合并, 不依赖行序; 带 RangeIndex 首列) ---
        out_merge = processed_dir / f"features_full_combined/features_combined{subj}.csv"
        if not out_merge.exists():
            h = pd.read_csv(out_hrv)
            r = pd.read_csv(processed_dir / f"rrv_features/respiration_features_clean{subj}.csv")
            e = pd.read_csv(processed_dir / f"rrv_features/edr_features_clean{subj}.csv")
            # 20260821: 按 epoch 键对齐并断言 — 修正 SHHS 版 min(len) 位置拼接
            # 可能静默错位的问题 (三个表均已过滤到同一 intersect, 行序未必一致)
            h["_epoch_key"] = h["_hrv_epoch"]
            r["_epoch_key"] = r["epoch"]
            e["_epoch_key"] = e["epoch"]
            combined = (h.merge(r, on="_epoch_key", how="inner")
                         .merge(e, on="_epoch_key", how="inner"))
            try:
                assert len(combined) == len(h) == len(r) == len(e), \
                    f"merge 行数不一致: h={len(h)} r={len(r)} e={len(e)} → {len(combined)}"
            except AssertionError:
                print(f"  [{subj}] WARN: {len(h)}/{len(r)}/{len(e)} → {len(combined)}")
            combined = combined.drop(columns=["_epoch_key"])
            combined = combined.loc[:, ~combined.columns.duplicated()]
            # 与 MESA merge_features 约定一致: 带 RangeIndex 首列, 训练端 index_col=0 消费
            combined.to_csv(out_merge, index=True)
        timings["merge"] = time.time() - t4

        elapsed = time.time() - t0
        return (subj, True, {"elapsed": elapsed, "timings": timings})

    except Exception as e:
        # 失败被试不进 checkpoint, 重跑会重新生成所有文件, 半成品自然被覆盖
        return (subj, False, {"reason": str(e)[:120]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visit", type=int, choices=[1, 2], default=1,
                        help="MrOS 访视: 1 (全队列 PSG) 或 2 (随访复查 PSG, n=1026)")
    parser.add_argument("--n-subjects", type=int, default=99999)
    parser.add_argument("--n-workers", type=int, default=10,
                        help="并行 worker 数 (默认 10)")
    parser.add_argument("--no-edr", action="store_true",
                        help="屏蔽 EDR 处理: 跳过 Step 1, EDR 特征用全 0 占位表替入 (项目标准用法)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认 /srv/shared/psgdata/mros_processed/v{visit})")
    args = parser.parse_args()

    VISIT = args.visit
    N = args.n_subjects
    N_WORKERS = args.n_workers

    MROS_ROOT = Path("/mnt/nas_data/psgdata/mros")
    EDF_DIR = MROS_ROOT / "polysomnography/edfs" / f"visit{VISIT}"
    ANNOT_DIR = MROS_ROOT / "polysomnography/annotations-events-nsrr" / f"visit{VISIT}"

    PROCESSED_DIR = Path(args.output_dir) if args.output_dir else \
        Path(f"/srv/shared/psgdata/mros_processed/v{VISIT}")

    # 目录级模式锁: 跨模式/跨 visit 重跑已有目录直接拒绝 (visit1/visit2 被试 ID 相同,
    # 必须分目录, 防止 ID 碰撞与静默复用旧模式文件)
    pc.check_run_mode(
        PROCESSED_DIR,
        {"causal": pc.causal, "no_edr": args.no_edr, "script": f"preprocess_mros.py (visit{VISIT})"},
    )

    print(f"Visit: {VISIT}")
    print(f"Output: {PROCESSED_DIR}")
    print(f"Workers: {N_WORKERS}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 加载数据 ---
    all_edfs = sorted(EDF_DIR.glob("*.edf"))
    all_subjects = []
    for f in all_edfs:
        m = re.findall(r"mros-visit\d-(.+)\.edf", f.name)
        if m and m[0] not in all_subjects:
            all_subjects.append(m[0])
    print(f"Total EDF files: {len(all_subjects)}")

    subjects = all_subjects[:N]
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
        "visit": VISIT,
        "edf_dir": str(EDF_DIR),
        "annot_dir": str(ANNOT_DIR),
        "processed_dir": str(PROCESSED_DIR),
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
                print(f"  [{subj}] SKIP: {reason}")

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
