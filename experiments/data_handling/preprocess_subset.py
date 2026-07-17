"""
MESA 数据预处理管线 — 按指定被试数量运行
=========================================
与同目录下 data_handling.py 功能完全一致（EDR 提取 → RRV 提取 → 清洗对齐 → 特征提取 → 合并），
唯一区别是可以通过命令行参数限制处理的被试数量，而不是默认跑全量。

用法:
    python experiments/data_handling/preprocess_subset.py 5    # 只处理前5个被试
    python experiments/data_handling/preprocess_subset.py 50   # 50个被试（约750 MB）
    python experiments/data_handling/preprocess_subset.py 2056 # 全量（约15-20 GB）

预处理管线（6 步）:
    1. EDR 特征提取 — 从 ECG 心电信号提取呼吸波形，计算 RRV 特征
    2. RRV 特征提取 — 从 EDF 胸腔呼吸带信号提取真实呼吸特征
    3. MESA 数据预处理 — 加载体动/ECG/PSG/呼吸数据，清洗、时间对齐、睡眠分期标注转换
    4. 体动特征提取 — 对活动计数序列用不同窗口大小计算统计量，产出 370 维特征
    5. HRV 特征提取 — 从 RR 间期计算时域/频域/非线性心率变异性特征，产出 31 维
    6. 特征合并 — 将 ACT/HRV/RRV/EDR 四模态特征按 epoch 对齐，产出 460 维特征表

输入路径（在 study_data.json 中配置）:
    - mesa_path: MESA 原始数据目录 (EDF、体动 CSV、PSG XML 标注等)
    - mesa_path_edf: EDF 文件目录
    - processed_mesa_path: 预处理产出目录

输出:
    处理后的特征文件写入 processed_mesa_path 下的各子目录。
    具体结构详见文档末尾的 Step 6 输出部分。
"""

import sys
import re
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 参数解析 & 被试选择
# ---------------------------------------------------------------------------

# 命令行参数: 被试数量，默认为 5
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

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
print(f"Selected {len(subjects)} subjects: {subjects}")

# 处理后的数据统一存放在这个目录下
processed_path = Path(cfg["processed_mesa_path"])

# ---------------------------------------------------------------------------
# Step 1: EDR 特征提取 (ECG-Derived Respiration)
# ---------------------------------------------------------------------------
# 目的: 从 ECG 心电信号中提取呼吸信号 (EDR)，然后计算呼吸率变异性特征。
# 流程: EDF读ECG通道 → EDR算法提取呼吸波形 → 下采样到32Hz → 计算RRV特征
# 输入: 原始 EDF 的 EKG 通道 (256Hz)
# 输出: edr_respiration_features_raw/edr_respiration{ID}.csv
# ---------------------------------------------------------------------------

print("\n=== Step 1: EDR features (ECG-derived respiration) ===")

# EDR 提取核心函数
from sleep_analysis.preprocessing.mesa_dataset.edr import _extract_edr, process_resp
# RRV 特征计算 (EDR 提取出的呼吸信号和真实呼吸信号使用同一套特征算法)
from sleep_analysis.feature_extraction.mesa_datasst.rrv import extract_rrv_features_helper
# 从 EDF 文件中读取指定通道的数据
from sleep_analysis.preprocessing.utils import extract_edf_channel
import tqdm

for subj in tqdm.tqdm(subjects, desc="EDR"):
    out = processed_path / f"edr_respiration_features_raw/edr_respiration{subj}.csv"
    # 支持断点续跑: 如果输出已存在则跳过
    if out.exists():
        print(f"  {subj}: skip (exists)")
        continue
    try:
        # (1) 从 EDF 读取 ECG 通道原始数据 (256Hz, 一整夜的信号)
        raw_ecg, epochs = extract_edf_channel(edf_dir, subj_id=int(subj), channel="EKG")
        # (2) 用 Charlton 算法从 ECG 提取呼吸波形 (EDR)
        edr_signal = _extract_edr(raw_ecg, sampling_rate=256)
        # (3) 下采样到 32Hz，按 epoch 切分
        resp_df, epochs = process_resp(edr_signal.respiratory_signal, epochs)
        # (4) 计算 RRV 特征 (5/7/9分钟滑动窗口，总计~70维)
        features = extract_rrv_features_helper(resp_df, nan_pad=0.0, sampling_rate=32)
        features.to_csv(out)
        print(f"  {subj}: done ({features.shape[0]} epochs)")
    except Exception as e:
        print(f"  {subj}: FAILED - {e}")

# ---------------------------------------------------------------------------
# Step 2: RRV 特征提取 (Respiratory Rate Variability)
# ---------------------------------------------------------------------------
# 目的: 从 EDF 的胸腔呼吸带信号中提取呼吸率变异性特征。
# 流程: EDF读Thor通道 → 下采样 → 提取峰值 → 计算RRV特征
# 与 Step 1 的区别: Step 1 用 ECG 推导呼吸，Step 2 用真实呼吸带信号
# 输入: 原始 EDF 的 Thor 通道 (胸腔呼吸带, 256Hz)
# 输出: respiration_features_raw/respiration{ID}.csv
# ---------------------------------------------------------------------------

print("\n=== Step 2: RRV features (respiration from EDF) ===")

for subj in tqdm.tqdm(subjects, desc="RRV"):
    out = processed_path / f"respiration_features_raw/respiration{subj}.csv"
    if out.exists():
        print(f"  {subj}: skip (exists)")
        continue
    try:
        # (1) 从 EDF 读取 Thor (胸腔呼吸带) 通道
        resp_df, epochs = extract_edf_channel(edf_dir, subj_id=int(subj), channel="Thor")
        # (2) 下采样到 32Hz，按 epoch 切分
        resp_df, epochs = process_resp(resp_df, epochs)
        # (3) 计算 RRV 特征 (与 Step 1 相同的算法)
        features = extract_rrv_features_helper(resp_df)
        features.to_csv(out)
        print(f"  {subj}: done ({features.shape[0]} epochs)")
    except Exception as e:
        print(f"  {subj}: FAILED - {e}")

# ---------------------------------------------------------------------------
# Step 3: MESA 数据预处理
# ---------------------------------------------------------------------------
# 目的: 将被试的多个数据流 (体动、ECG、PSG睡眠分期、呼吸、EDR) 进行:
#   1. 清洗 (去异常值、插值缺失)
#   2. 时间对齐 (所有信号对齐到相同的30秒epoch)
#   3. 睡眠分期标注 (AASM 6-stage → 5stage/4stage/3stage/binary)
#   4. 质量筛选 (排除数据不完整或睡眠时长<2小时的被试)
# 输入: 原始 MESA 数据 + Step 1-2 产生的呼吸/EDR特征
# 输出: actigraph_data_clean/ 和 ecg_data_clean/ (对齐后的清洗数据)
# ---------------------------------------------------------------------------

print("\n=== Step 3: Preprocess MESA ===")

# mesa_data_importer: 读取 MESA 原始数据格式 (体动CSV、R-point CSV、PSG XML)
import mesa_data_importer as importer
import numpy as np
import pandas as pd

# 各预处理子模块
from sleep_analysis.preprocessing.mesa_dataset.ecg import process_rpoint               # R点→RR间期→HR
from sleep_analysis.preprocessing.mesa_dataset.actigraphy import process_actigraphy    # 体动数据清洗裁剪
from sleep_analysis.preprocessing.mesa_dataset.ground_truth import sleep_stage_convert_binary  # PSG分期转换
from sleep_analysis.preprocessing.mesa_dataset.respiration import check_resp_features  # 呼吸特征校验
from sleep_analysis.preprocessing.mesa_dataset.utils import (
    align_datastreams,          # 多数据流时间对齐
    clean_data_to_csv,          # 清洗结果写入CSV
    match_exclusion_criteria,   # 根据数据质量排除被试
)
from sleep_analysis.preprocessing.mesa_dataset.preprocess_mesa import _clean_data_helper  # 单个被试的完整清洗管线

# PSG-体动重叠时间表 (记录了每个被试 PSG 和体动数据同时可用的时间段)
mesa_path = Path(cfg["mesa_path"])
overlap = pd.read_csv(mesa_path / "overlap/mesa-actigraphy-psg-overlap.csv")
# 数据集信息表 (被试人口学信息，用于质量筛选)
dataset_info = pd.read_csv(mesa_path / "datasets/mesa-sleep-dataset-0.5.0.csv").set_index("mesaid")

valid = []   # 记录预处理成功的被试
for subj in tqdm.tqdm(subjects, desc="Preprocess"):
    try:
        # 根据人口学数据排除质量不达标的被试
        if match_exclusion_criteria(dataset_info, subj):
            print(f"  {subj}: excluded (quality)")
            continue

        # 加载该被试的全部数据流
        df_act = importer.load_single_actigraphy(mesa_path, int(subj))       # 7天体动数据
        df_rpt = importer.load_single_r_point(mesa_path, int(subj))          # ECG R点标注
        df_psg = importer.load_single_psg(mesa_path, int(subj))             # PSG 睡眠分期
        df_resp = importer.load_single_resp_features(processed_path, int(subj))  # Step 2 产出
        df_edr = importer.load_single_edr_feature(processed_path, int(subj))     # Step 1 产出

        # 调用原始管线中的核心函数: 清洗→对齐→标注→写入
        _clean_data_helper(df_act, df_rpt, df_psg, df_resp, df_edr, overlap, int(subj))
        valid.append(subj)
        print(f"  {subj}: done")
    except Exception as e:
        print(f"  {subj}: FAILED - {e}")

print(f"\n{len(valid)}/{len(subjects)} subjects preprocessed successfully")

if not valid:
    print("No subjects processed, stopping.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: 体动特征提取 (Actigraphy Features)
# ---------------------------------------------------------------------------
# 目的: 从清洗后的体动数据中提取时序特征。
# 算法: 对活动计数序列用不同窗口大小 (1~20 epoch) 计算:
#   - 统计量: mean, median, std, max, min, var, skew, kurt
#   - 衍生量: 是否非零活动 (nat), 是否有任何活动 (anyact)
#   - 变换量: log(activity+1)
#   - 每种统计量有 centered (窗口居中) 和 non-centered (窗口靠前) 两种模式
# 输出: 每个被试 ~1260 epochs × 370 特征的 DataFrame
# 输入: Step 3 产出的 actigraph_data_clean/
# 输出: actigraph_features/actigraph_features{ID}.csv
# ---------------------------------------------------------------------------

print("\n=== Step 4: Actigraphy features ===")
from sleep_analysis.feature_extraction.mesa_datasst.actigraphy import calc_actigraph_features

for subj in tqdm.tqdm(valid, desc="Actigraphy"):
    try:
        # 读取 Step 3 产出的清洗后体动数据
        act = pd.read_csv(processed_path / f"actigraph_data_clean/actigraph_data_clean{subj}.csv")
        # 仅取 activity 列，计算 370 维时序特征
        feats = calc_actigraph_features(act["activity"])
        feats.to_csv(processed_path / f"actigraph_features/actigraph_features{subj}.csv", index=False)
    except Exception as e:
        print(f"  {subj}: FAILED - {e}")

# ---------------------------------------------------------------------------
# Step 5: HRV 特征提取 (Heart Rate Variability)
# ---------------------------------------------------------------------------
# 目的: 从清洗后的 RR 间期数据中提取心率变异性特征。
# 算法: 使用 hrvanalysis 库，对每个 epoch 计算:
#   - 时域: mean_nni, sdnn, rmssd, pnn50 等
#   - 频域: LF, HF, LF/HF ratio 等 (用 Welch 法做功率谱)
#   - 非线性: Poincare plot 参数 (SD1, SD2), CSI, CVI
#   - 几何: triangular_index
# 输出: 每个被试 ~1260 epochs × 31 特征的 DataFrame
# 输入: Step 3 产出的 ecg_data_clean/
# 输出: hrv_features/hrv_features{ID}.csv
# ---------------------------------------------------------------------------

print("\n=== Step 5: HRV features ===")
from sleep_analysis.feature_extraction.mesa_datasst.hrv import calc_hrv_features

for subj in tqdm.tqdm(valid, desc="HRV"):
    try:
        # 读取 Step 3 产出的清洗后 ECG 数据 (含 RR 间期)
        hr = pd.read_csv(processed_path / f"ecg_data_clean/ecg_data_clean{subj}.csv")
        # 计算 31 维 HRV 特征 (时域+频域+非线性+几何)
        feats = calc_hrv_features(hr)
        feats.to_csv(processed_path / f"hrv_features/hrv_features{subj}.csv", index=False)
    except Exception as e:
        print(f"  {subj}: FAILED - {e}")

# ---------------------------------------------------------------------------
# Step 6: 特征合并 (Merge Features)
# ---------------------------------------------------------------------------
# 目的: 将各模态特征按 epoch 对齐合并为一个统一特征表。
# 合并内容:
#   - 体动特征 (ACT): 370 维
#   - 心率变异性特征 (HRV): ~29 维 (去掉 epoch 列和 tinn 列)
#   - 呼吸特征 (RRV): ~62 维 (5/7/9分钟窗口)
#   - EDR 特征: 同 RRV 维度，列名加 EDR 前缀
# 最终: 每个被试 ~1260 epochs × ~460 维特征的 DataFrame
# 输入: Step 4-5 产出的 actigraph_features/, hrv_features/ 以及
#        Step 3 产出的 respiration_features_clean/, edr_features_clean/
# 输出: features_full_combined/features_combined{ID}.csv
# ---------------------------------------------------------------------------

print("\n=== Step 6: Merge features ===")
from sleep_analysis.feature_extraction.mesa_datasst.utils import merge_features
merge_features(overwrite=True)

# ---------------------------------------------------------------------------
# 完成
# ---------------------------------------------------------------------------

print(f"\n=== Test run complete: {len(valid)} subjects ===")
print(f"Output: {processed_path}")
print(f"\n各步骤产出目录:")
for d in ["actigraph_data_clean", "ecg_data_clean",
          "respiration_features_raw", "edr_respiration_features_raw",
          "actigraph_features", "hrv_features", "features_full_combined"]:
    full = processed_path / d
    count = len(list(full.glob("*.csv"))) if full.exists() else 0
    print(f"  {d}: {count} files")
