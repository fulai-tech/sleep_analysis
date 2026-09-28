"""RR 间期清洗 — process_rpoint 的公共实现（MESA / SHHS 共用）。

当前决策（2026-08-05）: HRV 相关处理退回原版（hrvanalysis 流程）。

原因:
  1. 实测 causal 版与原版的 HRV 特征差异中位数为 0%（正常心跳下
     前向填充与双向插值输出相同，差异只出现在第一拍和极少数异常拍处），
     对分类性能的影响可忽略；
  2. 原版已训练模型的可比性更重要（换预处理会使新旧模型不可比）；
  3. RRV 特征差异实测超 10%（中位 11.7%），不可忽略，故 RRV 保留因果
     （见 feature_extraction/mesa_datasst/rrv.py）。

如需将来启用因果版 HRV 清洗，取消下方 process_rpoint 中 causal 分支的注释即可
（同时 import sleep_analysis.processing_config）。
"""

import numpy as np
import pandas as pd
from hrvanalysis import interpolate_nan_values, remove_ectopic_beats, remove_outliers

#: 一个 epoch 内至少多少个**真实**（非插值）RR 间期才算 HRV 可用。
#: ✅2026-09-24: 此前是按**总拍数**（含插值补出来的拍）计数的 —— 见 `process_rpoint`
#: 里关于 `interpolated` 列的说明。与 MESA 论文路径的 10 拍门槛同值。
MIN_REAL_RR_PER_EPOCH = 10


# ============================================================================
# 以下为 causal 版 RR 间期清洗实现（当前未启用，仅供将来参考）
#
# 解决的问题 — 原版流程中的非因果操作:
#   1. fillna(整夜均值)  — 首样本/兜底用整夜 RR 均值填充（含未来数据）
#   2. 双向线性插值      — 填补异常拍空隙时使用未来样本
#   3. malik 异位检测    — 比较 rr[i] 与 rr[i+1]（前瞻一拍）
#
# 因果对应:
#   1. 首有效值填充      — 只使用已见数据
#   2. 前向填充 (ffill)  — 只用历史值填补
#   3. 滞后一拍比较      — abs(rr[i-1] - rr[i]) > 0.2 * rr[i-1]，阈值与 malik 相同
#
# 已通过截断不变性验证（t 时刻输出不依赖 t 之后数据）:
#   - 合成 RR 序列截断 50% 后前一半输出与全量完全一致
#   - 与原版差异仅出现在: 第一拍（均值 vs 首有效值, ~12ms）和异常拍处（±1 拍）
# ============================================================================

# def _fill_causal(rr: np.ndarray) -> np.ndarray:
#     """因果填充: 前向填充 + 开头用第一个有效值（不使用任何未来数据）。"""
#     s = pd.Series(rr)
#     s = s.ffill()
#     first_valid = s.dropna()
#     if len(first_valid) > 0:
#         s = s.fillna(first_valid.iloc[0])
#     return s.values


# def _remove_ectopic_causal(rr: np.ndarray) -> np.ndarray:
#     """因果版异位搏动检测: 滞后一拍比较。
#
#     hrvanalysis 的 malik 规则比较 rr[i] 与 rr[i+1]（前瞻一拍）;
#     因果版比较 rr[i-1] 与 rr[i]（滞后一拍，判断当前拍时只用过去数据），
#     判定阈值相同: 差异超过前一拍的 ±20% 视为异位，置 NaN 由前向填充恢复。
#     """
#     rr = np.asarray(rr, dtype=float)
#     out = rr.copy()
#     for i in range(1, len(rr)):
#         if abs(rr[i - 1] - rr[i]) > 0.2 * rr[i - 1]:
#             out[i] = np.nan
#     return out


def process_rpoint(ecg_df: pd.DataFrame) -> pd.DataFrame:
    """R-point → RR 间期 → 去异常 → HR（MESA/SHHS 共用，原版 hrvanalysis 流程）。

    ✅2026-09-24 变更（两条，一起看才完整）:

      1. **新增 `interpolated` 列（0/1）** —— 标出哪些拍结尾的 RR 是
         `interpolate_nan_values` 补出来的。两次插值（`remove_outliers` 后、
         `remove_ectopic_beats` 后）之前各取一次 NaN 掩码再合并。
      2. **epoch 剔除门槛改为按「非插值拍数」计** —— 原实现按总拍数，
         等于让编造的拍凑数。见下方 `MIN_REAL_RR_PER_EPOCH`。

    为什么需要这两条: 线性插值补出来的段是**等差数列** → `SDSD≈0` →
    `SD1=sqrt(SDSD²/2)≈0` → `ratio_sd2_sd1 = SD2/SD1` 发散。实测混进
    402 个 1e13~1e15 的值，把 `StandardScaler` 的 mean/std 拉到 8.2e9/4.8e12，
    **该特征在模型里标准化后恒为常数、完全失效**。且不止 ratio —— 频域
    4 个槽位（vlf/lf/hf/total_power）实测偏 12~35%。

    历史说明（2026-08-05 回退）: 此前实现了 causal 分支（processing_config.causal
    为 True 时用前向填充 + 首有效值 + 滞后一拍异位检测）。因 HRV 特征实测差异为
    0%（中位），且需保持与原版已训练模型的可比性，现无条件走原版。如需启用
    causal 版，取消下方注释并改为:
        if pc.causal:
            clean_rri = remove_outliers(...)
            clean_rri = _remove_ectopic_causal(clean_rri)
            clean_rri = _fill_causal(clean_rri)
        else:
            ...原版...

    消费方: `feature_extraction/mesa_datasst/hrv.py::calc_hrv_features` 会读
    `interpolated` 列并排除 `== 1` 的拍。**旧的特征表（2026-09-24 之前产出的）
    没有这一列，用新代码读会报错** —— 需要重新生成预处理数据。
    """
    ecg_df = ecg_df[ecg_df["TPoint"] > 0].copy()

    rr_intervals = pd.DataFrame(ecg_df["seconds"].diff() * 1000)
    rr_intervals = rr_intervals.rename(columns={"seconds": "RR Intervals"})

    # ---- 原版清洗 (hrvanalysis, 复现作者结果) ----
    rr_intervals["RR Intervals"] = rr_intervals["RR Intervals"].fillna(
        rr_intervals["RR Intervals"].mean()
    )  # fill mean for first sample
    clean_rri = rr_intervals["RR Intervals"].values
    clean_rri = remove_outliers(rr_intervals=clean_rri, low_rri=300, high_rri=2000, verbose=False)
    # ✅2026-09-24: 记下两次插值**之前**的 NaN 位置 —— 那些是被剔除、随后被"补出来"的拍。
    #   补出来的是等差数列（线性插值），拿它算 HRV 会造出伪影：
    #   SDSD≈0 → SD1=sqrt(SDSD²/2)≈0 → ratio_sd2_sd1 发散（实测 1e13~1e15，
    #   把 StandardScaler 的 mean/std 拉到 8e9/5e12，该特征在模型里恒为常数）。
    #   下游 `calc_hrv_features` 会排除这些拍。
    _nan_after_outlier = np.isnan(np.asarray(clean_rri, dtype=float))
    clean_rri = interpolate_nan_values(rr_intervals=clean_rri, interpolation_method="linear")
    clean_rri = remove_ectopic_beats(rr_intervals=clean_rri, method="malik", verbose=False)
    _nan_after_ectopic = np.isnan(np.asarray(clean_rri, dtype=float))
    clean_rri = interpolate_nan_values(rr_intervals=clean_rri)
    rr_intervals["RR Intervals"] = clean_rri
    rr_intervals["RR Intervals"] = rr_intervals["RR Intervals"].fillna(
        rr_intervals["RR Intervals"].mean()
    )  # eventually fill mean for first samples

    # ✅2026-09-24: `interpolated` 列（0/1）—— 1 = 这一拍结尾的 RR 间期是插值补出来的。
    #   随 `ecg_data_clean/*.csv` 落盘，供 `calc_hrv_features` 排除。
    rr_intervals["interpolated"] = (_nan_after_outlier | _nan_after_ectopic).astype(int)

    hr_df = pd.DataFrame(np.round((60000.0 / rr_intervals["RR Intervals"]), 0))
    hr_df = hr_df.rename(columns={"RR Intervals": "HR"})
    ecg_df = pd.concat([ecg_df, hr_df, rr_intervals], axis=1)

    # ✅2026-09-24: epoch 剔除门槛从「**总拍数** < 10」改为「**非插值拍数** < 10」。
    #   理由：插值补出来的拍不是真实测量，拿它凑数等于用一个编造的 epoch 去算 HRV。
    #   改门槛（而不是在 HRV 层兜底）是因为下游 `align_datastreams` 按 epoch 取交集，
    #   上游少一个 epoch 会一致地从 ACT/RRV/EDR 全部消失，行数始终相等；
    #   而 `merge_features` 是按位置 pd.concat(axis=1)，行数不齐会**静默错位**。
    real_count = ecg_df.loc[ecg_df["interpolated"] == 0, "epoch"].value_counts()
    ok_epochs = set(real_count[real_count >= MIN_REAL_RR_PER_EPOCH].index)
    ecg_df = ecg_df[ecg_df["epoch"].isin(ok_epochs)]

    return ecg_df
