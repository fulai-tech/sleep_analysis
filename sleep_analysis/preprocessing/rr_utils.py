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
    clean_rri = interpolate_nan_values(rr_intervals=clean_rri, interpolation_method="linear")
    clean_rri = remove_ectopic_beats(rr_intervals=clean_rri, method="malik", verbose=False)
    clean_rri = interpolate_nan_values(rr_intervals=clean_rri)
    rr_intervals["RR Intervals"] = clean_rri
    rr_intervals["RR Intervals"] = rr_intervals["RR Intervals"].fillna(
        rr_intervals["RR Intervals"].mean()
    )  # eventually fill mean for first samples

    hr_df = pd.DataFrame(np.round((60000.0 / rr_intervals["RR Intervals"]), 0))
    hr_df = hr_df.rename(columns={"RR Intervals": "HR"})
    ecg_df = pd.concat([ecg_df, hr_df, rr_intervals], axis=1)

    # filter RRI: 每 epoch 少于 10 个 RR 样本的 epoch 整体剔除
    t1 = ecg_df.epoch.value_counts().reset_index()
    t1.columns = ["epoch_idx", "count"]
    invalid_idx = set(t1[t1["count"] < 10]["epoch_idx"].values)
    ecg_df = ecg_df[~ecg_df["epoch"].isin(list(invalid_idx))]

    return ecg_df
