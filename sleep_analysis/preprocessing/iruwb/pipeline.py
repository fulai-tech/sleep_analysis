"""IR-UWB 20 Hz 雷达信号 → 特征表 的管线编排。

三条支路（与 MESA / D04 的模态划分一致, 保证 13 个特征槽位可比）。

**所有特征提取函数都是 ``preprocessing/iruwb/`` 内的自包含副本**（``actigraphy.py``
/ ``hrv.py`` / ``rrv.py``）, 不 import MESA/D04 的模块 —— 本管线与那两者在硬件和
数据格式上都无关, 只在**特征列名与语义**上对齐（迁移权重的前提）。

    ACT  体动      signal → extract_movement → calc_actigraph_features   (370 维)
    HRV  心率变异  signal → detect_beats     → 两套口径并列产出          (150 维)
                                                 ├ per-epoch  _hrv_*        ( 30)
                                                 └ windowed   150_hrv_* 等  (120)
    RRV  呼吸变异  signal → extract_rrv_features_helper                  ( 60 维)

训练实际只取其中 13 维，其余列一并保留以便将来调整特征选择而无需重跑预处理。

**13 个训练槽位 —— 对齐 MESA 口径**（HRV 用逐 30 s epoch 的实现, 列名无窗口前缀）::

    ACT  _acc_mean_1
    HRV  _hrv_median_nni, _hrv_ratio_sd2_sd1, 150_hrv_median_nni,
         _hrv_vlf, _hrv_lf, _hrv_hf, _hrv_lf_hf_ratio, _hrv_total_power
    RRV  150_RRV_MedianBB, 150_RRV_LF, 270_RRV_MCVBB, 150_RRV_CVBB

其中 HRV 槽位 3 用 ``150_hrv_median_nni``（150 s 窗口）, 而非 MESA 原表里那个
重复的 ``_hrv_median_nni``——MESA 的 HRV 列表第 1 与第 3 项历史上是同一个列名。

因果性
------
由 ``processing_config.causal``（环境变量 ``SLEEP_CAUSAL``）统一驱动，三条支路
各自有因果/非因果分支:

    ACT: expanding min/max + expanding mean/std  vs  整文件 min/max + StandardScaler
    HRV: 左对齐回顾窗口 [t+30-W, t+30]           vs  向前看窗口 [t, t+W]
    RRV: 左对齐回顾窗口                           vs  居中（含未来）窗口

⚠️ 两个模式产出的特征**不能混用**，必须用不同的输出目录（``check_run_mode`` 会拦）。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import sleep_analysis.processing_config as pc
from sleep_analysis.preprocessing.iruwb.actigraphy import calc_actigraph_features
from sleep_analysis.preprocessing.iruwb.beat_detection import BeatDetectionResult, detect_beats
from sleep_analysis.preprocessing.iruwb.hrv import (
    MIN_RR_PER_EPOCH, get_hrv_features, get_hrv_features_per_epoch,
)
from sleep_analysis.preprocessing.iruwb.movement import extract_movement
from sleep_analysis.preprocessing.iruwb.rrv import extract_rrv_features_helper

EPOCH_S = 30.0


@dataclass
class PipelineDiagnostics:
    """各支路的中间量, 供验证脚本检查（不进入特征表）。"""

    beat_result: BeatDetectionResult
    movement: pd.DataFrame
    resp_features: pd.DataFrame
    hrv_features: pd.DataFrame
    epoch_index: pd.DatetimeIndex


def fill_signal_nans(signal: np.ndarray):
    """线性插值补齐输入信号里的 NaN，返回 ``(补齐后的信号, 填补数量)``。

    真实采集丢一个采样点就会产生 NaN。若不处理，下游的 IIR 滤波
    （Butterworth，递归）会把 NaN 扩散到其后**所有**样本；非因果分支的
    ``filtfilt``（前向 + 反向两遍）更是**前后两个方向都扩散** —— 实测注入
    单个 NaN 就能让整夜检出 0 拍。

    ``pandas.interpolate`` 对首尾 NaN 用最近有效值外推；若整段无有效值则抛错。
    """
    signal = np.asarray(signal, dtype=float).ravel()
    n_nan = int(np.isnan(signal).sum())
    if n_nan == 0:
        return signal, 0
    filled = pd.Series(signal).interpolate(limit_direction="both").to_numpy()
    if np.isnan(filled).any():
        raise ValueError(f"输入信号含 {n_nan} 个 NaN 且无有效样本可插值")
    return filled, n_nan


def validate_blocks(features: pd.DataFrame) -> None:
    """校验三条支路各自产出了预期数量的列，否则抛错。

    **不做这一步的话，退化输入会让整块特征静默消失。** 实测：呼吸信号无峰时
    （恒定/斜坡），RRV 三个窗口全部抛异常，``dict.fromkeys([], 0)`` 产出空 dict，
    最终 RRV 的 60 列**一列不剩** —— 特征表从 581 列变 521 列，而
    ``process_signal`` 照常返回，不报任何错。下游按列名取特征时才 KeyError。

    同样的机制也会让 HRV 逐 epoch 块（30 列）在无可用心搏时消失。
    """
    groups = [
        ("ACT", [c for c in features.columns if c.startswith("_acc")], 370),
        ("HRV(逐epoch)", [c for c in features.columns if c.startswith("_hrv")], 30),
        ("RRV", [c for c in features.columns if "RRV" in c], 60),
    ]
    bad = [(n, len(c), e) for n, c, e in groups if len(c) != e]
    if bad:
        detail = "; ".join(f"{n} 产出 {got} 列/预期 {exp} 列" for n, got, exp in bad)
        raise ValueError(
            f"特征支路产出异常（{detail}）—— 输入信号可能退化"
            f"（呼吸无峰 / 无可用心搏）。请检查上游信号质量。"
        )


def make_epoch_index(start_time, n_samples: int, fs: float) -> pd.DatetimeIndex:
    """构造与信号等长的采样级时间轴, 以及对齐到 30 s 的 epoch 轴。"""
    t = pd.Timestamp(start_time).floor("30s") + pd.to_timedelta(
        np.arange(n_samples) / fs, unit="s"
    )
    return pd.DatetimeIndex(t)


def epoch_axis(sample_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """由采样级时间轴推出 30 s epoch 轴。"""
    return pd.DatetimeIndex(sample_index.floor("30s").unique())


def process_signal(signal: np.ndarray, fs: float, start_time,
                   min_beats_per_epoch: int = MIN_RR_PER_EPOCH,
                   max_bad_epoch_frac: float = 0.2,
                   thr_resp_variability_warn: bool = True) -> tuple:
    """跑完整管线, 返回 (特征表, 诊断信息)。

    Parameters
    ----------
    signal : np.ndarray
        单通道 20 Hz 混叠生理信号（胸腔位移量级）。
    fs : float
        采样率。
    start_time : datetime-like
        录制起始时刻, 决定 epoch 时间轴。
    min_beats_per_epoch : int
        一个 epoch 内至少多少 RR 间期才算 HRV。不足的 epoch **整行剔除**
        （与 MESA 一致），而不是填 0。
    max_bad_epoch_frac : float
        可容忍的剔除比例上限，超过则抛错（该被试数据质量不足）。
    thr_resp_variability_warn : bool
        当呼吸特征在某窗口全部被填 0 时打印警告（提示信号质量问题）。

    Returns
    -------
    (features, diag)
        ``features`` 为合并后的特征表（索引 = epoch 时间轴）,
        ``diag`` 为 ``PipelineDiagnostics``。

    Raises
    ------
    ValueError
        输入信号退化（NaN 无法补齐 / 某支路产出列数异常 / 剔除 epoch 过多）。
    """
    signal, n_filled = fill_signal_nans(signal)
    if n_filled:
        print(f"  [WARN] 输入信号有 {n_filled} 个 NaN 样本, 已线性插值补齐", flush=True)
    sample_index = make_epoch_index(start_time, len(signal), fs)
    ep_index = epoch_axis(sample_index)

    # --- ACT: 体动 ---
    movement = extract_movement(signal, fs, sample_index)
    act_features = calc_actigraph_features(movement["movement"], causal=pc.causal)
    act_features.index = movement.index

    # --- HRV: 心搏 ---
    beat_result = detect_beats(signal, fs)
    # detect_beats 的 peak_times_s 相对信号起点, 换算成绝对时刻后喂给 hrv
    abs_peaks = sample_index[0] + pd.to_timedelta(beat_result.peak_times_s, unit="s")
    peaks_df = beat_result.peaks.copy()
    peaks_df.index = pd.DatetimeIndex(abs_peaks)
    # 两套口径都算:
    #   per-epoch (_hrv_*)      —— MESA 口径, 训练 13 槽位里的 7 个取自这里
    #   windowed  (150_hrv_* …) —— D04 口径, 槽位 3 (150_hrv_median_nni) 取自这里
    hrv_epoch = get_hrv_features_per_epoch(peaks_df, ep_index, min_rr=min_beats_per_epoch)
    hrv_win = get_hrv_features(peaks_df, epoch_index=ep_index)
    hrv_win.index = ep_index
    hrv_features = pd.concat([hrv_epoch, hrv_win], axis=1)

    # --- RRV: 呼吸 ---
    resp_series = pd.Series(signal, index=sample_index)
    resp_features = extract_rrv_features_helper(resp_series, nan_pad=0.0, sampling_rate=fs)
    resp_features.index = epoch_axis(resp_features.index)

    # --- 合并: 按 epoch 取交集, 保证三条支路的时间轴严格对齐 ---
    common = ep_index.intersection(act_features.index).intersection(
        hrv_features.index).intersection(resp_features.index)

    if len(common) == 0:
        raise ValueError(
            "三条支路的 epoch 无交集 —— 通常是录制时长过短。"
            f"信号 {len(signal) / fs / 60:.1f} min, 而 RRV 最长窗口 270 s、"
            "HRV 最长窗口 270 s 都需要足够的上下文。"
        )

    # 剔除拍数不足的 epoch —— HRV 逐 epoch 块整行为 NaN 即该标记。
    # 与 MESA 一致（rr_utils.py 里把 count < 10 的 epoch 整块删掉），
    # 而不是填 0 让 "median_nni = 0 ms" 这种非物理值混进特征表。
    bad = hrv_epoch.reindex(common).isna().all(axis=1).to_numpy()
    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / len(common)
        msg = (f"剔除 {n_bad}/{len(common)} ({frac:.1%}) 个 epoch"
               f"（RR 间期 < {min_beats_per_epoch}）")
        if frac > max_bad_epoch_frac:
            raise ValueError(
                f"{msg} —— 超过上限 {max_bad_epoch_frac:.0%}, 该被试信号质量不足")
        print(f"  [INFO] {msg}", flush=True)
        common = common[~bad]

    merged = pd.concat([
        act_features.loc[common].reset_index(drop=True),
        hrv_features.loc[common].reset_index(drop=True),
        resp_features.loc[common].reset_index(drop=True),
    ], axis=1)
    # 重复列名（如 HRV 列表里历史上重复的 _hrv_median_nni）去重
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged.index = common

    validate_blocks(merged)      # 分支退化时抛错, 不让整块特征静默消失

    if thr_resp_variability_warn:
        resp_cols = [c for c in merged.columns if "RRV" in c]
        if resp_cols:
            all_zero = (merged[resp_cols] == 0).all(axis=1).mean()
            if all_zero > 0.5:
                print(f"  [WARN] {all_zero:.0%} 的 epoch 的 RRV 特征全为 0 "
                      f"—— 呼吸信号质量可能有问题", flush=True)

    diag = PipelineDiagnostics(beat_result, movement, resp_features,
                               hrv_features, ep_index)
    return merged, diag
