"""IR-UWB HRV 特征提取（从心搏拍表算逐 epoch 的 HRV）。

**本文件是自包含副本, 不 import 任何 MESA/D04 模块。**

来源: ``feature_extraction/d04_main/hrv.py``（D04 雷达的 HRV 特征）。
复制而非 import 的三个原因, 每个都是硬性的:
  1. 原函数的 epoch 时间轴来自模块级全局变量 ``processed_folder`` / ``subj_id``
     （只在 ``__main__`` 里被赋值）, 脚本外调用直接 ``NameError``。这里改为显式
     参数 ``epoch_index``。
  2. 原实现的窗口是 ``[t, t+W]`` —— **全程向前看**最多 270 s。实时分期不可用。
     这里加了左对齐的因果窗口分支。
  3. IR-UWB 管线要保持独立, 不依赖 D04 的文件。

⚠️ **列名与语义必须与 D04/MESA 对齐** —— 这是训练侧 13 槽位对齐的前提, 也是从
MESA 预训练模型迁移权重的前提。非因果路径（``causal=False``）应与
``d04_main/hrv.py`` 产出一致的列名与窗口口径。

接口: 输入是 ``R_Peak_Idx`` / ``RR_Interval`` 两列的拍表（``beat_detection`` 的
``BeatDetectionResult.peaks`` 即此格式）, 索引为拍时刻。

**本文件提供两套口径**（列名与语义不同, 不要混用）:

  - ``get_hrv_features_per_epoch`` —— **MESA 口径**: 逐 30 s epoch 各算各的,
    无滑窗, 列名 ``_hrv_*``。
  - ``get_hrv_features`` —— **D04 口径**: 30/150/210/270 s 滑动窗口,
    列名 ``30_hrv_*`` / ``150_hrv_*`` / ...
"""

from typing import Optional

import numpy as np
import pandas as pd
from hrvanalysis import (
    get_csi_cvi_features,
    get_frequency_domain_features,
    get_geometrical_features,
    get_poincare_plot_features,
    get_time_domain_features,
)

import sleep_analysis.processing_config as pc

# 一个 epoch 内至少多少 RR 间期才算 HRV —— 与 MESA 一致
# (``preprocessing/rr_utils.py``: ``t1[t1["count"] < 10]`` 的 epoch 整块剔除)
MIN_RR_PER_EPOCH = 10


def overlapping_windows(df: pd.DataFrame, window_seconds: int,
                        causal: Optional[bool] = None) -> dict:
    """按 ``window_seconds`` 切重叠窗口, 返回 ``{epoch 时间戳: 窗口数据}``。

    **非因果分支（默认, 复刻 D04 行为）**: 窗口为 ``[t, t+W]``, key 取窗口**左端**
    的 30 s 对齐时刻 —— 即 epoch t 的特征**全程向前看** W 秒（最多 270 s）。

    **因果分支**（``pc.causal`` / ``causal=True``）: 窗口为 ``[t+30-W, t+30]``,
    即截止到该 epoch **末尾**的 W 秒回顾窗口, 只用已见数据。key 仍是 epoch t。
    取"截止到 epoch 末尾"而非"截止到 epoch 开头", 是为了让 30 s 窗口
    至少覆盖本 epoch 自身的全部数据。

    **两个分支的 epoch 覆盖范围不同**（因果性的固有代价）: 非因果需要 W 秒未来,
    因此录制**最后 W 秒**产不出特征; 因果需要 W 秒历史, 因此**最开始 W-30 秒**
    产不出特征。窗口长度一致, 所以特征列语义可比; 但特征表行数会不同。
    """
    if causal is None:
        causal = pc.causal

    overlap_seconds = window_seconds - 30
    step_size = pd.Timedelta(seconds=(window_seconds - overlap_seconds))

    if causal:
        windows_dict = {}
        first, last = df.index.min(), df.index.max()
        t = first.floor("30s")
        while t + step_size <= last:
            end = t + step_size                      # 本 epoch 的右端
            start = end - pd.Timedelta(seconds=window_seconds)
            # 数据起点之前没有 W 秒历史 → 跳过, 不产出不完整窗口
            # （不完整窗口的 RR 序列偏短, 特征分布与非因果分支不可比）
            if start >= first:
                window = df.loc[start:end]
                if not window.empty:
                    windows_dict[t] = window
            t += step_size
        return windows_dict

    window_size = pd.Timedelta(seconds=window_seconds)
    start_time = df.index.min()
    end_time = df.index.max()
    windows_dict = {}
    while start_time + window_size <= end_time:
        floored_start_time = start_time.floor("30s")
        if floored_start_time not in windows_dict:
            window = df.loc[start_time : start_time + window_size]
            if not window.empty:
                windows_dict[floored_start_time] = window
        start_time += step_size
    return windows_dict


def get_hrv_features_windows(r_peak_df: pd.DataFrame, window: int,
                             epoch_index: Optional[pd.DatetimeIndex] = None,
                             causal: Optional[bool] = None) -> pd.DataFrame:
    """单个窗口长度上的逐 epoch HRV 特征。

    比 D04 原版多一个显式的 ``epoch_index`` —— 原版从 ``movement_features``
    文件读时间轴（依赖脚本级全局变量）, 脚本外调用不可用。
    """
    if epoch_index is None:
        raise ValueError(
            "必须显式传 epoch_index。D04 原版从 movement_features 文件读时间轴, "
            "那依赖脚本级全局变量 processed_folder/subj_id, 从外部调用不可用。"
        )
    windows = overlapping_windows(r_peak_df, window, causal=causal)
    index = pd.DatetimeIndex(epoch_index)
    epochs = len(index)

    columns = [
        "mean_nni", "sdnn", "sdsd", "nni_50", "pnni_50", "nni_20", "pnni_20",
        "rmssd", "median_nni", "range_nni", "cvsd", "cvnni", "mean_hr", "max_hr",
        "min_hr", "std_hr", "lf", "hf", "lf_hf_ratio", "lfnu", "hfnu",
        "total_power", "vlf", "sd1", "sd2", "ratio_sd2_sd1", "csi", "cvi",
        "Modified_csi", "triangular_index", "tinn",
    ]
    feature_data = np.empty((epochs, len(columns)))
    feature_data.fill(np.nan)
    df_features = pd.DataFrame(feature_data, columns=columns, index=index)

    for key in windows.keys():
        if key not in df_features.index:
            continue
        # hrvanalysis 接受的是**毫秒**
        RR_values = windows[key]["RR_Interval"].values * 1000
        all_hr_features = {}
        try:
            all_hr_features.update(get_time_domain_features(RR_values))
            all_hr_features.update(get_frequency_domain_features(RR_values))
            all_hr_features.update(get_poincare_plot_features(RR_values))
            all_hr_features.update(get_csi_cvi_features(RR_values))
            all_hr_features.update(get_geometrical_features(RR_values))
            all_hr_features = pd.DataFrame(all_hr_features, index=[key])
            df_features.loc[key] = all_hr_features.loc[key]
        except Exception:
            # 拍数不足或退化窗口 —— 保持 NaN, 由下面的插值/填零兜底
            continue

    df_features = df_features.drop("tinn", axis=1)   # 恒为 None
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.interpolate(limit_direction="both")
    df_features = df_features.fillna(0.0)

    return df_features


def get_hrv_features_per_epoch(peaks_df: pd.DataFrame,
                               epoch_index: pd.DatetimeIndex,
                               min_rr: int = MIN_RR_PER_EPOCH) -> pd.DataFrame:
    """**MESA 口径**的 HRV: 每个 30 s epoch 内各自的 RR 间期上算, 不做滑动窗口。

    与同文件的 ``get_hrv_features``（**D04 口径**, 30/150/210/270 s 滑动窗口,
    列名带窗口前缀）是两套不同的口径。MESA 用的是这一套。

    对应 MESA 的 ``feature_extraction/mesa_datasst/hrv.py::calc_hrv_features``:
    按 R 点所属 epoch 分组, 每组单独算 HRV, 列名 ``_hrv_*``（**无窗口前缀**）。

    Parameters
    ----------
    peaks_df : pd.DataFrame
        索引 = 拍时刻（DatetimeIndex）, 列含 ``RR_Interval``（秒）。
    epoch_index : pd.DatetimeIndex
        输出的 30 s epoch 时间轴; 每拍按其所在 epoch 归组。
    min_rr : int
        一个 epoch 内至少多少 RR 间期才算特征。**默认 10, 与 MESA 一致**
        （``rr_utils.py``: ``t1[t1["count"] < 10]`` 的 epoch 整块剔除）。
        不足的 epoch 整行留 **NaN**, 由 ``pipeline.process_signal`` 剔除
        —— **不填 0**: 填 0 会让 ``_hrv_median_nni = 0 ms`` 这种物理上不可能的
        值被当成测量值喂进模型。

    Returns
    -------
    pd.DataFrame
        行 = epoch, 列 = ``_hrv_*``（30 个）。拍数不足的 epoch 整行为 NaN。
    """
    ep_of_beat = peaks_df.index.floor("30s")
    rr_ms = peaks_df["RR_Interval"].to_numpy() * 1000.0     # hrvanalysis 要毫秒

    rows = {}
    for e in epoch_index:
        vals = rr_ms[np.asarray(ep_of_beat == e)]
        if len(vals) < min_rr:
            continue
        f = {}
        try:
            f.update(get_time_domain_features(vals))
            f.update(get_frequency_domain_features(vals))
            f.update(get_poincare_plot_features(vals))
            f.update(get_csi_cvi_features(vals))
            f.update(get_geometrical_features(vals))
        except Exception:
            continue
        rows[e] = f

    df = pd.DataFrame(rows).T.reindex(epoch_index)
    if "tinn" in df.columns:                 # 恒为 None
        df = df.drop(columns=["tinn"])
    # inf → 0 (与 MESA 一致); **NaN 保留** —— 那是"该 epoch 没有可用 HRV"的标记,
    # 由调用方剔除整行, 不能填成 0 冒充测量值。
    df = df.replace([np.inf, -np.inf], 0.0)
    df.columns = ["_hrv_" + str(c) for c in df.columns]
    return df


def get_hrv_features(r_peak_df: pd.DataFrame,
                     epoch_index: Optional[pd.DatetimeIndex] = None,
                     causal: Optional[bool] = None) -> pd.DataFrame:
    """逐窗口计算 HRV 特征, 拼成 ``{30,150,210,270}_hrv_*`` 四组列。

    Parameters
    ----------
    r_peak_df : pd.DataFrame
        含 ``R_Peak_Idx`` / ``RR_Interval`` 两列, 索引为拍时刻。
    epoch_index : pd.DatetimeIndex
        输出的 epoch 时间轴。
    causal : bool, optional
        默认取 ``processing_config.causal``。

    Returns
    -------
    pd.DataFrame
        120 列（4 个窗口 × 30 个特征）。
    """
    r_peak_df = r_peak_df.copy()
    r_peak_df.index = pd.to_datetime(r_peak_df.index)
    r_peak_df = r_peak_df.dropna()[["R_Peak_Idx", "RR_Interval"]]

    parts = []
    for window, prefix in [(30, "30_hrv_"), (150, "150_hrv_"),
                           (210, "210_hrv_"), (270, "270_hrv_")]:
        f = get_hrv_features_windows(r_peak_df, window=window,
                                     epoch_index=epoch_index, causal=causal)
        parts.append(f.add_prefix(prefix))
    return pd.concat(parts, axis=1)
