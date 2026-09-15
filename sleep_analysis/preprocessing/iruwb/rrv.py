"""IR-UWB RRV（呼吸率变异性）特征提取。

**本文件是自包含副本, 不 import 任何 MESA/D04 模块。**

来源: ``feature_extraction/mesa_datasst/rrv.py``（MESA/SHHS/MrOS 共用的 RRV 特征）。
复制而非 import 的原因:
  1. 原 ``_extract_rrv_features_causal`` **只支持整数采样率** ——
     ``np.full((W-1)*30*fs, ...)`` 在 ``fs=20.0``（浮点）时抛
     ``TypeError: expected a sequence of integers``。此前只在 ``sampling_rate=32``
     （int）下跑过, 从未触发。这里做了显式取整。
  2. IR-UWB 管线要保持独立, 不依赖 MESA 的文件。

⚠️ 非因果路径（``causal=False``）的**列名与窗口口径必须与 MESA 一致** —— 这是训练侧
13 槽位对齐与 MESA 预训练权重迁移的前提。要核对一致性可以跑::

    # 用相同的整数采样率对比 (原版在浮点采样率下会崩, 所以只能用 int)
    python -c "
    import numpy as np, pandas as pd
    from sleep_analysis.feature_extraction.mesa_datasst.rrv import extract_rrv_features_helper as mesa
    from sleep_analysis.preprocessing.iruwb.rrv import extract_rrv_features_helper as ours
    x = pd.Series(np.random.RandomState(0).rand(32*600),
                  index=pd.date_range('2026-01-01', periods=32*600, freq=pd.Timedelta(seconds=1/32)))
    print(mesa(x, nan_pad=1.0, sampling_rate=32).shape == ours(x, nan_pad=1.0, sampling_rate=32).shape)
    "

窗口口径（与原作者一致, 注意原 docstring 里的"5/7/9 min"是笔误）:
  150 s (5 epoch) / 210 s (7 epoch) / 270 s (9 epoch), hop 30 s。
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
from biopsykit.utils.array_handling import sliding_window

import sleep_analysis.processing_config as pc

# 各窗口的 epoch 数与列名前缀（与 MESA 完全一致）
_WINDOWS = [(5, "150_"), (7, "210_"), (9, "270_")]


def _drop_dfa_columns(features: pd.DataFrame) -> pd.DataFrame:
    """剔除 neurokit 在呼吸次数足够时才产出的 DFA / MFDFA 列。

    这些列**只在数据够长时出现**（依赖 neurokit 内部的呼吸次数阈值），
    若不管就会让 RRV 的列数随录制时长/信号质量变化（实测 45s 窗口 20 列、
    210/270 窗口各 29 列的形态）。剔除后列数恒为 20×3 = 60。

    与 MESA/SHHS 的口径一致 —— 它们在 ``check_resp_features`` 里把
    ``*_RRV_DFA*`` / ``*_RRV_MFDFA*`` 全部丢弃，断言剩 61 列（60 特征 + epoch）。
    """
    drop = [c for c in features.columns
            if "RRV_DFA" in c or "RRV_MFDFA" in c]
    return features.drop(columns=drop) if drop else features


def extract_peaks(resp_df, sampling_rate: int):
    """呼吸峰检测（neurokit2 biosppy 方法 + fixpeaks）。"""
    _, peaks_dict = nk.rsp_peaks(resp_df, sampling_rate=sampling_rate, method="biosppy")
    return nk.rsp_fixpeaks(peaks_dict)


def calc_rrv_features(rsp_rate, peaks_dict, sampling_rate: int):
    """用 neurokit2 计算 RRV 特征, 返回 dict 列表。"""
    rrv = nk.rsp_rrv(rsp_rate, peaks_dict, sampling_rate=sampling_rate, show=False)
    return rrv.to_dict("records")


def _extract_rrv_features_causal(resp_arr, nan_pad=1.0, sampling_rate=32):
    """左对齐回顾窗口版本（严格因果, 实时分期用）。

    与原版（伪居中, 含未来）的差异:
      - 窗口: epoch j 的特征 = 数据 ``[j-W+1, j]``（含当前 epoch）, 而非原版的
        ``[j-p, j+p]``（向前看 p 个 epoch 的未来数据）
      - 首部: 前 W-1 个 epoch 的窗口用信号首值填充补齐（已见数据, 不引入未来信息）,
        因此**不丢行**, 行数仍为 n_epochs
      - 尾部: 无 pad（左对齐窗口天然覆盖到数据末尾）

    ⚠️ 与 MESA 原版的唯一实质差异: ``pad_samples`` / ``win_samples`` / ``ovl_samples``
    显式取整。原版直接传浮点给 ``np.full``, 在 ``sampling_rate`` 为 float 时崩溃。
    """
    fs = sampling_rate
    arr = np.asarray(resp_arr, dtype=float).ravel()

    parts = []
    for W_epochs, prefix in _WINDOWS:
        # 首部 pad (W-1) 个 epoch 的首值 (已见数据)
        pad_samples = int(round((W_epochs - 1) * 30 * fs))
        padded = np.concatenate([np.full(pad_samples, arr[0]), arr])
        win_samples = int(round(W_epochs * 30 * fs))
        ovl_samples = int(round((W_epochs * 30 - 30) * fs))
        windows = sliding_window(padded, win_samples, overlap_samples=ovl_samples)
        windows = np.nan_to_num(windows, nan=nan_pad)

        feature_list = []
        feature_keys = None
        for win in windows:
            try:
                peaks = extract_peaks(win, fs)
                feats = calc_rrv_features(win, peaks, fs)
                if feature_keys is None:
                    feature_keys = list(feats[0].keys())
                feature_list.append(feats[0])
            except (ValueError, IndexError):
                feature_list.append(dict.fromkeys(feature_keys or [], 0))
        parts.append(pd.DataFrame(feature_list).add_prefix(prefix))

    features = pd.concat(parts, axis=1)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0.0, inplace=True)

    time_axis = resp_arr.index.round("30s").drop_duplicates()[0 : features.shape[0]]
    features.index = time_axis
    features["epoch"] = np.arange(1, features.shape[0] + 1)

    return _drop_dfa_columns(features)


def extract_rrv_features_helper(resp_arr, nan_pad=1.0, sampling_rate=32):
    """计算 150/210/270 s 三个窗口的 RRV 特征。

    Parameters
    ----------
    resp_arr : pd.Series
        呼吸信号, 索引为时间轴（用于回填 epoch 时间戳）。
    nan_pad : float
        NaN 填充值。胸腔呼吸带用 1.0, EDR 用 0.0。
    sampling_rate : float
        采样率。**可以是浮点**（IR-UWB 是 20.0）。

    Returns
    -------
    pd.DataFrame
        60 个特征列（20 特征 × 3 窗口）+ ``epoch`` 列。
    """
    if pc.causal:
        return _extract_rrv_features_causal(resp_arr, nan_pad=nan_pad,
                                            sampling_rate=sampling_rate)

    # 非因果: 与原版一致 —— 首尾各补 (窗长-1)/2 行均值, 使窗口伪居中
    parts = []
    for W_epochs, prefix in _WINDOWS:
        win = int(round(W_epochs * 30 * sampling_rate))
        ovl = int(round((W_epochs * 30 - 30) * sampling_rate))
        pad_n = (W_epochs - 1) // 2
        arr = np.nan_to_num(
            np.pad(
                sliding_window(resp_arr, win, overlap_samples=ovl),
                ((pad_n, pad_n), (0, 0)),
                mode="mean",
            ),
            nan=nan_pad,
        )
        feature_list = []
        feature_keys = None
        for w in arr:
            try:
                peaks = extract_peaks(w, sampling_rate)
                feats = calc_rrv_features(w, peaks, sampling_rate)
                if feature_keys is None:
                    feature_keys = list(feats[0].keys())
                feature_list.append(feats[0])
            except (ValueError, IndexError):
                feature_list.append(dict.fromkeys(feature_keys or [], 0))
        parts.append(pd.DataFrame(feature_list).add_prefix(prefix))

    features = pd.concat(parts, axis=1)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0.0, inplace=True)

    time_axis = resp_arr.index.round("30s").drop_duplicates()[0 : features.shape[0]]
    features.index = time_axis
    features["epoch"] = np.arange(1, features.shape[0] + 1)

    return _drop_dfa_columns(features)
