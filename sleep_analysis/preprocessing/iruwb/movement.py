"""IR-UWB 20 Hz 雷达信号的体动提取。

从 ``preprocessing/d04_main/movement_radar.py`` 移植，两处必要修改:

1. **按秒参数化**。原实现 ``_moving_average(window_size=19000)`` 是**采样点数**,
   在 D04 的 1953.125 Hz 下 = 9.73 s；直接用在我们 20 Hz 信号上会变成 950 s。
   这里统一用 ``window_s`` 秒, 内部换算 ``round(window_s * fs)``。

2. **因果分支**。原实现有两处整夜统计量 —— ``_scale_df`` 的 ``StandardScaler``
   和 ``_normalize`` 的整文件 min/max —— 都是未来泄漏。因果分支改用
   扩展窗口统计（只用已见数据）。

⚠️ 因果分支的已知偏差: 扩展 min/max 在**夜间早期**分母偏小, 会让早期 epoch
的归一化值偏大、更容易越过阈值被判为体动。这是"只用已见数据"的固有代价,
不是 bug；若这个偏差影响明显, 正确做法是用训练集标定的固定常数替换
（与模型 ``scaler.json`` 同一思路），而不是回退到整夜统计量。

另注意原实现的一个无效操作: ``_moving_average`` 里的 padding 是**死代码** ——
``padded[9500 : len(padded)-9500]`` 恰好切回原始 N 个样本, padding 从未生效;
因 ``min_periods=1``, 录制首尾各约 4.9 s 的窗宽会缩短。这里保留同样的行为
（不"顺手修好"），以免与 D04 的可比性被破坏。
"""

from typing import Optional

import numpy as np
import pandas as pd
import scipy.signal as ss

import sleep_analysis.processing_config as pc

# 原实现在 1953.125 Hz 下用 19000 样本 → 9.728 s
DEFAULT_MA_WINDOW_S = 19000 / 1953.125
DEFAULT_THRESHOLD = 0.2


def _moving_average(x: np.ndarray, window_samples: int, causal: bool = False) -> np.ndarray:
    """滑动平均。

    非因果分支用 ``center=True``（复刻 D04）。因果分支用尾部窗口 ——
    中心窗口本身就是向前看, 这是体动支路的时序泄漏点。

    ``min_periods=1`` 使首尾窗宽自动缩短（D04 的行为）。
    """
    w = max(1, int(window_samples))
    return (pd.Series(x).rolling(window=w, center=not causal, min_periods=1)
            .mean().to_numpy())


def _scale(x: np.ndarray, causal: bool) -> np.ndarray:
    """标准化。因果分支用扩展窗口均值/标准差（只含已见数据）。"""
    if causal:
        s = pd.Series(x)
        mean = s.expanding(min_periods=1).mean().to_numpy()
        std = s.expanding(min_periods=2).std().to_numpy()
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        return (x - mean) / std
    std = x.std()
    return (x - x.mean()) / (std if std > 1e-12 else 1.0)


def _first_order_derivative(x: np.ndarray) -> np.ndarray:
    """一阶导数的绝对值。``np.diff`` 天然因果（只用 t 和 t-1）。"""
    return np.abs(np.diff(x, prepend=x[0] if len(x) else 0.0))


def _normalize(x: np.ndarray, causal: bool) -> np.ndarray:
    """min-max 归一到 [0,1]。

    非因果用整文件 min/max（复刻 D04）；因果用扩展 min/max（只用已见数据）。
    见模块文档的偏差说明。
    """
    if causal:
        s = pd.Series(x)
        lo = s.expanding(min_periods=1).min().to_numpy()
        hi = s.expanding(min_periods=1).max().to_numpy()
    else:
        lo, hi = np.min(x), np.max(x)
    rng = hi - lo
    return (x - lo) / np.where(np.abs(rng) > 1e-12, rng, 1.0)


def extract_movement(signal: np.ndarray, fs: float, index: pd.DatetimeIndex,
                     window_s: float = DEFAULT_MA_WINDOW_S,
                     threshold: float = DEFAULT_THRESHOLD,
                     causal: Optional[bool] = None) -> pd.DataFrame:
    """从原始雷达信号提取逐 30 s epoch 的体动量。

    流程与 D04 一致: 滑动平均 → 标准化 → 一阶导数 → 归一化 → 阈值 → 30 s 分组均值。

    Parameters
    ----------
    signal : np.ndarray
        单通道信号（混叠生理信号本身即可 —— 体动表现为大幅瞬态）。
    fs : float
        采样率。
    index : pd.DatetimeIndex
        与 signal 等长的时间索引, 用于按 30 s 分组。
    window_s : float
        滑动平均窗长（秒）。
    threshold : float
        归一化后的阈值, 低于该值的样本置 0。
    causal : bool, optional
        默认取 ``processing_config.causal``。

    Returns
    -------
    pd.DataFrame
        单列 ``movement``, 索引为 30 s 对齐的时间戳。
    """
    if causal is None:
        causal = pc.causal
    signal = np.asarray(signal, dtype=float).ravel()

    ma = _moving_average(signal, round(window_s * fs), causal=causal)
    scaled = _scale(ma, causal)
    deriv = _first_order_derivative(scaled)
    norm = _normalize(deriv, causal)
    thresh = np.where(norm > threshold, norm, 0.0)

    df = pd.DataFrame({"movement": thresh}, index=index)
    df.index = df.index.floor("30s")
    return df.groupby(pd.Grouper(freq="30s", origin="epoch")).mean()
