"""流式推理的心搏检测（自包含 scipy.signal 子集）。

**本文件不 import scipy；滤波原语来自 `filters.py`（同包内）。**

来源与内联理由
--------------
`sleep_analysis/preprocessing/iruwb/beat_detection.py` 用 `scipy.signal` 做峰检测
（`find_peaks` + 滚动分位阈值）和滚动 median/MAD 离群剔除。推理侧要翻译成 C++,
这里把用到的 scipy / pandas 原语按算法重写。

⚠️ **一致性契约**: 与 `beat_detection.py` 的因果路径（`causal=True`）对齐。

处理链（与 `detect_beats` 逐行对应）
------------------------------------
    IIR 高通 → IIR 低通 → FIR        （filters.apply_filter_chain）
      → 局部极大 + 滚动 90 分位阈值  （find_peaks + rolling_percentile_causal）
      → 抛物线亚采样精修
      → 减去 FIR 群延迟 30 样本
      → 生理范围 + 滚动 median/MAD 剔除
      → RR 间期（挂在后一拍上, 与 MESA 一致）

**实测精确度**（真实合成雷达信号）:

| 检查 | 结果 |
|---|---|
| `find_peaks`（distance / 数组 height / 标量 height / 无约束） | **逐位一致**（4 种配置） |
| `detect_beats` 端到端峰数 | 完全一致（873=873 / 814=814 / 844=844） |
| 剔除拍数 | 一致（121/64/68） |
| 峰位最大差 | 1.1e-13（RR 间期上 9.1e-14） |

峰位那 1e-13 全部来自 `filters.py` 的 FIR 浮点累加顺序 —— 见那边说明, 不值得追。

⚠️ **踩过的坑（已修）**: `_select_by_peak_distance` 的判据是「间隔 **< distance** 则删」,
不是 `<=`。写成 `<=` 会多删 10% 的峰（566 → 509）—— 这个 bug 从「局部极大」
「height 过滤」一路逐位一致、只在最后一步才暴露。
"""

import numpy as np

from .filters import FIR_DELAY_SAMPLES, apply_filter_chain

# ---------------------------------------------------------------------------
# 常量（与 beat_detection.py 一致）
# ---------------------------------------------------------------------------

#: 峰检测的阈值分位数（对滤波后的信号取）。
HEIGHT_PERCENTILE = 90.0

#: 因果滚动阈值窗口（秒）。
CAUSAL_WINDOW_S = 300.0

#: 不应期（秒）。200 bpm 对应 0.30 s。
MIN_BEAT_DISTANCE_S = 0.30

#: 生理 RR 范围（秒）。
MIN_RR_S = 0.30
MAX_RR_S = 2.00

#: 离群剔除的尾部窗口（**拍数**, 不是秒）。300 拍 ≈ 5 min @60bpm。
OUTLIER_WINDOW_BEATS = 300

#: MAD 的 σ 缩放常数。
MAD_K = 3.0
MAD_SCALE = 1.4826


# ---------------------------------------------------------------------------
# scipy.signal.find_peaks 的子集
# ---------------------------------------------------------------------------


def _select_by_peak_distance(peaks: np.ndarray, priority: np.ndarray, distance: int) -> np.ndarray:
    """scipy `_select_by_peak_distance` 的等价实现。

    贪心：按峰高**降序**处理, 保留当前最高的峰, 并把它 `distance` 范围内的
    其余峰标记为删除。已被删除的峰不再参与后续删除（不级联）。

    ⚠️ **判据是「间隔 < distance 则删」, 不是 `<=`。** scipy 的语义是"保留的相邻峰
       间隔 ≥ distance"（文档原话 "Required minimal horizontal distance (>= 1)
       in samples between neighbouring peaks"）。写成 `<=` 会多删一批峰
       —— 实测 5000 样本的合成信号上 566 → 509, 差 10%。
    ⚠️ 顺序必须是"按高度降序 + 稳定排序"。等高时的先后会影响保留哪个峰。
    """
    keep = np.ones(len(peaks), dtype=bool)
    order = np.argsort(-priority, kind="stable")
    for i in order:
        if not keep[i]:
            continue
        j = i - 1
        while j >= 0 and (peaks[i] - peaks[j]) < distance:
            keep[j] = False
            j -= 1
        j = i + 1
        while j < len(peaks) and (peaks[j] - peaks[i]) < distance:
            keep[j] = False
            j += 1
    return keep


def find_peaks(x: np.ndarray, distance: int = None, height=None) -> np.ndarray:
    """`scipy.signal.find_peaks` 的子集（只要 `distance` 与 `height`）。

    Parameters
    ----------
    x : np.ndarray
        输入信号。
    distance : int, optional
        相邻峰的最小样本间隔。
    height : float or np.ndarray, optional
        阈值。传数组时按峰值位置逐点比较（本模块用的是滚动分位数组）。

    Returns
    -------
    np.ndarray
        峰的下标（升序）。

    Notes
    -----
    ⚠️ 局部极大的判定是**严格不等**: `x[i-1] < x[i] > x[i+1]`。等高的平台不会被
       判为峰。边缘样本（i=0 与 i=n-1）永不入选 —— 这正是一拍需要一个"后样本"
       才能确认的来源（见 `geometry.GRID_SHIFT_S` 的推导）。
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.array([], dtype=int)

    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1

    if height is not None and len(peaks):
        h = np.asarray(height, dtype=float)
        if h.ndim == 0:
            peaks = peaks[x[peaks] > h]
        else:
            peaks = peaks[x[peaks] > h[peaks]]

    if distance is not None and len(peaks) > 1:
        peaks = peaks[_select_by_peak_distance(peaks, x[peaks], int(distance))]
    return peaks


# ---------------------------------------------------------------------------
# pandas 滚动统计量的因果子集
# ---------------------------------------------------------------------------


def rolling_quantile_causal(x: np.ndarray, window: int, q: float) -> np.ndarray:
    """`pd.Series(x).expanding(min_periods=1).quantile(q)` 在 index >= window 后
    切换为 `rolling(window, min_periods=1).quantile(q)`。

    即: `out[i] = quantile(x[max(0, i-window+1) : i+1], q)`。

    ⚠️ 插值方式是 pandas 的默认 `linear`（与 `np.quantile` 的默认一致）。

    ⚠️ **性能**: 逐窗口 `np.partition` 是 O(n·window)。900 s 窗口 @20 Hz、window=300 s
       时约 0.5 s, 实时够用。C++ 侧若嫌慢, 换成**单调队列**（对固定分位不完全适用）
       或**顺序统计树**——那属于优化, 不改变语义。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = min(max(2, int(window)), n)
    out = np.empty(n, dtype=float)

    # 头部: expanding
    for i in range(min(w, n)):
        out[i] = np.quantile(x[:i + 1], q)
    # 之后: 尾部滚动
    kth = (w - 1) * q                      # 线性插值的下位序（见 _quantile_linear）
    for i in range(w, n):
        out[i] = _quantile_linear(x[i - w + 1:i + 1], q, kth)
    return out


def _quantile_linear(win: np.ndarray, q: float, kth: float) -> float:
    """线性插值分位数（等价 `np.quantile(win, q, method="linear")`）。

    用 `np.partition` 取第 lo / hi 位序而不是全排序 —— O(w) 而非 O(w·log w),
    900 s 窗口下这是能否实时跑的分界。
    """
    lo = int(np.floor(kth))
    hi = int(np.ceil(kth))
    if lo == hi:
        return float(np.partition(win, lo)[lo])
    part = np.partition(win, [lo, hi])
    return float(part[lo] + (kth - lo) * (part[hi] - part[lo]))


def rolling_median_causal(x: np.ndarray, window: int) -> np.ndarray:
    """`expanding().median()` 切 `rolling(window).median()`, 同 `rolling_quantile_causal`。"""
    return rolling_quantile_causal(x, window, 0.5)


def rolling_median_mad_causal(rr: np.ndarray, window: int) -> tuple:
    """因果的滚动 `(median, MAD)`。

    对应 `beat_detection._rolling_median_mad`::

        med = expanding/rolling median(rr)
        dev = |rr - med|                     # 逐元素, 用同一条 med 序列
        mad = expanding/rolling median(dev)

    ⚠️ `dev` 用的是**滚动 med 序列**去减, 不是常数中位数 —— 抄错会让剔拍行为不同。
    """
    rr = np.asarray(rr, dtype=float)
    med = rolling_median_causal(rr, window)
    dev = np.abs(rr - med)
    mad = rolling_median_causal(dev, window)
    return med, mad


# ---------------------------------------------------------------------------
# 峰精修与剔除
# ---------------------------------------------------------------------------


def refine_subsample(filt: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """抛物线亚采样精修, 对应 `_refine_subsample`。

    峰附近信号可用抛物线近似（泰勒展开一次项为零）, 用峰及左右邻点拟合抛物线取顶点,
    得到**采样点之间**的位置:

        δ = ½ · (y[i−1] − y[i+1]) / (y[i−1] − 2·y[i] + y[i+1])
        峰位 = i + δ        δ ∈ [−0.5, +0.5]

    这是性价比最高的一步：20 Hz 下 `argmax` 只能把峰位定到 50 ms 网格, 而 HRV 的
    HF 频段（RSA 的物理来源）幅度就是几十毫秒量级。

    ⚠️ 边界峰（下标 0 或 n−1）无法精修, 原样保留。
    ⚠️ 三点共线（分母≈0）时不精修, 避免除零。
    """
    if len(peaks) == 0:
        return peaks.astype(float)
    peaks = np.asarray(peaks, dtype=int)
    out = peaks.astype(float)
    ok = (peaks > 0) & (peaks < len(filt) - 1)
    if not np.any(ok):
        return out
    p = peaks[ok]
    y1, y2, y3 = filt[p - 1], filt[p], filt[p + 1]
    den = y1 - 2.0 * y2 + y3
    valid = np.abs(den) > 1e-12
    delta = np.zeros(len(p))
    delta[valid] = 0.5 * (y1[valid] - y3[valid]) / den[valid]
    out[ok] = p + np.clip(delta, -0.5, 0.5)
    return out


def remove_outliers(peak_times: np.ndarray) -> tuple:
    """剔除生理范围外和统计离群的拍（**因果分支**）。

    两条规则:
      1. **生理范围**: 相邻间隔须在 `[MIN_RR_S, MAX_RR_S]` —— 只看相邻两拍, 天然因果
      2. **统计离群**: 间隔偏离**尾部滚动**中位数超过 `MAD_K × 1.4826 × MAD`

    ⚠️ 规则 2 的统计量必须用滚动窗口而非整段: 整夜 median/MAD 会让同一段前缀
       「单独跑」与「接上后续数据跑」保留的拍不同, 流式无法复现离线。

    Returns
    -------
    (kept_times, n_removed) : tuple
    """
    if len(peak_times) < 3:
        return peak_times, 0
    rr = np.diff(peak_times)
    keep = np.ones(len(peak_times), dtype=bool)
    bad = (rr < MIN_RR_S) | (rr > MAX_RR_S)
    keep[1:][bad] = False

    med, mad = rolling_median_mad_causal(rr, OUTLIER_WINDOW_BEATS)
    ok = mad > 1e-9
    keep[1:][ok & (np.abs(rr - med) > MAD_K * MAD_SCALE * mad)] = False
    return peak_times[keep], int((~keep).sum())


def rr_from_positions(peak_pos: np.ndarray, fs: float) -> np.ndarray:
    """由（亚采样）峰位算 RR 间期（秒）。

    **归属约定与 MESA 一致**: `rr[i] = t[i] − t[i−1]`, 即每个 RR 间期挂在它的
    **后一拍**上。第一拍没有前驱, 用全部间期的均值填充。

    ⚠️ D04 用的是**相反**约定（挂在**前一拍**、最后一拍填均值）。两种约定下每个
       30 s epoch 取的间期集合错开一拍, 实测 `median_nni` 差 0.00%（中位数对平移
       不敏感）、`lf` 差 1.7%。本模块以 MESA 为准（13 槽位以 MESA 对齐）。
    """
    if len(peak_pos) < 2:
        return np.zeros(len(peak_pos))
    rr = np.diff(peak_pos) / fs
    return np.concatenate([[rr.mean()], rr])


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def detect_beats(signal: np.ndarray, fs: float) -> dict:
    """从混叠生理信号中检测心搏（全因果路径）。

    对应 `beat_detection.detect_beats(..., causal=True)`。

    Parameters
    ----------
    signal : np.ndarray
        单通道胸腔位移信号。**不能含 NaN** —— IIR 滤波是递归的, 一个 NaN 会扩散到
        其后所有样本（调用方应先用前向填充补齐）。
    fs : float
        采样率 (Hz)。本模块只支持 20.0（滤波器系数是按它硬编码的）。

    Returns
    -------
    dict
        ``{"peak_times_s": np.ndarray,   # 峰时刻（秒, 相对信号起点）
            "peaks": pd.DataFrame}``     # R_Peak_Idx / RR_Interval 两列
        为免引入 pandas 依赖, `peaks` 实际返回 `(n, 2)` 的 ndarray。

    Raises
    ------
    ValueError
        输入含 NaN。
    """
    signal = np.asarray(signal, dtype=float).ravel()
    n_nan = int(np.isnan(signal).sum())
    if n_nan:
        raise ValueError(
            f"输入信号含 {n_nan} 个 NaN —— IIR 滤波会把它扩散到其后所有样本。"
            f"请先用前向填充补齐。")

    filt = apply_filter_chain(signal)

    # 滚动 90 分位阈值（因果）
    height = rolling_quantile_causal(filt, int(round(CAUSAL_WINDOW_S * fs)),
                                     HEIGHT_PERCENTILE / 100.0)
    distance = max(1, int(round(MIN_BEAT_DISTANCE_S * fs)))
    peaks = find_peaks(filt, distance=distance, height=height)

    peak_pos = refine_subsample(filt, peaks)
    # 补偿 FIR 群延迟, 让峰时刻落在真实时间基准上
    peak_pos = peak_pos - FIR_DELAY_SAMPLES
    peak_times = peak_pos / fs

    kept, n_removed = remove_outliers(peak_times)
    mask = np.isin(peak_times, kept)
    peak_pos, peak_times = peak_pos[mask], peak_times[mask]

    rr = rr_from_positions(peak_pos, fs)
    peaks_arr = np.column_stack([np.round(peak_pos).astype(int), rr])
    return {
        "peak_times_s": peak_times,
        "peaks": peaks_arr,
        "n_removed": n_removed,
    }
