"""IR-UWB 20 Hz 雷达信号的心搏检测。

处理流程
--------
::

    原始信号
      → IIR 带通（Butterworth 4 阶，默认 0.5–3.33 Hz，即 30–200 bpm）
      → 线性相位 FIR 带通（默认 61 tap，1.0–5.0 Hz）
      → 峰值检测（不应期 0.3 s + 分位数阈值）
      → 抛物线亚采样精修
      → 异常拍剔除（生理范围 + MAD）
      → 输出 R_Peak_Idx / RR_Interval

各参数的含义见 ``build_filter_chain`` 与 ``detect_beats`` 的 docstring。

⚠️ **本模块的所有参数取值都是在合成信号上调的，没有在真实 IR-UWB 数据上验证过。**
合成信号只包含建模时写进去的成分；真实信号的频谱、心搏波形形态、噪声结构都可能
与之不同。接入真机时应先观察实际信号，再重新评估这些参数，不要默认当前取值适用。

关于 d04 的实现
--------------
``d04_main`` 的心搏路径依赖 ``empkins_micro.emrad.radar``：它用 15–60 Hz 心音带，
配合一个在 1953.125 Hz 上训练的 TensorFlow LSTM 检测搏动。本雷达采样率 20 Hz
（Nyquist 10 Hz），该频段超出采样范围，那个 LSTM 也不适用，因此这里独立实现。
"""


from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as ss

import sleep_analysis.processing_config as pc

# ---------------------------------------------------------------------------
# 滤波链参数
# ---------------------------------------------------------------------------
# 两级串联:
#   IIR 带通 —— 决定哪些频率通过。由生理心率范围换算（30–200 bpm → 0.5–3.33 Hz）。
#   FIR 带通 —— 线性相位，在 IIR 之后再补一层频率选择性。
#
# ⚠️ 这些取值是在合成信号上调的, 未在真实数据上验证。改动前请先在真机数据上
#    观察实际频谱与波形（详见模块顶部说明）。

# 心搏检测带, 用**生理心率范围**表示。IIR 带通的通带由它换算。
DEFAULT_HR_RANGE_BPM = (30.0, 200.0)


def hr_bpm_to_hz(bpm_range):
    """心率范围 (bpm) → 频率范围 (Hz)。"""
    return (bpm_range[0] / 60.0, bpm_range[1] / 60.0)


DEFAULT_IIR_BAND_HZ = hr_bpm_to_hz(DEFAULT_HR_RANGE_BPM)   # (0.5, 3.333)

# Butterworth 阶数（每边）。阶数越高滚降越陡、相位非线性越强。
DEFAULT_IIR_ORDER = 4

# 线性相位 FIR 的通带与抽头数。
#
# ⚠️ **下沿必须 ≥ 1.0 Hz。** 因果 FIR 要做出某个下沿，抽头数得装得下至少一个
#    周期的低频：0.5 Hz 的周期是 2 s = 40 样本 @20 Hz，41 tap 只够一个周期，
#    做不出来。这是 FIR 设计的固有约束，不是从仿真得出的结论。
# 61 tap @20 Hz → 群延迟 30 样本 = 1.5 s。
DEFAULT_FIR_BAND_HZ = (1.0, 5.0)
DEFAULT_N_TAPS = 61

# 生理约束
MIN_RR_S = 0.30              # 200 bpm
MAX_RR_S = 2.00              # 30 bpm
MIN_BEAT_DISTANCE_S = 0.30   # 不应期

# 检测阈值分位数与因果滚动窗口长度
HEIGHT_PERCENTILE = 90.0
CAUSAL_WINDOW_S = 300.0

# 因果剔除（统计离群规则）用的尾部窗口长度, 单位是**拍数**不是秒。
# 300 拍 ≈ 5 min @60bpm, 与 CAUSAL_WINDOW_S 同一量级。
OUTLIER_WINDOW_BEATS = 300

# 一个 epoch 内至少多少拍才认为 HRV 可用（与 MESA 路径的 ">=10 RR" 一致）
MIN_BEATS_PER_EPOCH = 10


@dataclass
class BeatDetectionResult:
    """心搏检测结果。

    Attributes
    ----------
    peaks : pd.DataFrame
        列 ``R_Peak_Idx``(样本下标, 整数) / ``RR_Interval``(秒), 供
        ``iruwb.hrv.get_hrv_features`` 消费。注意**亚采样精度只保留在
        ``RR_Interval`` 里**；需要绝对时刻时用 ``peak_times_s``。
    peak_times_s : np.ndarray
        亚采样精度的峰时刻（秒, 相对信号起点）。因果分支已补偿群延迟。
    n_detected : int
        剔除异常前的原始检出数。
    n_removed_outlier : int
        被生理/统计规则剔除的拍数。
    latency_s : float
        因果分支的固有时延（样本），非因果为 0。
    """

    peaks: pd.DataFrame
    peak_times_s: np.ndarray
    n_detected: int
    n_removed_outlier: int
    latency_s: float


@dataclass
class FilterChain:
    """滤波链: IIR 带通定频率范围 + FIR 补足低频抑制（详见模块顶部说明）。

    Attributes
    ----------
    iir_highpass / iir_lowpass : (b, a) 或 None
        Butterworth 系数。None = 该级关闭。
    fir : np.ndarray 或 None
        线性相位带通 FIR 系数。**None = 关闭, 会导致峰位弥散、召回掉 4–8 倍**
        None 时跳过该级（仅供调试/对比用）。
    """

    iir_highpass: Optional[Tuple[np.ndarray, np.ndarray]]
    iir_lowpass: Optional[Tuple[np.ndarray, np.ndarray]]
    fir: Optional[np.ndarray]

    @property
    def fir_delay(self) -> int:
        """FIR 的群延迟（样本）= ``(n_taps - 1) // 2``，线性相位下与频率无关。"""
        return 0 if self.fir is None else (len(self.fir) - 1) // 2


def build_filter_chain(fs: float,
                       hr_bpm: Tuple[float, float] = DEFAULT_HR_RANGE_BPM,
                       iir_order: int = DEFAULT_IIR_ORDER,
                       fir_band_hz: Optional[Tuple[float, float]] = DEFAULT_FIR_BAND_HZ,
                       n_taps: Optional[int] = DEFAULT_N_TAPS) -> FilterChain:
    """构造滤波链: IIR 带通（由心率范围换算）+ 可选线性相位 FIR。

    Parameters
    ----------
    fs : float
        采样率。
    hr_bpm : (float, float)
        **心搏检测带, 用生理心率范围表示**。默认 (30, 200) bpm → (0.5, 3.333) Hz。
        这个范围之外的成分被 IIR 带通滤除。
    iir_order : int
        每边的 Butterworth 阶数。越高滚降越陡、相位非线性越强。
    fir_band_hz : (float, float) or None
        FIR 的通带。**它不决定检测带** —— 取一个覆盖 IIR 通带的带宽即可,
        作用是频率选择性, 见 ``DEFAULT_FIR_BAND_HZ`` 的说明。
    n_taps : int or None
        FIR 抽头数, 决定群延迟 ``(n_taps-1)//2`` 样本。None = 跳过该级。

    """
    lo, hi = hr_bpm_to_hz(hr_bpm)
    nyq = fs / 2.0
    if not 0 < lo < hi < nyq:
        raise ValueError(
            f"心率范围 {hr_bpm} bpm → {lo:.3f}–{hi:.3f} Hz 非法 "
            f"(需在 (0, {nyq}) 内, fs={fs})"
        )

    hp = ss.butter(iir_order, lo / nyq, btype="high")
    lp = ss.butter(iir_order, hi / nyq, btype="low")

    fir = None
    if n_taps is not None:
        f_lo, f_hi = fir_band_hz
        if not 0 < f_lo < f_hi < nyq:
            raise ValueError(f"FIR 通带 {fir_band_hz} 必须在 (0, {nyq}) 内 (fs={fs})")
        fir = ss.firwin(n_taps, [f_lo, f_hi], pass_zero=False, fs=fs)

    return FilterChain(hp, lp, fir)


def _apply_iir(x: np.ndarray, coeffs: Tuple[np.ndarray, np.ndarray],
               causal: bool) -> np.ndarray:
    """IIR 单级滤波。

    因果用 ``lfilter`` + ``lfilter_zi`` 稳态初始化（消除启动瞬态）；
    非因果用 ``filtfilt``（零相位）。IIR 只用于带外抑制, 两种方式的相位差异
    落在被丢弃的频段里, 对检测带影响可忽略。
    """
    b, a = coeffs
    if causal:
        zi = ss.lfilter_zi(b, a) * x[0]
        out, _ = ss.lfilter(b, a, x, zi=zi)
        return out
    return ss.filtfilt(b, a, x)


def apply_filter_chain(signal: np.ndarray, chain: FilterChain,
                       causal: bool) -> np.ndarray:
    """按 IIR 高通 → IIR 低通 → FIR 带通 的顺序作用滤波链。"""
    x = np.asarray(signal, dtype=float).ravel()
    if chain.iir_highpass is not None:
        x = _apply_iir(x, chain.iir_highpass, causal)
    if chain.iir_lowpass is not None:
        x = _apply_iir(x, chain.iir_lowpass, causal)
    if chain.fir is None:
        return x
    if causal:
        return ss.lfilter(chain.fir, [1.0], x)
    return ss.filtfilt(chain.fir, [1.0], x)


def _rolling_percentile(x: np.ndarray, fs: float, q: float,
                        window_s: float = CAUSAL_WINDOW_S) -> np.ndarray:
    """因果的滚动分位阈值。

    前 ``window_s`` 秒用扩展窗口（只用已见数据），之后用尾部滚动窗口。
    不使用任何未来样本。
    """
    s = pd.Series(x)
    w = min(max(2, int(round(window_s * fs))), len(x))
    head = s.expanding(min_periods=1).quantile(q / 100.0).to_numpy()
    if len(x) > w:
        head[w:] = s.rolling(w, min_periods=1).quantile(q / 100.0).to_numpy()[w:]
    return head


def _find_peaks(filt: np.ndarray, fs: float, causal: bool,
                height_percentile: float = HEIGHT_PERCENTILE) -> np.ndarray:
    """带不应期约束的峰检测（抛物线精修前的整数峰位）。"""
    distance = max(1, int(round(MIN_BEAT_DISTANCE_S * fs)))
    if causal:
        height = _rolling_percentile(filt, fs, height_percentile)
    else:
        height = np.percentile(filt, height_percentile)
    peaks, _ = ss.find_peaks(filt, distance=distance, height=height)
    return peaks


def _refine_subsample(filt: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """抛物线插值做亚采样峰位精修。

    对峰及其左右邻点拟合抛物线取顶点。这是**性价比最高的一步** —— 20 Hz 采样
    意味着 ±25 ms 的量化误差，而 HRV 的 HF 频段（RSA 的物理来源）幅度量级就是
    几十 ms。精修把中位误差从 13 ms 压到 7 ms，直接决定 ``hf`` / ``ratio_sd2_sd1``
    是否可用。

    边界峰（下标 0 或 n-1）无法精修，原样保留。
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
    valid = np.abs(den) > 1e-12          # 三点共线/非极大时不做精修，避免除零
    delta = np.zeros(len(p))
    delta[valid] = 0.5 * (y1[valid] - y3[valid]) / den[valid]
    out[ok] = p + np.clip(delta, -0.5, 0.5)
    return out


def _rolling_median_mad(rr: np.ndarray, window_beats: int) -> Tuple[np.ndarray, np.ndarray]:
    """因果的滚动 median 与 MAD（只用已见数据）。

    开头样本不足一个窗口时用扩展窗口（expanding），之后切尾部固定长度窗口
    —— 与 ``_rolling_percentile`` 同一套路。
    """
    s = pd.Series(rr)
    w = min(max(2, int(window_beats)), len(rr))
    med = s.expanding(min_periods=1).median().to_numpy()
    if len(rr) > w:
        med[w:] = s.rolling(w, min_periods=1).median().to_numpy()[w:]
    dev = pd.Series(np.abs(rr - med))
    mad = dev.expanding(min_periods=1).median().to_numpy()
    if len(rr) > w:
        mad[w:] = dev.rolling(w, min_periods=1).median().to_numpy()[w:]
    return med, mad


def _remove_outliers(peak_times: np.ndarray, mad_k: float = 3.0,
                     causal: Optional[bool] = None) -> Tuple[np.ndarray, int]:
    """剔除生理范围外和统计离群的拍。

    两条规则:
      1. **生理范围**: 相邻间隔须在 ``[MIN_RR_S, MAX_RR_S]`` 内 —— 只看相邻两拍,
         本身就是因果的。
      2. **统计离群**: 间隔偏离中位数超过 ``mad_k`` 倍 MAD（1.4826 缩放到 σ 口径）。

    ⚠️ **规则 2 的统计量必须随 ``causal`` 分支**, 否则流式推理无法复现离线特征:
    同一段前缀信号「单独跑」与「接上后续数据跑」, 保留的拍会不一样 ——
    因为整夜的 median/MAD 会随后续数据变化。

    实测不可复现的峰占比（前缀 vs 全量, 排除边界效应）::

        干净信号            0.06%
        3x 噪声 + 6 体动/h   0.57%
        3x 噪声 + 30 体动/h  1.80%

    即信号越差、体动越多, 差异越大。

    Parameters
    ----------
    causal : bool, optional
        None 时取 ``processing_config.causal``。
        False = 整夜 median/MAD（与 D04 口径一致）;
        True  = 尾部滚动 median/MAD（窗口 ``OUTLIER_WINDOW_BEATS`` 拍）。
    """
    if causal is None:
        causal = pc.causal
    if len(peak_times) < 3:
        return peak_times, 0
    rr = np.diff(peak_times)
    keep = np.ones(len(peak_times), dtype=bool)
    bad = (rr < MIN_RR_S) | (rr > MAX_RR_S)
    keep[1:][bad] = False

    if causal:
        med, mad = _rolling_median_mad(rr, OUTLIER_WINDOW_BEATS)
        ok = mad > 1e-9
        keep[1:][ok & (np.abs(rr - med) > mad_k * 1.4826 * mad)] = False
    else:
        good_rr = rr[~bad]
        if len(good_rr) >= 5:
            med = np.median(good_rr)
            mad = np.median(np.abs(good_rr - med))
            if mad > 1e-9:
                keep[1:][np.abs(rr - med) > mad_k * 1.4826 * mad] = False
    return peak_times[keep], int((~keep).sum())


def _rr_from_positions(peak_pos: np.ndarray, fs: float) -> np.ndarray:
    """由（亚采样）峰位计算 RR 间期（秒）。

    **归属约定与 MESA 一致**: ``rr[i] = t[i] − t[i−1]``，即每个 RR 间期挂在它的
    **后一拍**上。第一拍没有前驱，用全部间期的均值填充
    （MESA ``preprocessing/rr_utils.py``: ``seconds.diff()`` 后 ``fillna(mean)``）。

    ⚠️ **D04 用的是相反约定**（``empkins_micro.get_rpeaks`` 里
    ``np.ediff1d(..., to_end=0)``）：挂在**前一拍**上，最后一拍填均值。
    两种约定下，每个 30 s epoch 取的间期集合会错开一拍 —— 实测 HRV 逐 epoch
    特征差异: ``median_nni`` 0.00%（中位数对平移不敏感）, ``lf`` 1.7%（受影响最大）。
    这里选 MESA 约定，因为 13 个训练槽位以 MESA 为准。

    **亚采样精度在这里体现**：直接用整数下标做差会丢掉精修的全部收益。
    """
    if len(peak_pos) < 2:
        return np.zeros(len(peak_pos))
    rr = np.diff(peak_pos) / fs                     # rr[k] = t[k+1] − t[k]
    return np.concatenate([[rr.mean()], rr])        # 前置 → rr[i] = t[i] − t[i−1]


def detect_beats(signal: np.ndarray, fs: float,
                 hr_bpm: Tuple[float, float] = DEFAULT_HR_RANGE_BPM,
                 iir_order: int = DEFAULT_IIR_ORDER,
                 fir_band_hz: Optional[Tuple[float, float]] = DEFAULT_FIR_BAND_HZ,
                 n_taps: Optional[int] = DEFAULT_N_TAPS,
                 height_percentile: float = HEIGHT_PERCENTILE,
                 causal: Optional[bool] = None,
                 refine: bool = True,
                 remove_outliers: bool = True) -> BeatDetectionResult:
    """从混叠生理信号中检测心搏。

    Parameters
    ----------
    signal : np.ndarray
        单通道胸腔位移信号（呼吸 + 心搏混叠）。
    fs : float
        采样率 (Hz)。
    hr_bpm : (float, float)
        **心搏检测带, 用生理心率范围表示**。默认 (30, 200) bpm → 0.5–3.333 Hz。
        该范围外的成分被 IIR 带通滤除。
    iir_order : int
        每边 Butterworth 阶数。
    fir_band_hz : (float, float) or None
        FIR 通带（**不决定检测带**, 用于补足低频抑制）。
    n_taps : int or None
        FIR 抽头数。None = 跳过该级（仅供调试/对比用）。
    height_percentile : float
        检测阈值分位数。
    causal : bool, optional
        是否走全因果路径。默认取 ``processing_config.causal``。
        True 时**四步全部因果**: 滤波（``lfilter``）、检测阈值（滚动分位）、
        群延迟补偿、异常拍剔除（滚动 median/MAD）。
    refine : bool
        是否做抛物线亚采样精修。
    remove_outliers : bool
        是否剔除生理/统计异常拍。

    Returns
    -------
    BeatDetectionResult
        ``peaks`` 为 ``R_Peak_Idx`` / ``RR_Interval`` 两列, 可直接喂给
        ``iruwb.hrv.get_hrv_features``。
    """
    if causal is None:
        causal = pc.causal
    signal = np.asarray(signal, dtype=float).ravel()

    # NaN 守卫: IIR 滤波是递归的, 会把一个 NaN 扩散到其后**所有**样本;
    # 非因果分支的 filtfilt（前向+反向两遍）更是前后双向扩散。
    # 实测注入单个 NaN: 非因果分支检出 0 拍（整夜全废）, 因果分支只剩前半段。
    n_nan = int(np.isnan(signal).sum())
    if n_nan:
        raise ValueError(
            f"输入信号含 {n_nan} 个 NaN —— IIR 滤波会把它扩散到其后所有样本。"
            f"请先用 ``pipeline.fill_signal_nans`` 补齐（或自行插值）。"
        )

    chain = build_filter_chain(fs, hr_bpm, iir_order, fir_band_hz, n_taps)
    filt = apply_filter_chain(signal, chain, causal)
    # 只补偿 FIR 的群延迟 —— 线性相位下它与频率无关, 等于 (n_taps-1)//2, 可精确
    # 补偿。IIR 的群延迟随频率变化, 不作补偿, 因此因果分支的**绝对峰时刻**存在
    # 常数偏移。注意这对 HRV 没有影响 —— HRV 用 RR 间期, 常数偏移在相邻拍差值中抵消。
    delay = chain.fir_delay if causal else 0

    peaks = _find_peaks(filt, fs, causal, height_percentile)
    n_detected = len(peaks)

    peak_pos = _refine_subsample(filt, peaks) if refine else peaks.astype(float)
    # 因果分支补偿群延迟，让两条路径的峰时刻落在同一时间基准上
    peak_pos = peak_pos - delay
    peak_times = peak_pos / fs

    n_removed = 0
    if remove_outliers:
        kept, n_removed = _remove_outliers(peak_times, causal=causal)
        mask = np.isin(peak_times, kept)
        peak_pos, peak_times = peak_pos[mask], peak_times[mask]

    peaks_df = pd.DataFrame({
        "R_Peak_Idx": np.round(peak_pos).astype(int),
        "RR_Interval": _rr_from_positions(peak_pos, fs),
    })
    return BeatDetectionResult(peaks_df, peak_times, n_detected, n_removed, delay / fs)


def epoch_quality(peak_times: np.ndarray, n_epochs: int, epoch_s: float = 30.0,
                  min_beats_per_epoch: int = MIN_BEATS_PER_EPOCH) -> np.ndarray:
    """逐 epoch 的心搏质量标记 (1=可用, 0=不可用)。

    HRV 频域估计需要足够拍数；仓库既有的 MESA 路径用 "≥10 个 RR 间期" 作门槛
    （``preprocessing/rr_utils.py``），这里保持一致。

    调用方应据此屏蔽低质量 epoch，而不是让特征被算出来再用插值掩盖。
    """
    quality = np.zeros(n_epochs, dtype=int)
    if len(peak_times) == 0:
        return quality
    idx = (np.asarray(peak_times) / epoch_s).astype(int)
    idx = idx[(idx >= 0) & (idx < n_epochs)]
    counts = np.bincount(idx, minlength=n_epochs)
    quality[counts >= min_beats_per_epoch] = 1
    return quality
