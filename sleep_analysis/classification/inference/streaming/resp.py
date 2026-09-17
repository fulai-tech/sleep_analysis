"""流式推理的呼吸率变异性（RRV）特征（自包含 neurokit2 子集）。

**本文件是自包含实现, 不 import neurokit2 / scipy。**

来源与内联理由
--------------
调用方 `sleep_analysis/preprocessing/iruwb/rrv.py` 用 `neurokit2` 的
`rsp_peaks(method="biosppy")` / `rsp_fixpeaks` / `rsp_rrv`。推理侧只要其中
**4 个**特征（见下）, 且要翻译成 C++。

对齐版本：`neurokit2 0.2.10`。

**实测精确度**（真实合成雷达信号的 150 s / 270 s 滑窗）:

| 特征 | 与 neurokit2 的差异 |
|---|---|
| `MedianBB` / `CVBB` / `MCVBB` | **逐位一致** |
| `MeanBB` / `SDBB` / `MadBB`（中间量） | **逐位一致** |
| `LF` / `VLF` / `HF` | ≤ 1 ULP（相对 ~1e-15） |

**前六个逐位一致 ⇒ 呼吸峰检测（`rsp_findpeaks`）的移植是精确的** —— 这是整个
RRV 支路最容易写错的部分, 它对齐了就说明六个"照抄陷阱"都抄对了。
频域那 1 ULP 与 `hrv.py` 同源: `np.fft.rfft` 与 scipy 内部的 FFT 末位不同,
**不值得追**（C++ 侧用自己的 FFT 库, 同样是 1 ULP 级）。

覆盖的 4 个特征
--------------
`RRV_MedianBB` / `RRV_LF` / `RRV_MCVBB` / `RRV_CVBB`

其余 16 个输出里, `ApEn`/`SampEn` 依赖 scikit-learn、`DFA_*`/`MFDFA_*` 被调用方
`_drop_dfa_columns` 丢弃, 都不实现。

⚠️ 六个必须照抄的反直觉点（照抄错了数值会静默偏掉）
--------------------------------------------------
1. **`rsp_peaks` 完全不滤波**。文档说输入应该先 `rsp_clean`, 但函数自己不调用它;
   调用方直接传原始窗口。本模块同样不滤波。
2. **`method="biosppy"` 不用 `scipy.signal.find_peaks`**。走的是自实现的
   **零交叉 + 区间极值**, 且**零交叉以 `0.0` 为基准**（不是均值）。
   原始信号若带 DC 偏置, 检测结果会整体偏掉。
3. **`rsp_fixpeaks` 是空操作** —— 源码注释 "nothing for now"。直接不要移植。
4. **`bbi` 取自 `troughs` 而非 `peaks`**（吸气起点之间的间期）, 单位毫秒。
5. **`RRV_LF` 的输入是 `60 × fs / x[i]`** —— 逐样本的倒数变换, 不是呼吸率序列。
   `x[i] ≈ 0` 时产生 ±inf, 调用方随后 `replace([inf,-inf], nan)` 再填 0。
6. **PSD 归一化 `power /= max(power)` 发生在频带裁剪之前**, 除数是**全频段**最大值。
   所以 `RRV_LF` 是**无量纲**的: 同一个窗口内 VLF/LF/HF 可比, 跨窗口不可比。

另有一处**浮点陷阱**（见 `_resp_psd`）: `nperseg` 的推导式子不能化简为 `N//2`,
否则 LF 差 0.7%（实测）。浮点误差使其通常比 `N/2` 小 1。
"""

import numpy as np

#: 呼吸峰之间的最小间期（秒）。fast-breathing 时这个规则会大量剔峰。
MIN_PEAK_DISTANCE_S = 1.7

#: RRV 频带, 左闭右开。
RRV_BAND_VLF = (0.0, 0.04)
RRV_BAND_LF = (0.04, 0.15)
RRV_BAND_HF = (0.15, 0.40)

#: PSD 裁剪上限（Hz）, 由 `signal_power(max_frequency=0.5)` 传入。
RRV_MAX_FREQ = 0.5

#: MAD 的正态一致性常数（neurokit2 `stats.mad` 的默认值）。
MAD_CONSTANT = 1.4826


# ---------------------------------------------------------------------------
# 呼吸峰检测（neurokit2 biosppy 路径）
# ---------------------------------------------------------------------------


def _findpeaks_extrema(rsp: np.ndarray) -> np.ndarray:
    """零交叉分段 + 区间极值。对应 `_rsp_findpeaks_extrema`。

    ⚠️ 零交叉是相对 **`0.0`** 判断的, 不是相对均值。

    - `risex` / `fallx` 存的是**过零点前一侧样本的下标**: `risex` 处 `x[i] < 0 <= x[i+1]`
    - 区间 `rsp[beg:end]` 是**左闭右开**
    - 从 `startx` 推导首段搜 max 还是 min, 之后严格交替
    """
    greater = rsp > 0
    smaller = rsp < 0
    risex = np.where(np.bitwise_and(smaller[:-1], greater[1:]))[0]
    fallx = np.where(np.bitwise_and(greater[:-1], smaller[1:]))[0]

    if len(risex) == 0 or len(fallx) == 0:
        raise IndexError("信号没有完整的上升/下降过零点 —— 无法定位极值区间")

    if risex[0] < fallx[0]:
        startx = "rise"
    elif fallx[0] < risex[0]:
        startx = "fall"
    else:
        raise IndexError("首个上升沿与下降沿同下标 —— 信号退化")

    allx = np.concatenate((risex, fallx))
    allx.sort(kind="mergesort")

    extrema = []
    for i in range(len(allx) - 1):
        if startx == "rise":
            argextreme = np.argmax if (i + 1) % 2 != 0 else np.argmin
        else:
            argextreme = np.argmin if (i + 1) % 2 != 0 else np.argmax
        beg, end = allx[i], allx[i + 1]
        extrema.append(beg + argextreme(rsp[beg:end]))
    return np.asarray(extrema)


def _findpeaks_outliers(rsp: np.ndarray, extrema: np.ndarray) -> tuple:
    """幅度过滤 + 强制 peak/trough 交替。对应 `_rsp_findpeaks_outliers(amplitude_min=0)`。

    ⚠️ `amplitude_min=0` 使条件退化成 `vertical_diff > 0` —— **只有幅度差恰好为 0
       的才被剔除**, 实际几乎不删任何东西。biosppy 路径的调用方传的就是 0
       （khodadad 路径传 0.3, 不是本路径）。名字叫 `amplitude_min` 但传 0 时它
       退化成一个"去重"操作, 容易误解。
    """
    vertical_diff = np.abs(np.diff(rsp[extrema]))
    median_diff = np.median(vertical_diff)
    min_diff = np.where(vertical_diff > (median_diff * 0.0))[0]
    extrema = extrema[min_diff]

    amplitudes = rsp[extrema]
    extdiffs = np.sign(np.diff(amplitudes))
    extdiffs = np.add(extdiffs[0:-1], extdiffs[1:])
    removeext = np.where(extdiffs != 0)[0] + 1
    extrema = np.delete(extrema, removeext)
    amplitudes = np.delete(amplitudes, removeext)
    return extrema, amplitudes


def _findpeaks_sanitize(extrema: np.ndarray, amplitudes: np.ndarray) -> tuple:
    """强制序列"以 trough 开头、以 peak 结尾"。对应 `_rsp_findpeaks_sanitize`。

    处理后 `len(peaks) == len(troughs)`, 且 `troughs[k]` 是 `peaks[k]` 之前的那个波谷
    —— 两者**一一配对**, 后续任何删除都必须同步删。
    """
    if len(amplitudes) < 2:
        raise IndexError(f"极值只有 {len(amplitudes)} 个, 不足以配对")
    if amplitudes[0] > amplitudes[1]:
        extrema = np.delete(extrema, 0)
    if amplitudes[-1] < amplitudes[-2]:
        extrema = np.delete(extrema, -1)
    peaks = extrema[1::2]
    troughs = extrema[0:-1:2]
    return peaks, troughs


def rsp_findpeaks(rsp: np.ndarray, sampling_rate: float) -> tuple:
    """呼吸峰检测, 对齐 `neurokit2.rsp_peaks(method="biosppy")`。

    Parameters
    ----------
    rsp : np.ndarray
        **原始**呼吸波形窗口（不去均值、不滤波 —— 见模块文档第 1/2 点）。
    sampling_rate : float
        采样率, 只用于 1.7 s 最小间期规则的秒换算。

    Returns
    -------
    (peaks, troughs) : tuple of np.ndarray
        样本下标。

    Raises
    ------
    IndexError
        信号没有完整的过零点对, 或极值不足以配对（对应 neurokit2 会抛
        `IndexError` 的情形, 调用方靠 `except (ValueError, IndexError)` 兜底）。
    """
    extrema = _findpeaks_extrema(rsp)
    extrema, amplitudes = _findpeaks_outliers(rsp, extrema)
    peaks, troughs = _findpeaks_sanitize(extrema, amplitudes)

    # ⚠️ 1.7 s 最小间期: 删的是**每对中较早的那个峰**, peaks/troughs 同步删。
    #    呼吸快于 ~35 bpm 时这条规则会把峰几乎删光（0.7 Hz 时 300 s 只剩 1 个峰）。
    outlier_idcs = np.where((np.diff(peaks) / sampling_rate) < MIN_PEAK_DISTANCE_S)[0]
    peaks = np.delete(peaks, outlier_idcs)
    troughs = np.delete(troughs, outlier_idcs)
    return peaks, troughs


# ---------------------------------------------------------------------------
# RRV 特征
# ---------------------------------------------------------------------------


def _mad(x: np.ndarray) -> float:
    """`neurokit2.stats.mad`（`constant=1.4826`）。"""
    arr = np.ma.array(x).compressed()
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(x - med)) * MAD_CONSTANT)


def _resp_psd(signal: np.ndarray, sampling_rate: float) -> tuple:
    """`neurokit2.signal_psd` 的 Welch 路径（去均值 → 归一化 → 裁频）。

    ⚠️ **`nperseg` 的推导式子必须逐字照抄**, 不能化简成 `N//2`:

        min_frequency = (2 * fs) / (N / 2)
        nperseg = int((2 / min_frequency) * fs)      # 浮点误差 → 常比 N/2 小 1
        if nperseg > N / 2: nperseg = int(N / 2)

    实测 `N=4200` 时上式得 2099（而 `N//2 = 2100`）。化简后 `RRV_LF` 差 0.7%;
    照抄则与 neurokit2 逐位一致。

    ⚠️ **`power /= max(power)` 在频带裁剪之前**, 除数是全频段最大值 —— 见模块文档第 6 点。
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    n = len(signal)
    if n < 2:
        raise ValueError(f"PSD 输入长度 {n} 过短")

    min_frequency = (2.0 * sampling_rate) / (n / 2.0)
    if min_frequency == 0:
        min_frequency = 0.001
    nperseg = int((2.0 / min_frequency) * sampling_rate)
    if nperseg > n / 2.0:
        nperseg = int(n / 2.0)
    if nperseg < 1:
        raise ValueError(f"nperseg={nperseg} 非法")

    nfft = int(nperseg * 2)
    freq, power = _welch_density(signal, sampling_rate, nperseg, nfft)

    # ⚠️ 归一化在裁剪之前
    mx = np.max(power)
    if mx != 0:
        power = power / mx

    keep = (freq >= min_frequency) & (freq <= RRV_MAX_FREQ)
    return freq[keep], power[keep]


def _welch_density(x: np.ndarray, fs: float, nperseg: int, nfft: int) -> tuple:
    """`scipy.signal.welch(scaling="density", detrend=False, average="mean", window="hann")`。

    ⚠️ 与 `hrv.py::_welch_psd` 的两处关键差别（不要互相抄错）:
      - **`detrend=False`** —— 每段**不**再减均值（那边是 `detrend="constant"`）
      - `noverlap = nperseg // 2`（与那边一致）, 但这边 `nfft = nperseg * 2` 由调用方算好
    """
    n = len(x)
    nstep = nperseg - nperseg // 2

    # 周期 Hann —— 同 `hrv.py`, 照抄 scipy 的计算路径以逐位对齐
    _m = nperseg + 1
    win = (0.5 + 0.5 * np.cos(np.linspace(-np.pi, np.pi, _m)))[:-1]

    scale = 1.0 / (fs * float(np.sum(win * win)))

    nseg = int(np.ceil((n - nperseg + 1) / nstep)) if n >= nperseg else 1
    if nseg < 1:
        nseg = 1

    psd = None
    for k in range(nseg):
        beg = k * nstep
        seg = x[beg:beg + nperseg]
        if len(seg) < nperseg:
            seg = np.concatenate([seg, np.zeros(nperseg - len(seg))])
        # detrend=False → 不再减均值
        spec = np.fft.rfft(seg * win, nfft)
        p = (spec.real**2 + spec.imag**2) * scale
        if nfft % 2 == 0:
            p[1:-1] *= 2.0
        else:
            p[1:] *= 2.0
        psd = p if psd is None else psd + p
    psd = psd / nseg

    return np.fft.rfftfreq(nfft, d=1.0 / fs), psd


def _band_power(freq: np.ndarray, power: np.ndarray, band: tuple) -> float:
    """左闭右开频带的梯形积分; 结果为 0 时返回 NaN。"""
    m = (freq >= band[0]) & (freq < band[1])
    p = float(np.trapz(y=power[m], x=freq[m]))
    return np.nan if p == 0.0 else p


def rrv_features(rsp: np.ndarray, sampling_rate: float) -> dict:
    """由一段呼吸波形算 RRV 特征。

    对应离线路径 `rrv.py` 里的::

        peaks = extract_peaks(win, fs)          # rsp_peaks + rsp_fixpeaks
        feats = calc_rrv_features(win, peaks, fs)[0]   # rsp_rrv

    Parameters
    ----------
    rsp : np.ndarray
        原始呼吸波形窗口（**不去均值、不滤波**）。
    sampling_rate : float
        采样率 (Hz)。

    Returns
    -------
    dict
        键含 `MedianBB` / `CVBB` / `MCVBB` / `LF`（以及其余不需要的中间特征,
        便于对拍时定位问题）。

    Raises
    ------
    IndexError / ValueError
        峰检测退化（无过零点、极值不足）或 `bbi` 为空 —— 与 neurokit2 一致,
        调用方靠 `except (ValueError, IndexError)` 把整个窗口判为无效。

    Notes
    -----
    ⚠️ `bbi` 取自 **troughs**（见模块文档第 4 点）; `LF` 的输入是**波形的逐点倒数**
    （第 5 点）—— 两者数据来源完全独立, 可分开验证。
    """
    rsp = np.asarray(rsp, dtype=float)
    peaks, troughs = rsp_findpeaks(rsp, sampling_rate)

    # 4. bbi 取自 troughs, 单位毫秒
    bbi = np.diff(troughs) / sampling_rate * 1000.0
    if len(bbi) == 0:
        raise ValueError("bbi 为空 —— 呼吸峰不足以构成间期")

    diff_bbi = np.diff(bbi)
    out = {}
    out["RMSSD"] = float(np.sqrt(np.mean(diff_bbi**2))) if len(diff_bbi) else np.nan
    out["MeanBB"] = float(np.nanmean(bbi))
    out["SDBB"] = float(np.nanstd(bbi, ddof=1))
    out["SDSD"] = float(np.nanstd(diff_bbi, ddof=1)) if len(diff_bbi) else np.nan
    out["CVBB"] = out["SDBB"] / out["MeanBB"]
    out["CVSD"] = out["RMSSD"] / out["MeanBB"]
    out["MedianBB"] = float(np.nanmedian(bbi))
    out["MadBB"] = _mad(bbi)
    out["MCVBB"] = out["MadBB"] / out["MedianBB"]

    # 5. 频域: 输入是波形的逐点倒数（neurokit2 的口径 quirk）
    with np.errstate(divide="ignore", invalid="ignore"):
        rsp_period = 60.0 * sampling_rate / rsp
    freq, power = _resp_psd(rsp_period, sampling_rate)
    out["VLF"] = _band_power(freq, power, RRV_BAND_VLF)
    out["LF"] = _band_power(freq, power, RRV_BAND_LF)
    out["HF"] = _band_power(freq, power, RRV_BAND_HF)
    total = out["VLF"] + out["LF"] + out["HF"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["LFHF"] = out["LF"] / out["HF"]
        out["LFn"] = out["LF"] / total
        out["HFn"] = out["HF"] / total

    # 附: Poincaré（不参与 4 个槽位, 但同源, 便于定位）
    if len(diff_bbi):
        sdsd = np.std(diff_bbi, ddof=1)
        sdnn = np.std(bbi, ddof=1)
        out["SD1"] = float(np.sqrt(sdsd**2 * 0.5))
        out["SD2"] = float(np.sqrt(2.0 * sdnn**2 - 0.5 * sdsd**2))
        out["SD2SD1"] = out["SD2"] / out["SD1"]
    return out
