"""流式推理的 HRV 特征（自包含 hrvanalysis 子集）。

**本文件是自包含实现, 不 import hrvanalysis / scipy。**

来源与内联理由
--------------
调用方 `sleep_analysis/preprocessing/iruwb/hrv.py` 用 `hrvanalysis` 的三个函数算
30 个 HRV 特征。推理侧只要其中 **7 个**（见下）, 且要翻译成 C++。把 hrvanalysis
整个搬过去不现实, 所以这里按公式重写。

⚠️ **一致性契约**: 本文件对齐 `hrvanalysis 1.0.5`。
   版本以 `hrv_analysis-1.0.5.dist-info` 为准（`hrvanalysis.__version__` 是上游
   忘记更新的陈旧值 "1.0.3"）。

**实测精确度**（随机 RR 序列，10 / 30 / 150 / 270 拍四档）:

| 特征 | 与 hrvanalysis 的差异 |
|---|---|
| `median_nni` | **逐位一致** |
| `ratio_sd2_sd1` | **逐位一致** |
| `vlf` / `lf` / `hf` / `lf_hf_ratio` / `total_power` | ≤ 1 ULP（相对 ~2e-16） |

频域那 1 ULP **来自 FFT 实现本身**：`np.fft.rfft` 与 `scipy.fft.rfft`（scipy
内部用的）在末位不同，两者都是 pocketfft 但调用路径不同。已逐一排除窗函数
（改为照抄 scipy 的计算路径后**逐位一致**）和插值（`np.interp` 与 `interp1d`
逐位一致）—— 剩下的只有 FFT。

⇒ **不值得追**：C++ 侧会用目标平台的 FFT 库，同样是 1 ULP 级差异。
   对拍时频域特征用 `rtol=1e-12` 级容差, 时域/Poincaré 用严格相等。

覆盖的 7 个特征
--------------
===================  ==========================================
`median_nni`         NN 间期的中位数（**原始值**, 未插值未去 NaN）
`ratio_sd2_sd1`      Poincaré `sd2 / sd1`（**是 sd2/sd1**, 不是倒数）
`vlf` `lf` `hf`      频带功率积分（ms²）
`lf_hf_ratio`        `lf / hf`
`total_power`        `vlf + lf + hf`（**含 VLF**, 不含 >0.4 Hz）
===================  ==========================================

单位：输入 `nn_ms` 是**毫秒**（hrvanalysis 的约定）, 频域输出因此是 ms²。

⚠️ 三个必须照抄的细节（照抄错了数值会静默偏掉）:
  1. **插值用 4 Hz 均匀栅格**, 且去直流用的是**插值后序列的均值**（不是原始 RR 均值）
  2. **`total_power` 含 VLF** —— 不是 LF+HF
  3. **频带左闭右开**: VLF `[0.003, 0.04)`、LF `[0.04, 0.15)`、HF `[0.15, 0.40)`

⚠️ 退化行为（调用方靠 `try/except` 统一吞掉并入 NaN, 内联实现要对齐这个语义）:
  - `n == 1` 时 `hrvanalysis` 会在 `pnni_50` 处 `ZeroDivisionError`,
    **整个 epoch 的所有特征一起失效**（尽管 `median_nni` 本身算得出来）。
    本模块用 `_check_length` 复刻这个"一损俱损"的语义。
  - `sd1 → 0`（RR 序列近乎等差）时 `ratio_sd2_sd1` 会得到 ~1e15 的**有限巨值**,
    不是 `inf` —— 调用方的 `replace([inf,-inf], nan)` **兜不住**。
"""

import numpy as np

#: 频域分析的重采样频率（Hz）。hrvanalysis 的默认值, 不是采样率。
HRV_INTERP_FS = 4.0

#: Welch 的零填充长度。hrvanalysis 显式传入, 与 nperseg 无关。
HRV_NFFT = 4096

#: Welch 分段长度上限。更短的输入会被截断为输入长度（scipy 的行为）。
HRV_MAX_NPERSEG = 256

#: 频带边界（Hz），左闭右开。
BAND_VLF = (0.003, 0.04)
BAND_LF = (0.04, 0.15)
BAND_HF = (0.15, 0.40)


# ---------------------------------------------------------------------------
# 时域 / Poincaré
# ---------------------------------------------------------------------------


def _check_length(nn: np.ndarray) -> None:
    """复刻 hrvanalysis 的退化语义。

    `get_time_domain_features` 在 `n == 1` 时算 `pnni_50 = 100 * nni_50 / (n-1)`
    会 `ZeroDivisionError`。调用方是**一个 try 包住五个函数**, 所以这个异常会让
    该 epoch 的**全部** HRV 特征失效 —— 包括本来算得出来的 `median_nni`。

    ⚠️ 不要"顺手修好"成 `n == 1` 时只返回 `median_nni` —— 那会让本模块与
       离线路径在同一批 epoch 上一个出 NaN、一个有值, 对拍时难以定位。
    """
    if len(nn) < 2:
        raise ValueError(f"RR 序列长度 {len(nn)} < 2 —— hrvanalysis 会在此抛异常")


def median_nni(nn_ms: np.ndarray) -> float:
    """NN 间期的中位数（毫秒）。"""
    return float(np.median(nn_ms))


def poincare_sd1_sd2(nn_ms: np.ndarray) -> tuple:
    """Poincaré 的 `(sd1, sd2)`。

    `sd1 = sqrt(std(diff, ddof=1)² × 0.5)`
    `sd2 = sqrt(2·std(nn, ddof=1)² − 0.5·std(diff, ddof=1)²)`

    ⚠️ 一律 `ddof=1`（样本标准差）。`sd2` 的 sqrt 参数**可以为负** → NaN。
    """
    diff = np.diff(nn_ms)
    sdsd = np.std(diff, ddof=1)
    sdnn = np.std(nn_ms, ddof=1)
    sd1 = np.sqrt(sdsd**2 * 0.5)
    sd2 = np.sqrt(2.0 * sdnn**2 - 0.5 * sdsd**2)
    return float(sd1), float(sd2)


def ratio_sd2_sd1(nn_ms: np.ndarray) -> float:
    """Poincaré `sd2 / sd1`。

    ⚠️ `sd1 → 0` 时返回 ~1e15 的**有限巨值**（不是 inf）—— 与 hrvanalysis 一致,
       这是已知隐患, 调用方需自行判断是否合理。
    """
    sd1, sd2 = poincare_sd1_sd2(nn_ms)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(sd2 / sd1)


# ---------------------------------------------------------------------------
# 频域（Welch）
# ---------------------------------------------------------------------------


def _welch_psd(x: np.ndarray, fs: float, nperseg: int, nfft: int) -> tuple:
    """单边功率谱密度, 对齐 `scipy.signal.welch(x, fs, window="hann", nfft=nfft)`。

    生效的默认参数（scipy 1.13）: `noverlap = nperseg // 2`、`detrend="constant"`
    （每段各自再减均值）、`scaling="density"`、`average="mean"`、`return_onesided=True`。

    实现要点:
      - 窗是**周期 Hann**（DFT-even）: `0.5 − 0.5·cos(2πn/N)`。scipy 的
        `get_window("hann", N, fftbins=True)` 就是它, 不要用 `np.hanning`（那是对称型）
      - 分段数 `nseg = ceil((N − nperseg + 1) / nstep)`, 段起点 `k*nstep`
      - 标定 `scale = 1 / (fs · Σw²)`; 单边谱中除 DC 与 Nyquist 外乘 2
    """
    n = len(x)
    nstep = nperseg - nperseg // 2                      # noverlap = nperseg // 2

    # 周期 Hann。
    # ⚠️ 下面这四行刻意照抄 `scipy.signal.get_window("hann", N, fftbins=True)` 的
    #    **计算路径**（`general_cosine` 的 `_extend` + `linspace` + 截断），而不是写
    #    数学上等价的 `0.5 - 0.5*cos(2*pi*n/N)`。两者数值相差 1.67e-16 —— 就是这个
    #    1 ULP 让整个频域特征无法逐位对齐 scipy。照抄后逐位一致。
    _m = nperseg + 1                                    # sym=False → M+1
    _fac = np.linspace(-np.pi, np.pi, _m)
    win = 0.5 + 0.5 * np.cos(_fac)
    win = win[:-1]                                      # 截断回 nperseg

    scale = 1.0 / (fs * float(np.sum(win * win)))

    nseg = int(np.ceil((n - nperseg + 1) / nstep)) if n >= nperseg else 0
    if nseg <= 0:
        nseg = 1

    psd = None
    for k in range(nseg):
        beg = k * nstep
        seg = x[beg:beg + nperseg]
        if len(seg) < nperseg:                          # 末段不足则补零（scipy 亦如此）
            seg = np.concatenate([seg, np.zeros(nperseg - len(seg))])
        seg = seg - np.mean(seg)                        # detrend="constant"
        spec = np.fft.rfft(seg * win, nfft)
        p = (spec.real**2 + spec.imag**2) * scale
        if nfft % 2 == 0:
            p[1:-1] *= 2.0                              # 单边谱: 除 DC 与 Nyquist
        else:
            p[1:] *= 2.0
        psd = p if psd is None else psd + p
    psd = psd / nseg                                    # average="mean"

    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freq, psd


def _band_power(freq: np.ndarray, psd: np.ndarray, band: tuple) -> float:
    """`np.trapz` 在左闭右开频带上的积分; 结果为 0 时返回 NaN（hrvanalysis 的行为）。"""
    m = (freq >= band[0]) & (freq < band[1])
    power = float(np.trapz(y=psd[m], x=freq[m]))
    return np.nan if power == 0.0 else power


def freq_features(nn_ms: np.ndarray) -> dict:
    """频域特征：`vlf / lf / hf / lf_hf_ratio / total_power`。

    输入是原始 NN 间期（毫秒）。内部:
      1. 时间轴 `t = cumsum(nn)/1000`, 再整体平移使 `t[0] = 0`
      2. 线性插值到 `np.arange(0, t[-1], 1/4)` 的 4 Hz 均匀栅格
      3. **减去插值后序列的均值**去直流（⚠️ 不是原始 RR 的均值）
      4. Welch（见 `_welch_psd`）
      5. 按频带 trapz 积分
    """
    nn_ms = np.asarray(nn_ms, dtype=float)

    # 1. 时间轴（秒）
    t = np.cumsum(nn_ms) / 1000.0
    t = t - t[0]

    # 2. 4 Hz 均匀栅格（np.arange 不含右端点）
    grid = np.arange(0.0, t[-1], 1.0 / HRV_INTERP_FS)
    if len(grid) < 2:
        raise ValueError(f"插值栅格只有 {len(grid)} 点 —— RR 序列过短")
    y = np.interp(grid, t, nn_ms)              # 线性插值

    # 3. 去直流（用插值后序列的均值）
    y = y - np.mean(y)

    # 4. Welch
    nperseg = min(HRV_MAX_NPERSEG, len(y))
    freq, psd = _welch_psd(y, HRV_INTERP_FS, nperseg, HRV_NFFT)

    # 5. 频带积分
    vlf = _band_power(freq, psd, BAND_VLF)
    lf = _band_power(freq, psd, BAND_LF)
    hf = _band_power(freq, psd, BAND_HF)
    with np.errstate(divide="ignore", invalid="ignore"):
        lf_hf = lf / hf
    return {
        "vlf": vlf,
        "lf": lf,
        "hf": hf,
        "lf_hf_ratio": float(lf_hf),
        "total_power": vlf + lf + hf,
    }


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def hrv_features(nn_ms: np.ndarray) -> dict:
    """由一段 NN 间期（毫秒）算 7 个特征。

    对应离线路径 `hrv.py` 里这一段的等价物::

        vals = rr_ms[...]                    # 一个 epoch 或一个窗口内的 RR
        f = {}
        f.update(get_time_domain_features(vals))
        f.update(get_frequency_domain_features(vals))
        f.update(get_poincare_plot_features(vals))

    Parameters
    ----------
    nn_ms : array-like
        NN 间期, **毫秒**。

    Returns
    -------
    dict
        键: `median_nni` / `ratio_sd2_sd1` / `vlf` / `lf` / `hf` /
        `lf_hf_ratio` / `total_power`。

    Raises
    ------
    ValueError
        序列长度 < 2（复刻 hrvanalysis 的 `ZeroDivisionError` 语义, 见
        `_check_length`）。**调用方应当把整条 epoch 判为无效, 而不是只丢这一个特征。**
    """
    nn_ms = np.asarray(nn_ms, dtype=float)
    _check_length(nn_ms)

    out = {
        "median_nni": median_nni(nn_ms),
        "ratio_sd2_sd1": ratio_sd2_sd1(nn_ms),
    }
    out.update(freq_features(nn_ms))
    return out
