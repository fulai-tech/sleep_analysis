"""流式推理的滤波原语（自包含, 不依赖 scipy）。

**本文件是自包含实现, 不 import scipy / pandas。**

来源与内联理由
--------------
`sleep_analysis/preprocessing/iruwb/beat_detection.py::build_filter_chain` 用
`scipy.signal.butter` / `firwin` 现算系数, 用 `scipy.signal.lfilter` / `lfilter_zi` 滤波。
推理侧要翻译成 C++, 在 arm32 嵌入式上搬一个 Butterworth 设计器不划算, 所以这里
**把系数硬编码成常量** —— Python 与 C++ 共用同一组数值, C++ 侧零风险。

⚠️ **一致性契约**: 本文件的 `lfilter` 语义对齐 `scipy.signal.lfilter`,
`moving_average_causal` 对齐 `pd.Series.rolling(..., min_periods=1).mean()`。

**实测的精确度**（`seed=42` 随机信号 + 真实合成雷达信号）:

| 原语 | 对拍对象 | 结果 |
|---|---|---|
| `lfilter`（IIR, 无 zi） | `scipy.signal.lfilter` | **逐位一致** |
| `apply_iir_causal`（IIR + zi） | `scipy.signal.lfilter(..., zi=...)` | **逐位一致** |
| `lfilter`（FIR, 61 tap） | `scipy.signal.lfilter` | 差 ~9e-16（**相对 4.6e-16**，1 ULP 级） |
| `moving_average_causal` | `pd.Series.rolling().mean()` | 差 ~3e-16 |
| `apply_filter_chain` 端到端 | `beat_detection.apply_filter_chain` | 差 ~1.4e-14（相对 4.6e-16） |

**FIR / MA 做不到逐位一致, 且不建议追**：numpy / pandas 内部用 SIMD 分块累加,
复刻其浮点求和顺序不现实。实测这 1e-15 级差异**对下游峰检测无影响** ——
1 小时合成信号上峰数完全相同（3483 = 3483），峰位最大差 2.9e-15（RR 间期上约 3 飞秒）。

⇒ **C++ 翻译与对拍时**：IIR 用严格相等；FIR / MA / 端到端用 `rtol=1e-12` 级容差。
   C++ 侧写朴素的延迟线卷积即可, 不需要（也不可能）复刻 numpy 的累加顺序。

   对应的 scipy 调用（供对拍）::

       y = scipy.signal.lfilter(b, a, x)                 # 注意 zi 版本返回元组 (y, zf)
       y, _ = scipy.signal.lfilter(b, a, x, zi=scipy.signal.lfilter_zi(b, a) * x[0])

系数推导（生成命令, 换频段时重跑）
----------------------------------
    import scipy.signal as ss
    fs, nyq = 20.0, 10.0
    ss.butter(4, 0.5 / nyq, btype="high")            # → IIR_HP_B / IIR_HP_A
    ss.butter(4, (200/60) / nyq, btype="low")        # → IIR_LP_B / IIR_LP_A
    ss.firwin(61, [1.0, 5.0], pass_zero=False, fs=fs)  # → FIR_TAPS
    ss.lfilter_zi(IIR_HP_B, IIR_HP_A)                # → IIR_HP_ZI_UNIT
    ss.lfilter_zi(IIR_LP_B, IIR_LP_A)                # → IIR_LP_ZI_UNIT

参数含义（详见 `beat_detection.py` 模块文档）:
  - IIR 带通决定检测带, 由生理心率范围换算: 30–200 bpm → 0.5–3.33 Hz
  - FIR 是线性相位, 在 IIR 之后再补一层频率选择性; **下沿必须 ≥ 1.0 Hz**
    (因果 FIR 做出某个下沿, 抽头数得装得下至少一个周期的低频)
  - FIR 群延迟 = `(n_taps−1)//2` = 30 样本 = **1.5 s** @20 Hz。
    这个 1.5 s 就是 `geometry.GRID_SHIFT_S` 的主要来源。

⚠️ 这些取值都是在**合成信号**上调的, 未在真实数据上验证。
"""

import numpy as np

# ---------------------------------------------------------------------------
# 硬编码系数（float64 字面量, 由上面的 scipy 命令生成）
# ---------------------------------------------------------------------------

#: IIR 高通 4 阶, 截止 0.5 Hz (30 bpm)。分子。
IIR_HP_B = [0.81425455688624626, -3.257018227544985, 4.8855273413174771,
            -3.257018227544985, 0.81425455688624626]

#: IIR 高通 4 阶, 截止 0.5 Hz。分母（`a[0]` 恒为 1）。
IIR_HP_A = [1.0, -3.5897338871121756, 4.8512758825194169,
            -2.9240526561624587, 0.66301048438589105]

#: IIR 低通 4 阶, 截止 3.3333… Hz (200 bpm)。分子。
IIR_LP_B = [0.0260777217010923, 0.1043108868043692, 0.15646633020655382,
            0.1043108868043692, 0.0260777217010923]

#: IIR 低通 4 阶, 截止 3.3333… Hz。分母。
IIR_LP_A = [1.0, -1.3066051441010487, 1.0304538354195745,
            -0.36236904476885767, 0.055763900667808813]

#: 线性相位 FIR, 61 tap, 通带 [1.0, 5.0] Hz @20 Hz。
FIR_TAPS = [
    4.2705140428828538e-18, 0.00062695714932324586, -0.00060278740071890947,
    -0.0021901392204850025, -0.001396970950468496, -2.461160091002078e-19,
    -0.0021209563285069723, -0.0049696516272236301, -0.0019781887237915719,
    0.0028288366288402897, -4.8421940147820179e-18, -0.0040923849742414394,
    0.0041426436296591908, 0.015077833432758861, 0.0093236882061853085, 0.0,
    0.012738744688781299, 0.028204586561769045, 0.01065431622836968,
    -0.01456390347828466, 1.2027385133490823e-17, 0.019839187062896956,
    -0.019862403518860847, -0.072668686604822119, -0.046103080842955953, 0.0,
    -0.072800048057430594, -0.18794967823560113, -0.092770666427086049,
    0.21977649544134456, 0.40070015802125586, 0.21977649544134456,
    -0.092770666427086049, -0.18794967823560113, -0.072800048057430594, 0.0,
    -0.046103080842955953, -0.072668686604822133, -0.019862403518860847,
    0.019839187062896956, 1.2027385133490826e-17, -0.014563903478284665,
    0.01065431622836968, 0.028204586561769055, 0.012738744688781304, 0.0,
    0.0093236882061853137, 0.015077833432758869, 0.0041426436296591908,
    -0.0040923849742414411, -4.842194014782024e-18, 0.0028288366288402897,
    -0.0019781887237915732, -0.004969651627223637, -0.0021209563285069723,
    -2.4611600910020789e-19, -0.0013969709504684966, -0.0021901392204850025,
    -0.00060278740071890947, 0.00062695714932324586, 4.2705140428828538e-18,
]

#: `scipy.signal.lfilter_zi(IIR_HP_B, IIR_HP_A)` 的输出（单位输入下的稳态初值）。
#: 实际使用时乘 `x[0]`, 见 `apply_iir_causal`。
IIR_HP_ZI_UNIT = [-0.81425455688648862, 2.4427636706593665,
                  -2.4427636706592866, 0.81425455688640702]

#: `scipy.signal.lfilter_zi(IIR_LP_B, IIR_LP_A)` 的输出。
IIR_LP_ZI_UNIT = [0.97392227829890743, -0.43699375260651013,
                  0.43699375260651024, -0.029686178966716506]

#: FIR 的群延迟（样本）= `(len(FIR_TAPS) - 1) // 2`。线性相位下与频率无关。
FIR_DELAY_SAMPLES = (len(FIR_TAPS) - 1) // 2


# ---------------------------------------------------------------------------
# 滤波原语
# ---------------------------------------------------------------------------


def lfilter(b, a, x, zi=None):
    """因果 IIR 直接II型转置实现, 语义对齐 `scipy.signal.lfilter`。

    Parameters
    ----------
    b, a : sequence of float
        分子 / 分母系数。`a[0]` 必须为 1（本模块所有滤波器都满足）。
    x : np.ndarray
        输入信号, 一维。
    zi : sequence of float, optional
        延迟线初值, 长度须为 `max(len(a), len(b)) - 1`。
        注意 scipy 的 `lfilter_zi` 返回的是**单位输入**下的稳态初值,
        调用方需自行乘上 `x[0]` —— 本模块在 `apply_iir_causal` 里做了。

    Returns
    -------
    np.ndarray
        与 `x` 等长的滤波输出。

    Notes
    -----
    ⚠️ 求和顺序刻意与 scipy 的转置直接II型保持一致（右结合的延迟线累加）,
    改成朴素的从左到右卷积会让浮点末位不同, 破坏逐位对拍。
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    nb, na = len(b), len(a)
    nz = max(nb, na) - 1

    # ⚠️ scipy 把短的系数数组补零到等长再跑转置直接II型。FIR 的 a=[1.0] 比
    #    b 短 60 个, 不补零就会在 `a[j+1]` 越界。补零后 a 的高阶项恒为 0,
    #    对 FIR 无贡献, 同时保持与 scipy 相同的运算结构。
    bp = list(b) + [0.0] * (nz + 1 - nb)
    ap = list(a) + [0.0] * (nz + 1 - na)

    z = [0.0] * nz
    if zi is not None:
        for i in range(nz):
            z[i] = float(zi[i])

    b0 = bp[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        xi = x[i]
        yi = b0 * xi + z[0] if nz else b0 * xi
        for j in range(nz - 1):
            z[j] = bp[j + 1] * xi + z[j + 1] - ap[j + 1] * yi
        if nz:
            z[nz - 1] = bp[nz] * xi - ap[nz] * yi
        out[i] = yi
    return out


def apply_iir_causal(x, b, a, zi_unit):
    """单级因果 IIR, 带稳态初始化（消除启动瞬态）。

    等价于 `scipy.signal.lfilter(b, a, x, zi=scipy.signal.lfilter_zi(b, a) * x[0])`。

    ⚠️ 非因果分支的 `filtfilt`（零相位双向）**没有**内联 —— 实时推理用不到,
    它依赖未来样本。
    """
    zi = [v * x[0] for v in zi_unit]
    return lfilter(b, a, x, zi=zi)


def apply_filter_chain(x):
    """按 IIR 高通 → IIR 低通 → FIR 带通的顺序作用滤波链（全因果）。

    对应 `beat_detection.apply_filter_chain(x, chain, causal=True)`。
    """
    x = apply_iir_causal(x, IIR_HP_B, IIR_HP_A, IIR_HP_ZI_UNIT)
    x = apply_iir_causal(x, IIR_LP_B, IIR_LP_A, IIR_LP_ZI_UNIT)
    return lfilter(FIR_TAPS, [1.0], x)


def moving_average_causal(x, window):
    """因果滑动平均, 语义对齐 `pd.Series(x).rolling(window, center=False, min_periods=1).mean()`。

    首部窗宽自动缩短（`min_periods=1`）。

    ⚠️ 用**运行和**（加新值、减出窗值）而非每点重新求和 —— pandas 内部就是这么
    实现的, 重新求和会让浮点末位不同。
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    out = np.empty(n, dtype=float)
    s = 0.0
    for i in range(n):
        s += x[i]
        if i >= window:
            s -= x[i - window]
        out[i] = s / min(i + 1, window)
    return out


def fir_convolve_causal(x):
    """FIR 单级因果（`lfilter(FIR_TAPS, [1.0], x)`）。

    单独抽出来是因为它在一条整夜信号上最耗时, C++ 侧可以用环形缓冲优化。
    """
    return lfilter(FIR_TAPS, [1.0], x)
