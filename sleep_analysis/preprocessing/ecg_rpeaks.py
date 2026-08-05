"""因果 R 峰检测 — Pan-Tompkins 简化变体（仅使用已见数据）。

当前状态（2026-08-05）: 未启用。HRV 相关处理已回退原版
（见 experiments/data_handling/preprocess_shhs.py::generate_r_points_from_ecg 的
历史说明）。本模块保留供将来参考 — 启用方式: 在 generate_r_points_from_ecg
的 causal 分支取消注释。已通过验证:
  - 合成 ECG（60/75/90 BPM, 噪声 0.05-0.2）: RR 间期中位误差 0ms
  - 截断不变性: 完全一致（严格因果）
  - 真实 SHHS1 9h ECG: 15s 检出 32321 峰 vs neurokit2 32898（差 1.8%）

neurokit2 的 ``nk.ecg_peaks``（method="neurokit"）整体非因果，无法在因果模式下复用:
  - np.gradient 中心差分依赖 x[i+1]
  - signal_smooth 对称卷积（用未来样本）
  - 梯度阈值、QRS 最短长度阈值都是全局统计（用整段信号）
  - find_peaks 的 prominence 是段内全局计算

因此 causal 分支用本模块的自写实现，所有步骤严格因果（t 时刻输出只依赖 [0, t]）:
  1. butter(2) bandpass 5-15 Hz + lfilter（正向滤波, zi 稳态初始化）
  2. 一阶差分 + 平方（能量变换）
  3. 150 ms 滑动窗口积分（trailing, np.convolve 因果卷积）
  4. 自适应阈值: mwi 的因果 EMA 基线（时间常数 2 s, 长于峰间隔使其反映平均能量而非峰近值）
  5. 上穿段内找滤波信号 x 的局部最大（R 峰 = QRS 段内最大正峰; 段边界用 mwi 确定）
  6. 200 ms 不应期过滤
  7. 开头 2 s 不检测（EMA 爬升期, 阈值未稳定; 对整夜数据无实际影响）

滤波的恒定群延迟只整体平移峰位置, 不改变峰间隔 — 对 RR 间期 / HRV 特征无影响。
"""

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


def rpeaks_causal(ecg: np.ndarray, sampling_rate: int) -> np.ndarray:
    """检测 ECG 中的 R 峰位置（因果实现）。

    Parameters
    ----------
    ecg : np.ndarray
        原始 ECG 信号（1-D）
    sampling_rate : int
        采样率 (Hz)

    Returns
    -------
    np.ndarray
        R 峰样本索引（int），按时间升序
    """
    ecg = np.asarray(ecg, dtype=float)
    fs = int(sampling_rate)

    # 1. 带通滤波 (5-15 Hz, QRS 能量带) — 因果正向滤波
    b, a = butter(N=2, Wn=[5.0, 15.0], btype="bandpass", fs=fs)
    zi = lfilter_zi(b, a) * ecg[0]
    x, _ = lfilter(b, a, ecg, zi=zi)

    # 2. 差分 + 平方
    diff = np.diff(x, prepend=x[0])
    sq = diff * diff

    # 3. 150 ms 滑动窗口积分 (trailing, np.convolve full 模式是因果卷积)
    win = max(1, int(0.15 * fs))
    mwi = np.convolve(sq, np.ones(win) / win, mode="full")[: len(sq)]

    # 4. 自适应阈值: 因果 EMA 基线, 时间常数 2 s (长于峰间隔 ~0.8 s, 反映平均能量)
    tau = max(1, int(2.0 * fs))
    alpha = 1.0 / tau
    baseline = lfilter([alpha], [1.0, alpha - 1.0], mwi)  # y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    c = 2.5
    threshold = c * baseline

    # 7. 开头 2 s 不检测 (EMA 爬升期, 阈值未稳定)
    skip = min(len(mwi), int(2 * fs))

    # 5. 候选 + 上穿段内找 x 的局部最大
    mask = mwi > threshold
    mask[:skip] = False
    rise = mask & ~np.roll(mask, 1)
    rise[0] = False
    fall = mask & ~np.roll(mask, -1)

    peaks = []
    for beg in np.flatnonzero(rise):
        end_candidates = np.flatnonzero(fall[beg:])
        if end_candidates.size == 0:
            end = len(mwi) - 1
        else:
            end = beg + end_candidates[0]
        seg = x[beg : end + 1]
        peaks.append(beg + int(np.argmax(seg)))

    # 6. 200 ms 不应期过滤
    ref_period = int(0.2 * fs)
    filtered = []
    for p in peaks:
        if not filtered or p - filtered[-1] > ref_period:
            filtered.append(p)

    return np.asarray(filtered, dtype=int)
