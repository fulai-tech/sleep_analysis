"""IR-UWB 雷达生理信号仿真器 — 为 20 Hz 预处理管线提供带真值的测试数据。

背景
----
我们的 IR-UWB 雷达输出的是**呼吸 + 心跳混叠的胸腔位移信号**，采样率 20 Hz
（硬件硬上限）。真实数据尚未采集，本模块生成仿真信号用于把预处理管线跑通并
定量评估各检测器的精度。

设计要点
--------
1. **事件真值是核心资产**。仿真先生成"逐次呼吸时刻"和"逐拍心搏时刻"这两个
   亚采样精度的点过程，再据此渲染波形。有了真值，"检测器误差"和"特征计算误差"
   才能分开度量 —— 否则只能用参考采样率互相对比，无法定位误差来源。

2. **幅度比必须真实**。真实胸腔位移：呼吸 5–20 mm、心跳 0.2–0.5 mm，比值
   20–50×。若仿真时给成同一量级，心搏检测会显得很好而真机必崩。这里用
   ``resp_amp_mm`` / ``card_amp_mm`` 显式控制，默认比值约 34×。

3. **RSA 必须与呼吸相位锁定**。呼吸性窦性心律不齐是 HF 频段 HRV 的物理来源，
   若心搏的 RR 调制与呼吸各自独立生成，HF 特征就失去意义。

4. **同一场景可渲染到任意采样率**。``render_waveform`` 接受 fs 参数，事件时刻
   不变 —— 这样就能用高采样率（如 200 Hz）的渲染结果作为参考基线，量化 20 Hz
   带来的损失。

用法
----
    from sleep_analysis.preprocessing.iruwb.simulate import SimConfig, simulate_subject

    cfg = SimConfig(duration_s=8 * 3600, seed=42)
    data = simulate_subject(cfg, subj_id="01")

    data["signal"]          # 20 Hz 波形 (np.ndarray)
    data["breath_times"]    # 逐次呼吸时刻 (秒, 亚采样)
    data["beat_times"]      # 逐拍心搏时刻 (秒, 亚采样)
    data["stages"]          # 逐 30 s epoch 分期
    data["motion_times"]    # 体动事件起始时刻 (秒)

命令行入口见 ``experiments/data_handling/simulate_iruwb.py``。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 生理参数默认值 — 单位: 毫米 (胸腔位移), 秒, 次/分
# ---------------------------------------------------------------------------

# 真实胸腔位移量级 (参考文献量级, 用于约束仿真幅度比)
RESP_DISPLACEMENT_MM = (5.0, 20.0)      # 呼吸引起的胸廓位移
CARD_DISPLACEMENT_MM = (0.2, 0.5)       # 心搏引起的胸壁微动 (BCG)

# 睡眠分期编码 (与 sleep_analysis 全仓库一致)
#   5stage: 0=Wake 1=N1 2=N2 3=N3 4=REM
#   4stage: 0=Wake 1=Light(N1+N2) 2=Deep(N3) 3=REM
#   3stage: 0=Wake 1=NREM 2=REM
STAGE_5_TO_4 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
STAGE_5_TO_3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}

# 正常夜间各期占比 (Wake 之外的睡眠期归一化后用于合成 hypnogram)
STAGE_PROPORTIONS = {0: 0.057, 1: 0.073, 2: 0.494, 3: 0.180, 4: 0.196}

# ---------------------------------------------------------------------------
# 分期对生理信号的调制系数
# ---------------------------------------------------------------------------
# ⚠️ 没有这些调制, hypnogram 与信号就是**互相独立**的 —— 模型不可能从特征里
# 学到任何东西, 仿真只能验证管线连通性, 无法验证"模型能学到"。
# 加了调制之后, 特征里才携带分期信息, 训练应当能显著超过多数类基线。
#
# 生理依据:
#   呼吸频率  — N3 深睡最慢最规律, REM 快且不规则, Wake 最快
#   RSA 幅度  — NREM 副交感张力高 (RSA 大), REM 交感激活 (RSA 显著降低)
#   体动      — Wake 最多, N3 最少
STAGE_RESP_RATE_FACTOR = {0: 1.20, 1: 1.00, 2: 1.00, 3: 0.85, 4: 1.15}
STAGE_RSA_FACTOR = {0: 0.70, 1: 0.90, 2: 1.00, 3: 1.30, 4: 0.60}
STAGE_MOTION_FACTOR = {0: 3.0, 1: 1.5, 2: 1.0, 3: 0.5, 4: 1.2}


# 随机游走的 OU 衰减系数 (逐次呼吸 / 逐拍)。越接近 1, 相关时间越长。
_OU_DECAY = 0.995
_OU_DECAY_HR = 0.999


def _ou_step(decay: float) -> float:
    """由目标稳态标准差反推 OU 每步的噪声强度: σ_step = std · √(1−decay²)。"""
    return float(np.sqrt(1.0 - decay ** 2))


def _stage_series(stages: np.ndarray, t: np.ndarray, epoch_s: float,
                  table: dict) -> np.ndarray:
    """把逐 epoch 的分期标签映射成逐样本的调制系数。"""
    if len(stages) == 0:
        return np.ones_like(t)
    idx = np.clip((t / epoch_s).astype(int), 0, len(stages) - 1)
    lookup = np.array([table.get(int(s), 1.0) for s in stages])
    return lookup[idx]


def _stage_factor_at(t: float, stages, epoch_s: float, table: dict) -> float:
    """标量版: 取时刻 t 所处 epoch 的调制系数。"""
    if stages is None or len(stages) == 0:
        return 1.0
    i = min(max(int(t / epoch_s), 0), len(stages) - 1)
    return float(table.get(int(stages[i]), 1.0))


@dataclass
class SimConfig:
    """仿真参数。所有时间单位秒, 幅度单位毫米。"""

    fs: float = 20.0                    # 采样率 (Hz)
    duration_s: float = 8 * 3600.0      # 录制时长 (默认整夜)
    seed: int = 0

    # --- 呼吸 ---
    resp_rate_hz: float = 0.25          # 平均呼吸率 (0.25 Hz = 15 次/分)
    resp_rate_drift_hz: float = 0.05    # 呼吸率的慢漂移幅度
    resp_drift_period_s: float = 300.0  # 慢漂移周期
    resp_interval_cv: float = 0.05      # 逐次呼吸间隔的变异系数 (RRV 的物质基础)
    resp_lf_cv: float = 0.02            # 0.04-0.15 Hz 频段的额外变异
    resp_amp_mm: float = 12.0           # 呼吸幅度 (胸腔位移 mm, 峰到谷的一半)
    resp_rate_wander_hz: float = 0.02   # 基线呼吸率随机游走的**稳态标准差** (Hz)

    # --- 心搏 ---
    heart_rate_bpm: float = 62.0
    heart_rate_wander_bpm: float = 4.0  # 基线心率随机游走的**稳态标准差** (bpm)
    rsa_ms: float = 45.0                # RSA (呼吸性窦性心律不齐) 幅度, 锁到呼吸相位
    lf_ms: float = 25.0                 # LF 频段 RR 调制幅度
    vlf_ms: float = 30.0                # VLF 频段慢漂移
    rr_noise_ms: float = 8.0            # 逐拍随机噪声
    card_amp_mm: float = 0.35           # 心搏幅度 (胸腔位移 mm)
    card_j_sigma_s: float = 0.04        # J 峰宽度
    card_k_ratio: float = 0.35          # K 峰相对 J 峰幅度
    card_k_delay_s: float = 0.12        # K 峰延迟

    # --- 运动伪迹与环境 ---
    motion_rate_per_h: float = 6.0      # 每小时运动事件数
    motion_dur_s: float = 8.0           # 单次运动持续
    motion_amp_mm: float = 40.0         # 运动幅度 (远大于生理信号)
    noise_mm: float = 0.015             # 加性白噪声标准差

    # --- 睡眠分期 ---
    epoch_s: float = 30.0
    stage_seed_offset: int = 1000       # 与信号种子隔离, 便于单独复现

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 点过程生成
# ---------------------------------------------------------------------------

def generate_breath_times(cfg: SimConfig, rng: np.random.Generator,
                          stages=None) -> np.ndarray:
    """生成逐次呼吸的**吸气峰**时刻 (秒, 亚采样精度)。

    建模为"瞬时呼吸率随时间慢漂移"的非齐次点过程:
        呼吸率(t) = resp_rate_hz + drift * sin(2*pi*t/drift_period) + LF 分量
        间隔 = 1/呼吸率 * (1 + N(0, resp_interval_cv) + LF 抖动)

    逐次间隔的变异正是 RRV 测量的对象, 所以 ``resp_interval_cv`` 与
    ``resp_lf_cv`` 直接决定了仿真数据里 RRV 特征的信号强度。
    """
    t = 0.0
    times = []
    walk = 0.0
    while t < cfg.duration_s:
        # 基线随机游走 (OU 型)。步长由**稳态标准差**反推:
        #   OU: x ← a·x + N(0, σ)  ⇒  稳态 std = σ/√(1−a²)
        # 限幅 ±3σ, 避免偶发长游走把呼吸率推到非生理值。
        walk = float(np.clip(
            walk * _OU_DECAY + rng.normal(0.0, cfg.resp_rate_wander_hz * _ou_step(_OU_DECAY)),
            -3 * cfg.resp_rate_wander_hz, 3 * cfg.resp_rate_wander_hz))
        rate = cfg.resp_rate_hz + walk
        rate += cfg.resp_rate_drift_hz * np.sin(2 * np.pi * t / cfg.resp_drift_period_s)
        # 0.04-0.15 Hz 频段: 用固定频率的相位项近似 (多个非整周期分量叠加)
        rate += cfg.resp_rate_hz * cfg.resp_lf_cv * np.sin(2 * np.pi * 0.1 * t)
        # 分期调制: N3 深睡呼吸最慢、REM/Wake 较快（见 STAGE_RESP_RATE_FACTOR）
        rate *= _stage_factor_at(t, stages, cfg.epoch_s, STAGE_RESP_RATE_FACTOR)
        rate = max(rate, 0.05)   # 防止游走到非生理值

        interval = (1.0 / rate) * (1.0 + rng.normal(0.0, cfg.resp_interval_cv))
        t += interval
        times.append(t)
    return np.asarray(times[:-1], dtype=float)


def _respiratory_phase(t: float, breath_times: np.ndarray) -> float:
    """把时刻 t 映射到呼吸相位 (弧度)。

    在相邻呼吸时刻之间线性插值相位, 保证 RSA 与呼吸严格锁相。
    """
    idx = np.searchsorted(breath_times, t)
    if idx <= 0:
        return 0.0
    if idx >= len(breath_times):
        return 2 * np.pi * (len(breath_times) - 1)
    t0, t1 = breath_times[idx - 1], breath_times[idx]
    frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return 2 * np.pi * (idx - 1 + frac)


def generate_beat_times(cfg: SimConfig, breath_times: np.ndarray,
                        rng: np.random.Generator, stages=None) -> np.ndarray:
    """生成逐拍心搏时刻 (秒, 亚采样精度)。

    ``RR = 60000/HR + RSA*sin(呼吸相位) + LF + VLF + 噪声``

    RSA 项使用 ``_respiratory_phase`` 得到与呼吸严格锁相的相位 —— 这是 HF 频段
    HRV 的物理来源, 必须锁相才有意义。
    """
    t = 0.0
    times = []
    walk_bpm = 0.0
    while t < cfg.duration_s:
        # 基线随机游走 (OU 型, 同上)。_OU_DECAY_HR 稍慢, 对应心率变化的分钟级时间尺度。
        walk_bpm = float(np.clip(
            walk_bpm * _OU_DECAY_HR + rng.normal(0.0, cfg.heart_rate_wander_bpm * _ou_step(_OU_DECAY_HR)),
            -3 * cfg.heart_rate_wander_bpm, 3 * cfg.heart_rate_wander_bpm))
        hr_bpm = float(np.clip(cfg.heart_rate_bpm + walk_bpm, 35.0, 180.0))
        mean_rr_s = 60.0 / hr_bpm

        phase = _respiratory_phase(t, breath_times)
        rr = mean_rr_s
        # RSA 幅度按分期缩放: NREM 副交感张力高, REM 显著降低
        # （HF 频段 HRV 的分期判别力主要来自这里）
        rsa_ms = cfg.rsa_ms * _stage_factor_at(t, stages, cfg.epoch_s, STAGE_RSA_FACTOR)
        rr += rsa_ms / 1000.0 * np.sin(phase)                  # RSA, 锁呼吸相位
        rr += cfg.lf_ms / 1000.0 * np.sin(2 * np.pi * 0.1 * t)  # LF (0.1 Hz)
        rr += cfg.vlf_ms / 1000.0 * np.sin(2 * np.pi * 0.02 * t)  # VLF (0.02 Hz)
        rr += rng.normal(0.0, cfg.rr_noise_ms / 1000.0)
        rr = max(rr, 0.25)  # 生理下限 ~240 bpm, 防止数值上出现负间隔
        t += rr
        times.append(t)
    return np.asarray(times[:-1], dtype=float)


# ---------------------------------------------------------------------------
# 波形渲染
# ---------------------------------------------------------------------------

def _breath_phase(t: np.ndarray, breath_times: np.ndarray) -> np.ndarray:
    """把时间轴映射到呼吸相位 (弧度), 使 ``breath_times[i]`` 处相位 = ``2*pi*i``。

    相邻呼吸时刻之间**线性推进**相位, 所以 ``cos(phase)`` 的峰值精确落在
    ``breath_times`` 上, 且瞬时周期等于该次呼吸间隔 —— 这正是 RRV 要测量的量。

    首尾各外推一个虚拟呼吸时刻, 避免边界处相位截断。
    """
    if len(breath_times) == 0:
        return np.zeros_like(t)
    if len(breath_times) == 1:
        return 2 * np.pi * t / max(breath_times[0], 1e-9)

    t0, t1 = breath_times[0], breath_times[-1]
    ext = np.concatenate([[t0 - (breath_times[1] - t0)], breath_times,
                          [t1 + (t1 - breath_times[-2])]])
    i = np.clip(np.searchsorted(ext, t, side="right") - 1, 0, len(ext) - 2)
    frac = (t - ext[i]) / np.maximum(ext[i + 1] - ext[i], 1e-9)
    return 2 * np.pi * ((i - 1) + frac)


def render_waveform(cfg: SimConfig, breath_times: np.ndarray, beat_times: np.ndarray,
                    rng: np.random.Generator, fs: Optional[float] = None,
                    motion: bool = True, stages=None) -> tuple:
    """把事件时刻渲染成胸腔位移波形 (mm)。

    事件时刻保持不变, 只改变渲染采样率 —— 因此可以用高采样率渲染作为参考基线。

    Parameters
    ----------
    fs : float, optional
        渲染采样率, 默认取 ``cfg.fs``。传更高值即可得到"参考"信号。
    motion : bool
        是否叠加运动伪迹 (参考基线通常也保留, 以保持场景一致)。

    Returns
    -------
    (signal, resp_component, card_component, motion_times)
        总信号、纯呼吸分量、纯心搏分量、体动事件起始时刻(秒)。
        后两者用于诊断与验证(体动时刻是仿真真值, 可用来核对 ACT 特征)。
    """
    fs = cfg.fs if fs is None else fs
    n = int(round(cfg.duration_s * fs))
    t = np.arange(n) / fs

    # --- 呼吸分量: 相位锁定的连续振荡 (吸气峰落在 breath_times 上) ---
    #
    # 用"相位在相邻呼吸时刻间线性推进"的余弦: 峰值精确落在 breath_times 上,
    # 且波形连续。不用高斯脉冲串 —— 那会让每次呼吸成为窄峰、间隔处近乎为零,
    # 既不像真实呼吸(吸气与呼气都会使胸廓位移), 也会让下游
    # `nk.rsp_peaks(method="biosppy")` 抛 IndexError。
    phase = _breath_phase(t, breath_times)
    # 幅度随呼吸次数缓慢起伏 (真实呼吸幅度并不恒定),
    # 按呼吸序号插值到时间轴, 避免各段幅度不连续造成峰处跳变
    if len(breath_times) >= 2:
        amp_per_breath = cfg.resp_amp_mm * (1.0 + 0.08 * rng.standard_normal(len(breath_times)))
        amp_smooth = np.interp(t, breath_times, amp_per_breath,
                               left=amp_per_breath[0], right=amp_per_breath[-1])
    else:
        amp_smooth = np.full(n, cfg.resp_amp_mm)
    resp = amp_smooth * np.cos(phase)

    # --- 心搏分量: 每拍一个 J 峰 + 一个延迟的负向 K 峰 (BCG 样) ---
    card = np.zeros(n)
    sj = cfg.card_j_sigma_s
    for bt in beat_times:
        lo = max(0, int((bt - 4 * sj) * fs))
        hi = min(n, int((bt + 4 * sj) * fs) + 1)
        if hi > lo:
            tt = t[lo:hi] - bt
            card[lo:hi] += cfg.card_amp_mm * np.exp(-(tt ** 2) / (2 * sj ** 2))
        # K 峰 (负向)
        tk = bt + cfg.card_k_delay_s
        lo = max(0, int((tk - 4 * sj) * fs))
        hi = min(n, int((tk + 4 * sj) * fs) + 1)
        if hi > lo:
            tt = t[lo:hi] - tk
            card[lo:hi] -= cfg.card_amp_mm * cfg.card_k_ratio * np.exp(-(tt ** 2) / (2 * sj ** 2))

    signal = resp + card

    # --- 运动伪迹 ---
    motion_times = []
    if motion and cfg.motion_rate_per_h > 0:
        n_events = int(round(cfg.motion_rate_per_h * cfg.duration_s / 3600.0))
        for start in rng.uniform(0, max(cfg.duration_s - cfg.motion_dur_s, 1e-6), n_events):
            motion_times.append(float(start))
            # 分期调制: Wake 体动最多、N3 最少（ACT 模态的分期判别力来源）
            amp_scale = _stage_factor_at(start + cfg.motion_dur_s / 2, stages,
                                         cfg.epoch_s, STAGE_MOTION_FACTOR)
            lo = int(start * fs)
            hi = min(n, int((start + cfg.motion_dur_s) * fs))
            if hi <= lo:
                continue
            # 带限的瞬态扰动, 幅度远大于生理信号
            seg = cfg.motion_amp_mm * amp_scale * rng.standard_normal(hi - lo)
            # 平滑一下, 避免成为纯白噪 (真实体动是低频大位移)
            if len(seg) > 5:
                k = np.ones(5) / 5
                seg = np.convolve(seg, k, mode="same")
            signal[lo:hi] += seg

    # --- 加性噪声 ---
    signal = signal + rng.normal(0.0, cfg.noise_mm, n)

    return signal, resp, card, np.sort(np.asarray(motion_times))


# ---------------------------------------------------------------------------
# 合成睡眠分期
# ---------------------------------------------------------------------------

def generate_hypnogram(cfg: SimConfig) -> np.ndarray:
    """合成一个合理的整夜 hypnogram, 返回逐 30 s epoch 的 5stage 标签。

    做法: 用睡眠周期结构 —— 每 ~90 min 一个 NREM→REM 循环, 循环内按
    N2/N3/N1 的顺序过渡, 并让 N3 集中在前半夜、REM 集中在后半夜 (真实的
    "深睡前半夜、REM 后半夜"分布)。各期占位接近正常成人夜间比例。

    这只是为了让端到端管线（特征表 ↔ 标签对齐、训练脚本）能跑通,
    不追求与真实个体一致。
    """
    rng = np.random.default_rng(cfg.seed + cfg.stage_seed_offset)
    n_epochs = int(cfg.duration_s / cfg.epoch_s)
    stages = np.zeros(n_epochs, dtype=int)

    # 入睡潜伏期 (Wake)
    latency = int(rng.uniform(10, 25) * 60 / cfg.epoch_s)
    latency = min(latency, n_epochs // 4)

    cycle_epochs = int(90 * 60 / cfg.epoch_s)
    i = latency
    cycle = 0
    while i < n_epochs:
        progress = i / n_epochs                    # 0=前半夜 1=后半夜
        # N3 占比前半夜高、后半夜低
        n3_frac = max(0.0, 0.45 * (1 - 1.4 * progress))
        n2_frac = max(0.0, 0.45 * (1 - 0.3 * progress))
        rem_frac = min(0.35, 0.10 + 0.30 * progress)
        n1_frac = max(0.0, 1.0 - n3_frac - n2_frac - rem_frac)

        block = int(cycle_epochs * rng.uniform(0.9, 1.1))
        # 周期内顺序: N1 → N2 → N3 → N2 → REM
        pattern = ([1] * int(block * n1_frac / 3) + [2] * int(block * n2_frac / 2)
                   + [3] * int(block * n3_frac) + [2] * int(block * n2_frac / 2)
                   + [4] * int(block * rem_frac))
        if not pattern:
            pattern = [2]
        pattern += [1] * int(block * n1_frac / 3)   # 周期尾部回到浅睡
        for s in pattern:
            if i >= n_epochs:
                break
            stages[i] = s
            i += 1
        cycle += 1

    # 醒来前的短暂觉醒
    n_wake = int(rng.uniform(2, 8) * 60 / cfg.epoch_s)
    n_wake = min(n_wake, max(0, n_epochs - i))
    stages[i:i + n_wake] = 0

    return stages


def stages_to_table(stages_5: np.ndarray) -> dict:
    """5stage 标签 → 与仓库其他数据集一致的 sleep/5stage/4stage/3stage 四套列。"""
    stages_5 = np.asarray(stages_5, dtype=int)
    return {
        "5stage": stages_5,
        # binary sleep: 仅 5stage != 0 的 epoch 为 Sleep=1
        "sleep": (stages_5 != 0).astype(int),
        "4stage": np.array([STAGE_5_TO_4[int(s)] for s in stages_5], dtype=int),
        "3stage": np.array([STAGE_5_TO_3[int(s)] for s in stages_5], dtype=int),
    }


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

def simulate_subject(cfg: SimConfig, subj_id: str = "01",
                     include_components: bool = False) -> dict:
    """生成一个被试的完整仿真数据。

    Returns
    -------
    dict
        ``subj_id`` / ``fs`` / ``signal`` / ``breath_times`` / ``beat_times`` /
        ``stages``(含 5stage/sleep/4stage/3stage) / ``config``
        当 ``include_components=True`` 时额外返回 ``resp_component`` /
        ``card_component``, 用于诊断呼吸-心搏分离难度。
    """
    rng = np.random.default_rng(cfg.seed)
    # 先生成分期 —— 呼吸率 / RSA 幅度 / 体动都按分期调制, 特征才携带分期信息
    stages = stages_to_table(generate_hypnogram(cfg))

    breath_times = generate_breath_times(cfg, rng, stages=stages["5stage"])
    beat_times = generate_beat_times(cfg, breath_times, rng, stages=stages["5stage"])
    signal, resp, card, motion_times = render_waveform(
        cfg, breath_times, beat_times, rng, stages=stages["5stage"])

    out = {
        "subj_id": subj_id,
        "fs": cfg.fs,
        "signal": signal,
        "breath_times": breath_times,
        "beat_times": beat_times,
        "stages": stages,
        "motion_times": motion_times,
        "config": cfg.to_dict(),
    }
    if include_components:
        out["resp_component"] = resp
        out["card_component"] = card
    return out


def render_at_rate(cfg: SimConfig, breath_times: np.ndarray, beat_times: np.ndarray,
                   fs: float, seed: Optional[int] = None) -> np.ndarray:
    """把**同一组事件时刻**渲染到指定采样率 —— 用于生成参考基线。

    噪声与运动伪迹用独立的 RNG 流 (种子派生自 ``seed``), 因此不同 fs 下
    渲染的是"同一场景的两次独立观测", 而不是逐点对应的同一信号。做参考基线
    对比时这是恰当的: 两者应当共享生理事件, 但采样噪声本就不同。
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    stages = stages_to_table(generate_hypnogram(cfg))["5stage"]
    signal, _, _, _ = render_waveform(cfg, breath_times, beat_times, rng, fs=fs,
                                      stages=stages)
    return signal
