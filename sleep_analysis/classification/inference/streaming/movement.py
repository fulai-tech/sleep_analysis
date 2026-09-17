"""流式推理的体动（ACT）支路 —— **唯一需要跨调用状态的支路**。

**本文件不 import pandas / scipy；滤波原语来自 `filters.py`。**

来源与内联理由
--------------
`sleep_analysis/preprocessing/iruwb/movement.py::extract_movement` 用 pandas 的
`rolling` / `expanding` 做四步变换。推理侧要翻译成 C++, 这里把用到的 pandas
原语按语义重写成**显式循环 + 跨调用累加器**。

⚠️ 为什么必须跨调用保留状态（这是本模块与其余支路最大的不同）
-----------------------------------------------------------
原实现的两个统计量是**会话锚定**的:

    _scale      : s.expanding(min_periods=1).mean() / .std()     # 对滑动平均序列
    _normalize  : s.expanding(min_periods=1).min() / .max()      # 对一阶导数序列

`expanding()` 的锚点是**传入数组的第 0 个元素**。离线跑整段时锚点是录制起点;
若每帧只传 900 s 窗口独立跑, 锚点就变成窗口起点 —— **实测入睡后的非零 epoch
从 280 个掉到 1 个（−99.6%）**, 特征基本全废。

所以本模块把这两个统计量做成**跨调用携带的累加器**（§`ActState`）。

⚠️ 另一个独立问题（本模块**不解决**, 只是忠实复现）
--------------------------------------------------
分母是"当晚最大值"意味着特征值取决于当晚最大的一次动作。床边部署时受试者
**入睡前在床边活动**会把分母撑大, 之后整夜的体动都被压到 0.2 阈值以下 ——
实测注入 10 分钟睡前活动即让入睡后非零 epoch 掉 99.6%。
这是**算法本身的设计缺陷**（从 D04 继承）, 与因果性/流式无关, 离线跑同样存在。
修法见 `movement.py` 模块文档（固定常数 / 长滚动窗）, 需另案处理。

处理链（与 `extract_movement` 逐行对应）
---------------------------------------
    滑动平均(10 s) → 标准化(expanding mean/std) → 一阶导数 →
    min-max 归一(expanding min/max) → 阈值 0.2 → 按 30 s 分组取均值

**实测精确度**（30 min 真实合成雷达信号）:

| 检查 | 结果 |
|---|---|
| 分段喂（每 600 点）vs 整段喂 | **逐位一致** ← 流式语义成立的关键证据 |
| `_acc_mean_1` 的 epoch 数 / 非零 epoch 数 | 一致（60/60，23/23） |
| `_acc_mean_1` 最大绝对差 | 5.8e-15（相对 1.8e-13） |

残余差异来自**滑动平均的运行和漂移**（pandas 的累加顺序与逐点运行和不同,
n 越大漂移越明显）。**稀疏结构完全一致**（哪些 epoch 为 0 一个不差）, 这对模型
才是关键的 —— 0.2 阈值让绝大多数 epoch 落在 0 上, 末位误差不会改变这个判定。

⚠️ **踩过的坑（已修）**: 环形缓冲长度必须等于滑动窗长（200）, 写成 199 会让最老的
样本永远减不掉, 滑动平均静默偏大, 实测 maxdiff 达 0.2（量级错误, 不是浮点）。
"""

from dataclasses import dataclass, field

import numpy as np

#: 滑动平均窗长（秒）。D04 原实现用 19000 样本 @1953.125 Hz = 9.728 s,
#: 本管线直接取 10 s —— 量级等价, 不维护采样率相关的换算。
MA_WINDOW_S = 10.0

#: 归一化后的阈值, 低于该值的样本置 0。
THRESHOLD = 0.2

#: 采样率（SDK 接口约定）。
FS = 20.0

#: 滑动平均窗长（样本数）。
MA_WINDOW = int(round(MA_WINDOW_S * FS))     # 200

#: 环形缓冲长度 = 滑动窗长。少一个槽会导致最老的样本减不掉, 滑动平均静默偏大。
_RING_SIZE = MA_WINDOW                        # 200


@dataclass
class ActState:
    """ACT 支路的跨调用状态（会话锚定统计量）。

    约 199 个 float + 7 个标量 —— 可直接映射成 C struct::

        struct ActState {
            double ma_ring[200];   // = MA_WINDOW，环形缓冲长度**必须等于滑动窗长**
            int    ring_pos;       // 环形写指针，取值 [0, 200)
            long   n_seen;         // 已见样本数
            double ma_sum;         // **原始样本**的运行和（供滑动平均用）
            double last_scaled;    // 上一个标准化值（求导用）
            int    has_last;       // last_scaled 是否有效
            double deriv_min;      // 一阶导数序列的 expanding min
            double deriv_max;      // 一阶导数序列的 expanding max
            /* 滑动平均**序列**的 expanding 统计量 —— 与上面的 ma_sum 是两套
               不同的累加器, 别混 */
            long   ma_n;           // 已累计的滑动平均值个数
            double ma_s1;          // Σ 滑动平均值
            double ma_s2;          // Σ 滑动平均值的平方
        };

    ⚠️ `ma_ring` 的长度是 **200**（`_RING_SIZE == MA_WINDOW`）, 与 `ring_pos % 200`
       的回绕范围一致。写成 199 会让 `ring[199]` 越界写 —— 这正是本模块文档里
       警告过的那个 bug（写成 199 时最老的样本永远减不掉, 滑动平均静默偏大）。

    ⚠️ 累加器**必须从会话起点开始维护**。中途重启（或每帧用新窗口的起点重新
       初始化）会让锚点前移, 特征失真 —— 见模块文档。
    进程重启后是否落盘由调用方决定（本项目当前的选择是**不落盘**, 接受重启偏差）。
    """

    ma_ring: np.ndarray = field(default_factory=lambda: np.zeros(_RING_SIZE))
    ring_pos: int = 0
    n_seen: int = 0
    ma_sum: float = 0.0
    last_scaled: float = 0.0
    has_last: bool = False
    deriv_min: float = np.inf
    deriv_max: float = -np.inf

    # 滑动平均**序列**的 expanding 统计量（与上面的 ma_sum「原始样本的运行和」
    # 是两套不同的累加器, 别混）
    ma_n: int = 0
    ma_s1: float = 0.0
    ma_s2: float = 0.0

    def copy(self) -> "ActState":
        """深拷贝（每帧重放前要拿一份起点状态, 不能就地改）。"""
        return ActState(
            ma_ring=self.ma_ring.copy(),
            ring_pos=self.ring_pos,
            n_seen=self.n_seen,
            ma_sum=self.ma_sum,
            last_scaled=self.last_scaled,
            has_last=self.has_last,
            deriv_min=self.deriv_min,
            deriv_max=self.deriv_max,
            ma_n=self.ma_n,
            ma_s1=self.ma_s1,
            ma_s2=self.ma_s2,
        )

    def process(self, samples: np.ndarray) -> np.ndarray:
        """把 `samples` 依次喂进状态机, 返回这些样本的**阈值化一阶导数**。

        就地修改自身状态（推进到 `samples` 末尾）。要保留起点状态请先 `copy()`。

        对应原实现的四步:

            ma      = _moving_average(x, 200, causal=True)   # 尾部窗, min_periods=1
            scaled  = _scale(ma, causal=True)                # (ma-mean)/std, expanding
            deriv   = |diff(scaled, prepend=scaled[0])|
            norm    = _normalize(deriv, causal=True)         # (d-lo)/(hi-lo), expanding
            thresh  = norm > 0.2 ? norm : 0
        """
        samples = np.asarray(samples, dtype=float).ravel()
        n = len(samples)
        out = np.empty(n, dtype=float)

        ring = self.ma_ring
        w = MA_WINDOW

        for i in range(n):
            x = samples[i]

            # --- 1. 滑动平均（尾部窗, min_periods=1）---
            # 运行和: 加新值、减出窗值。pandas 内部就是这么实现的, 每点重新求和
            # 会让浮点末位不同。
            if self.n_seen >= w:
                self.ma_sum -= ring[self.ring_pos]
            self.ma_sum += x
            ring[self.ring_pos] = x
            self.ring_pos = (self.ring_pos + 1) % _RING_SIZE
            count = min(self.n_seen + 1, w)
            ma = self.ma_sum / count

            # --- 2. 标准化（expanding mean / std, ddof=1）---
            self.n_seen += 1
            self.ma_n += 1
            self.ma_s1 += ma
            self.ma_s2 += ma * ma
            mean = self.ma_s1 / self.ma_n
            std = self._ma_running_std
            scaled = (ma - mean) / (std if std > 1e-12 else 1.0)

            # --- 3. 一阶导数的绝对值 ---
            # np.diff(x, prepend=x[0]) → 首元素恒为 0
            deriv = abs(scaled - self.last_scaled) if self.has_last else 0.0
            self.last_scaled = scaled
            self.has_last = True

            # --- 4. min-max 归一到 [0,1]（expanding min/max）---
            if deriv < self.deriv_min:
                self.deriv_min = deriv
            if deriv > self.deriv_max:
                self.deriv_max = deriv
            rng = self.deriv_max - self.deriv_min
            norm = (deriv - self.deriv_min) / (rng if abs(rng) > 1e-12 else 1.0)

            # --- 5. 阈值 ---
            out[i] = norm if norm > THRESHOLD else 0.0

        return out

    @property
    def _ma_running_std(self) -> float:
        """样本标准差（ddof=1），对应 `pd.Series.expanding(min_periods=2).std()`。

        ⚠️ `n < 2` 时 pandas 给 NaN, 原实现 `np.where(isfinite & >1e-12, std, 1.0)`
           会兜成 1.0 —— 这里直接返回 1.0, 等价。
        """
        if self.ma_n < 2:
            return 1.0
        var = (self.ma_s2 - self.ma_s1 * self.ma_s1 / self.ma_n) / (self.ma_n - 1)
        return float(np.sqrt(max(var, 0.0)))


def epoch_means(thresh: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """把阈值化序列按给定的样本区间取均值（对应原实现的 `groupby(30s).mean()`）。

    Parameters
    ----------
    thresh : np.ndarray
        `ActState.process` 的输出, 与窗口样本一一对应。
    starts, ends : np.ndarray
        每个 epoch 在**窗口内**的起止样本下标（左闭右开）。

    Returns
    -------
    np.ndarray
        每个 epoch 的均值。区间为空时返回 0.0（原实现对空组也会产出 NaN,
        但我们的 epoch 网格保证区间非空）。
    """
    out = np.empty(len(starts), dtype=float)
    for k, (a, b) in enumerate(zip(starts, ends)):
        a, b = int(a), int(b)
        out[k] = float(np.mean(thresh[a:b])) if b > a else 0.0
    return out
