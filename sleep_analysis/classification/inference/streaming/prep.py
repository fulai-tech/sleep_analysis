"""流式预处理编排 —— 13 槽位装配 + `StreamingPreprocessor`。

**本文件是推理侧的对外入口。** 它把三条支路（体动 / 心率变异 / 呼吸变异）在
900 s 窗口上组装成模型要的 `(21, 13)` 特征矩阵, 形态严格镜像 C ABI 接口
`SleepStagingInterface/include/sleep_staging_realtime.h` 的 `ss_rt_process`。

与 C++ 的对应关系
----------------
======================  ==================================================
本文件                   C++ 侧
======================  ==================================================
`SLOT_NAMES`            常量表
`WindowResult`          `struct ss_rt_result` + 特征缓冲
`StreamingPreprocessor` `RtContext`（含 `ActState` 的跨调用状态）
`.update()`             `ss_rt_process()`
`.reset()`              `ss_rt_init()` 时对上下文的清零
======================  ==================================================

几何（详见 `geometry.py`）
--------------------------
    输入 900 s @20 Hz = 18000 样本, latest = 最后一个样本时刻
    epoch 网格: k×30 − 1.55 秒
    完整 epoch: k=1..29;  模型窗 = k=9..29（21 个, 居中）
    产出 epoch = 窗内第 10 个（k=19）= [latest−331.55s, latest−301.55s]

跨调用状态
----------
**只有 ACT 支路有状态**（`ActState`）—— 见 `movement.py` 的说明。HRV / RRV 的窗口
都有界（≤270 s）, 每帧从 900 s 缓冲重算即可, 与离线结果逐位一致。

每帧 ACT 的处理（避免把重叠的 870 s 重复累加）
---------------------------------------------
1. 拿 `state_at_window_start` 的**副本**重放整个 900 s 窗口 → 得到窗口内每个样本的
   阈值化导数 → 算 21 个 epoch 的均值
2. 另取一份起点状态的副本, 只喂**前 30 s（600 点）** → 存回, 作为下一次调用的
   `state_at_window_start`（相邻窗口错开 30 s, 所以下一次的起点就是本次的第 30 s）

这样既拿到 ACT 需要的整窗导出序列, 又不会把重叠部分重复累加进 expanding 统计量。
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import geometry as G
from . import movement as M
from .beats import detect_beats
from .hrv import hrv_features
from .resp import rrv_features

#: 13 个训练槽位, 顺序即模型输入的列序（与 `base_sleep_dataset.FEATURE_COLUMNS` 一致,
#: 取 pipeline.py 文档口径: 7 个逐 epoch HRV + 槽位 3 用 150 s 窗口）。
SLOT_NAMES: List[str] = [
    "_acc_mean_1",          # 1  体动
    "_hrv_median_nni",      # 2  ┐
    "_hrv_ratio_sd2_sd1",   # 3  │
    "150_hrv_median_nni",   # 4  │ 150 s 窗口（其余 7 个是逐 epoch）
    "_hrv_vlf",             # 5  │
    "_hrv_lf",              # 6  │
    "_hrv_hf",              # 7  │
    "_hrv_lf_hf_ratio",     # 8  │
    "_hrv_total_power",     # 9  ┘
    "150_RRV_MedianBB",     # 10 ┐
    "150_RRV_LF",           # 11 │ 呼吸变异
    "270_RRV_MCVBB",        # 12 │
    "150_RRV_CVBB",         # 13 ┘
]

#: 每个 HRV 槽位对应的窗口长度（秒）。`None` = 本 epoch。
_HRV_SLOT_WINDOW = {
    "_hrv_median_nni": None,
    "_hrv_ratio_sd2_sd1": None,
    "150_hrv_median_nni": 150,
    "_hrv_vlf": None,
    "_hrv_lf": None,
    "_hrv_hf": None,
    "_hrv_lf_hf_ratio": None,
    "_hrv_total_power": None,
}

#: 每个 RRV 槽位对应的窗口长度（秒）。
_RRV_SLOT_WINDOW = {
    "150_RRV_MedianBB": 150,
    "150_RRV_LF": 150,
    "270_RRV_MCVBB": 270,
    "150_RRV_CVBB": 150,
}

#: 一个 epoch 内至少多少 RR 间期才算 HRV 可用（与 MESA 路径一致）。
MIN_RR_PER_EPOCH = 10

#: 相邻两次 `update` 的 `latest_ts_ms` 必须相差这么多毫秒（= 一个 epoch）。
#: 见 `update` 里的步进校验。
_WINDOW_STEP_MS = int(G.EPOCH_S * 1000)

#: 从槽位名反查特征键用的后缀表。**长的排前面**，避免 `lf` 抢先匹配 `lf_hf_ratio`。
_HRV_SLOT_SUFFIX = ["ratio_sd2_sd1", "median_nni", "lf_hf_ratio",
                    "total_power", "vlf", "lf", "hf"]


@dataclass
class WindowResult:
    """一次 `update` 的产出。

    Attributes
    ----------
    features : np.ndarray, shape (21, 13)
        模型窗 21 个 epoch 的特征, 列序 = `SLOT_NAMES`。
        ⚠️ 可能含 NaN —— 某个 epoch 的拍数不足 `MIN_RR_PER_EPOCH` 时, 它那 7 个
        逐 epoch 的 HRV 槽位为 NaN（离线路径是**整行剔除**, 流式不能剔除,
        由调用方决定填充策略; 与窗口化路径一致的 0 填充是一种合理选择）。
    result_start_ts_ms, result_end_ts_ms : int
        产出 epoch 的 unix 毫秒时间戳。**按实际算出的时刻返回**, 不是接口头文件里
        那个 `latest − 300000` —— 差额来源见 `geometry.py`。
    epoch_starts_s : np.ndarray, shape (21,)
        每个 epoch 起点相对窗口起点的秒数（268.45 … 868.45）。
    """

    features: np.ndarray
    result_start_ts_ms: int
    result_end_ts_ms: int
    epoch_starts_s: np.ndarray


class StreamingPreprocessor:
    """有状态的流式预处理器（单实例, 不建线程、不加锁）。

    调用方保证 `update` 串行调用 —— 与 C ABI 的约定一致。

    Parameters
    ----------
    fill_nan : float or None, default None
        拍数不足的 epoch, 其逐 epoch HRV 槽位怎么填。`None` = 保留 NaN。

    Examples
    --------
    >>> pre = StreamingPreprocessor()
    >>> # 每 30 s 推一次 900 s 窗口
    >>> res = pre.update(window_18000_float, 18000, latest_ts_ms)
    >>> res.features.shape
    (21, 13)
    """

    def __init__(self, fill_nan: Optional[float] = None):
        self.fill_nan = fill_nan
        self.reset()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空跨调用状态（对应 C++ 侧 `init` 时的上下文清零）。

        ⚠️ 清空后 ACT 的 expanding 锚点、以及心搏检测的滚动阈值锚点都回到"此刻"。
           这正是项目当前接受的「进程重启偏差」来源 —— 详见 `movement.py` 模块文档。
        """
        self._act_state = M.ActState()
        self._last_latest_ts_ms = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def update(self, samples: np.ndarray, sample_count: int, latest_ts_ms: int) -> WindowResult:
        """处理一个 900 s 窗口。

        Parameters
        ----------
        samples : array-like
            窗口样本, 长度须为 `G.WINDOW_SAMPLES`(18000)。**调用期间不得被并发修改**
            —— 与 C ABI 的约定一致（库不拷贝输入）。本实现会 `asarray`（不复制）
            并只读访问。
        sample_count : int
            必须等于 18000, 否则 `ValueError`（对应 `SS_RT_ERR_INVALID_ARG`）。
        latest_ts_ms : int
            窗口**最后一个样本**的 unix 毫秒时间戳。

        Returns
        -------
        WindowResult

        Raises
        ------
        ValueError
            长度不符、或输入含 NaN。
        """
        arr = np.asarray(samples, dtype=float).ravel()
        if sample_count != G.WINDOW_SAMPLES or len(arr) != G.WINDOW_SAMPLES:
            raise ValueError(
                f"sample_count 必须等于 {G.WINDOW_SAMPLES}, "
                f"收到 sample_count={sample_count}、实际长度={len(arr)}")

        # ⚠️ 窗口步进必须恰好 30 s。ACT 的跨调用状态按"每次推进一个 epoch"维护
        #    （`_act_features` 里用 `window[:EPOCH_SAMPLES]` 推到下一窗口起点）。
        #    漏推 / 重推 / 变步进会让 expanding 锚点与真实会话起点脱钩 ——
        #    实测锚点前移会把入睡后的非零 epoch 从 280 个打到 1 个（−99.6%），
        #    而且**不会报任何错**, 只会静默产出错的体动特征。
        #    与 C ABI 的对应: 该异常映射到 `SS_RT_ERR_INVALID_ARG`。
        if self._last_latest_ts_ms is not None:
            delta = int(latest_ts_ms) - self._last_latest_ts_ms
            if delta != _WINDOW_STEP_MS:
                raise ValueError(
                    f"窗口步进异常: 上次 latest_ts_ms={self._last_latest_ts_ms}, "
                    f"本次 {latest_ts_ms}, 相差 {delta} ms（期望 {_WINDOW_STEP_MS} ms）。"
                    f"ACT 的 expanding 锚点按'每次推进一个 30 s epoch'维护, "
                    f"步进不符会让锚点与真实会话起点脱钩, 且不会报错、只会静默产出"
                    f"错的体动特征。若这确实是新会话或有意重放, 请先调用 reset()。")
        self._last_latest_ts_ms = int(latest_ts_ms)

        epochs = G.model_epoch_ks()                     # k = 9..29
        n_feat = len(SLOT_NAMES)
        feats = np.full((len(epochs), n_feat), np.nan, dtype=float)
        starts_s = np.array([G.epoch_start_s(k) for k in epochs], dtype=float)

        # ---- 1. ACT（唯一有状态的支路）----
        act = self._act_features(arr, epochs)

        # ---- 2. 心搏检测（HRV 用）----
        peak_times, rr_all = self._beat_features(arr)

        # ---- 3. 呼吸变异（RRV）----
        rrv_cache = {}

        # ---- 4. 逐 slot 装配 ----
        for ei, k in enumerate(epochs):
            e0 = G.epoch_start_s(k)
            e1 = e0 + G.EPOCH_S
            feats[ei, 0] = act[ei]

            for si, slot in enumerate(SLOT_NAMES[1:], start=1):
                if slot in _HRV_SLOT_WINDOW:
                    win = _HRV_SLOT_WINDOW[slot]
                    t0 = e0 if win is None else e1 - win
                    vals = _rr_in(peak_times, rr_all, t0, e1)
                    feats[ei, si] = _hrv_slot(slot, vals, win is None)
                elif slot in _RRV_SLOT_WINDOW:
                    win = _RRV_SLOT_WINDOW[slot]
                    a = int(round((e1 - win) * G.FS))
                    b = int(round(e1 * G.FS))
                    key = (a, b)
                    if key not in rrv_cache:
                        # ⚠️ 必须兜住 —— 呼吸峰检测在退化输入上会抛（探头掉线、
                        #    饱和、直流漂移都会给出常量段）:
                        #      常量段            → IndexError（无完整过零点）
                        #      后半段恒定/无峰    → ValueError（bbi 为空）
                        #    离线路径 iruwb/rrv.py 每个窗口都 catch 并填 0;
                        #    HRV 支路（`_hrv_slot`）也有 try/except。这里漏掉的话,
                        #    一次 300 s 的信号异常会中断**整次推理**而不是退化成 NaN。
                        try:
                            rrv_cache[key] = rrv_features(arr[a:b], G.FS)
                        except (ValueError, IndexError):
                            rrv_cache[key] = None        # 该窗口退化, 整组槽位判无效
                    cached = rrv_cache[key]
                    feats[ei, si] = (np.nan if cached is None
                                     else cached[slot.split("_RRV_")[1]])

        if self.fill_nan is not None:
            feats = np.nan_to_num(feats, nan=float(self.fill_nan))

        start_ms, end_ms = G.result_tsms(latest_ts_ms)
        return WindowResult(features=feats, result_start_ts_ms=start_ms,
                            result_end_ts_ms=end_ms, epoch_starts_s=starts_s)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _act_features(self, window: np.ndarray, epochs: List[int]) -> np.ndarray:
        """ACT 的 21 个 epoch 均值, 并推进跨调用状态。

        见模块文档"每帧 ACT 的处理"—— 一次重放拿特征, 一次短重放推进状态。
        """
        # 1) 从窗口起点状态重放整窗（用副本, 不改真状态）
        st = self._act_state.copy()
        thresh = st.process(window)

        # 2) 推进到**下一个**窗口的起点 = 本窗口的第 30 s
        st_next = self._act_state.copy()
        st_next.process(window[:G.EPOCH_SAMPLES])
        self._act_state = st_next

        # 3) 按 epoch 网格取均值
        starts = np.array([G.epoch_start_sample(k) for k in epochs])
        ends = starts + G.EPOCH_SAMPLES
        return M.epoch_means(thresh, starts, ends)

    def _beat_features(self, window: np.ndarray) -> tuple:
        """心搏检测, 返回窗口内的 `(peak_times_s, rr_s)`（相对窗口起点）。

        ⚠️ **已知的跨调用不一致（未修, 见下）**: `beats.rolling_quantile_causal`
           在每个数组的前 `CAUSAL_WINDOW_S`(300 秒) 用 `expanding`（锚点 = **数组起点**）,
           之后才切尾部滚动。离线路径跑整夜时这个展开区只在夜间最开头; 流式每帧传的
           是新窗口, 于是**阈值锚点随窗口滑动**, 同一个 epoch 在不同窗口里会得到
           略微不同的特征。

           实测影响（连续 4 个窗口, 20 个重叠 epoch）:

           =====================  ============
           槽位                    跨调用最大相对差
           =====================  ============
           `_acc_mean_1`          0（逐位一致）
           RRV 四个槽位            0（逐位一致）
           HRV 逐 epoch 七个        ≤ 2.5e-12（浮点噪声）
           **`150_hrv_median_nni`** **8.2e-4**
           =====================  ============

           只有 150 s 窗口那一个槽位有实际差异（它的窗口回溯得更远, 更容易碰到
           展开区里被移动的阈值）。**要彻底修需要跨调用累计 ≥1200 s 原始缓冲**
           （窗口前 300 s 不在上一窗口内 —— 相邻窗口只差 30 s, 那 300 s 在上一窗口
           起点之前 270 s）, 代价是 ~192 KB 状态与 +33% 滤波耗时。已记入待办。

           当前实现与离线路径的差异仅限于"窗口前 300 s 的阈值", 30 s 的逐 epoch
           窗口不受影响。
        """
        beat = detect_beats(window, G.FS)
        return beat["peak_times_s"], beat["peaks"][:, 1]


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _rr_in(peak_times: np.ndarray, rr_all: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """取落在 `[t0, t1)` 内的拍的 RR 间期（毫秒）。

    ⚠️ RR 归属约定与 MESA 一致: 每个间期挂在它的**后一拍**上（`rr_all[i] = t[i] − t[i−1]`）,
       所以这里按拍时刻筛选即可, 不需要额外偏移。
    """
    if len(peak_times) == 0:
        return np.array([])
    m = (peak_times >= t0) & (peak_times < t1)
    return rr_all[m] * 1000.0                       # 秒 → 毫秒（hrvanalysis 的约定）


def _hrv_slot(slot: str, rr_ms: np.ndarray, per_epoch: bool) -> float:
    """算单个 HRV 槽位。

    ⚠️ 逐 epoch 槽位在拍数不足时返回 NaN（不是 0）—— 填 0 会让
       `_hrv_median_nni = 0 ms` 这种物理上不可能的值混进特征表。
    """
    if per_epoch and len(rr_ms) < MIN_RR_PER_EPOCH:
        return np.nan
    if len(rr_ms) < 2:
        return np.nan
    try:
        f = hrv_features(rr_ms)
    except Exception:
        # 退化输入（拍数过少 / RR 序列异常）—— 与离线路径 `try/except` 的语义一致
        return np.nan
    for suffix in _HRV_SLOT_SUFFIX:
        if slot.endswith(suffix):
            return float(f[suffix])
    raise KeyError(f"未知 HRV 槽位 {slot}")
