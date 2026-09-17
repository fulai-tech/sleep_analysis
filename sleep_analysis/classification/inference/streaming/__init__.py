"""流式睡眠分期预处理（推理侧参考实现, 对齐 C ABI 接口）。

面向端侧（arm32 + NPU）部署的**预处理链**参考实现 —— 不是训练/研究工具。
目标是让 `SleepStagingInterface` 的 C++ 算法模块能逐函数对照翻译:
**每个模块对应 C++ 侧一个 .cpp/.h**。

用法::

    from sleep_analysis.classification.inference.streaming import StreamingPreprocessor

    pre = StreamingPreprocessor()
    res = pre.update(window_18000_float, 18000, latest_ts_ms)
    res.features          # (21, 13)
    res.result_start_ts_ms, res.result_end_ts_ms

模块划分（与 C++ 文件的对应关系）
--------------------------------
=================  ==========================================================
`geometry.py`      窗口与 epoch 网格常量、时间戳换算（纯常量, 无 numpy）
`filters.py`       硬编码滤波系数 + `lfilter` + 滑动平均原语
`movement.py`      体动支路 + `ActState`（**唯一有跨调用状态的支路**）
`beats.py`         心搏检测（`find_peaks` / 滚动分位 / median-MAD 的内联实现）
`hrv.py`           HRV 特征（内联 hrvanalysis 子集）
`resp.py`          呼吸峰检测 + RRV 特征（内联 neurokit2 子集）
`prep.py`          13 槽位编排 + `StreamingPreprocessor`
=================  ==========================================================

⚠️ 与离线路径（`preprocessing/iruwb/pipeline.py`）的关系
------------------------------------------------------
相同之处: 三条支路的算法逐函数对应, 窗口口径一致。
不同之处: 本模块**有跨调用状态**（ACT 的 expanding 统计量）——
离线路径跑整段时锚点是录制起点, 本模块靠状态携带保持同一个锚点。
**只要从会话起点开始喂, 两者的特征应当一致**（实测差异 ≤1e-13 相对, 且稀疏结构一致）。

⚠️ 已知局限（**本模块不解决**, 只是忠实复现现状）
----------------------------------------------
1. **ACT 的睡前活动陷阱**：分母是"当晚最大值", 受试者入睡前活动会把特征压废。
   详见 `movement.py` 模块文档。
2. **HRV 的扩模态口径**：槽位 3 (`150_hrv_median_nni`) 用 150 s 窗口, 其余 7 个逐 epoch
   —— 与 MESA 预训练口径不完全一致, 见 `prep.SLOT_NAMES` 的说明。
3. **拍数不足的 epoch 返回 NaN**：离线路径是整行剔除, 流式不能剔除, 由调用方决定
   填充策略（`StreamingPreprocessor(fill_nan=0.0)` 可自动填）。
"""

from .geometry import (
    EPOCH_S, FS, MODEL_SEQ_LEN, WINDOW_S, WINDOW_SAMPLES,
)
from .prep import SLOT_NAMES, StreamingPreprocessor, WindowResult

__all__ = [
    "StreamingPreprocessor", "WindowResult", "SLOT_NAMES",
    "FS", "WINDOW_S", "WINDOW_SAMPLES", "EPOCH_S", "MODEL_SEQ_LEN",
]
