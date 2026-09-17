# streaming —— 流式睡眠分期预处理（推理侧参考实现）

面向端侧（arm32 + NPU）部署的**预处理链**参考实现。不是训练/研究工具。

**用途**：让 `SleepStagingInterface` 的 C++ 算法模块能逐函数对照翻译。每个 `.py`
对应 C++ 侧一个 `.cpp/.h`。

## 与 SDK 接口的对应

C ABI 定义在 `/home/rdwang/repo/SleepStagingInterface/include/sleep_staging_realtime.h`。

| 本模块 | C ABI |
|---|---|
| `StreamingPreprocessor.update(samples, sample_count, latest_ts_ms)` | `ss_rt_process` |
| `StreamingPreprocessor.reset()` | `ss_rt_init` 时清空 `RtContext` |
| `WindowResult.features` (21×13) | 喂给 ONNX/NPU 的输入缓冲 |
| `WindowResult.result_start_ts_ms / result_end_ts_ms` | `ss_rt_result.start_ts_ms / end_ts_ms` |
| `ActState` | `RtContext` 里的跨调用状态 |

## 用法

```python
from sleep_analysis.classification.inference.streaming import StreamingPreprocessor

pre = StreamingPreprocessor()          # 单实例；调用方保证串行调用
# SDK 攒满 900 s 后每 30 s 推一次
res = pre.update(window_18000_float, 18000, latest_ts_ms)
res.features                           # (21, 13)，列序 = SLOT_NAMES
res.result_start_ts_ms                 # = latest − 331550
```

## 几何

```
900 s 窗口 ─────────────────────────────────────────────────────┤
0                   28.45        268.45              898.45   900
                    │            ├──── 模型窗 21 epoch ────┤
                    └ 首个 RRV 270s 窗起点

epoch 网格: k×30 − 1.55 秒（偏移 1.55 s = FIR 群延迟 1.5 s + 峰确认 1 样本）
产出 epoch = 窗内第 10 个 = [latest−331.55s, latest−301.55s)
```

`geometry.py` 在导入时自检这些推导关系，改坏常量会直接抛异常。

⚠️ **时间戳**: 本模块按**实际算出的时刻**返回（`latest − 331550`），不是接口头文件里
写的 `latest − 300000`。差额 31.55 s 的来源见项目说明：30 s 是接口自身的 spec 问题
（居中窗要求窗尾落到 `latest+30`），1.55 s 是我们的网格偏移。**SDK 侧请用本模块返回的
时间戳，不要按接口公式硬套。**

## 模块划分

| 文件 | 职责 | 内联了什么 |
|---|---|---|
| `geometry.py` | 窗口/epoch 网格常量与换算 | 纯常量，无依赖 |
| `filters.py` | 滤波系数与 `lfilter` | `scipy.signal` 的 butter/firwin/lfilter/lfilter_zi |
| `movement.py` | 体动支路 + `ActState` | pandas 的 `rolling` / `expanding` |
| `beats.py` | 心搏检测 | `scipy.signal.find_peaks` + pandas 滚动分位/中位数 |
| `hrv.py` | HRV 特征 | `hrvanalysis` 的 7 个特征 |
| `resp.py` | 呼吸峰检测 + RRV 特征 | `neurokit2` 的 biosppy 路径 + 4 个 RRV 特征 |
| `prep.py` | 13 槽位编排 + `StreamingPreprocessor` | — |

**滤波系数是硬编码常量**（不是运行时设计），Python 与 C++ 共用同一组数值。
生成命令写在 `filters.py` 顶部。

## 精确度（实测）

| 环节 | 对拍对象 | 结果 |
|---|---|---|
| `filters.lfilter` / `apply_iir_causal` | `scipy.signal.lfilter` | **逐位一致** |
| `filters` 的 FIR / 滑动平均 | `scipy` / `pandas` | ~1e-16（1 ULP） |
| `hrv.py` 的 `median_nni` / `ratio_sd2_sd1` | `hrvanalysis` | **逐位一致** |
| `hrv.py` 频域 5 个 | `hrvanalysis` | ≤1 ULP |
| `resp.py` 峰检测 + `MedianBB`/`CVBB`/`MCVBB` | `neurokit2` | **逐位一致** |
| `resp.py` 的 `LF`/`VLF`/`HF` | `neurokit2` | ≤1 ULP |
| `beats.find_peaks`（4 种配置） | `scipy.signal.find_peaks` | **逐位一致** |
| `beats.detect_beats` 端到端 | `beat_detection.detect_beats` | 峰数一致，峰位 ≤1.1e-13 |
| `movement` 的阈值化序列 | `extract_movement` | 稀疏结构一致，值差 ≤5.8e-15 |

**残余的 1 ULP 都来自 FFT 与浮点累加顺序**（`np.fft.rfft` vs scipy 的 FFT；numpy/pandas
内部的 SIMD 分块累加）。**不值得追** —— C++ 侧用自己的 FFT 库和累加循环，同样是这个量级。
对拍时用 `rtol=1e-12`；上表标"逐位一致"的项可用严格相等。

## 跨调用状态

**只有 ACT 支路需要状态。** 原因：`_scale` / `_normalize` 用 `expanding()`，锚点是
传入数组的第 0 个元素。若每帧独立跑 900 s 窗口，锚点变成窗口起点 ——
**实测入睡后非零 epoch 从 280 个掉到 1 个（−99.6%）**。

`ActState` 约 200 float + 7 标量，可直接映射成 C struct（见 `movement.py` 的 docstring）。

每帧的推进方式（避免把重叠的 870 s 重复累加）：
1. 拿窗口起点状态的**副本**重放整窗 → 得到整窗的阈值化导数 → 算 21 个 epoch 均值
2. 另取副本只喂前 30 s → 存回，作为下一次的窗口起点状态

HRV / RRV 无状态（窗口有界 ≤270 s，每帧重算）。

## 调用契约（C++ 侧必须照做）

| 契约 | 不遵守的后果 | 本模块的处理 |
|---|---|---|
| 每次 `latest_ts_ms` 比上次**恰好 +30000 ms** | ACT 锚点与真实会话起点脱钩，**静默**产出错的体动特征（实测非零 epoch 从 280 → 1） | 抛 `ValueError`（映射 `SS_RT_ERR_INVALID_ARG`）。确实要开新会话请显式 `reset()` |
| 输入窗口长度恒为 18000 | — | 抛 `ValueError`（映射 `SS_RT_ERR_INVALID_ARG`） |
| 调用期间输入内存不得被并发写 | 未定义行为 | 库不拷贝，只读访问 |
| `init`/`update`/`reset` 串行调用 | 未定义行为 | 内部不加锁 |

**退化输入（探头掉线 / 饱和 / 直流漂移产生的常量段）不会中断推理**：
RRV 与 HRV 支路各自 `try/except`，受影响 epoch 的对应槽位写 **NaN**
（`resp.py` 的峰检测在常量段上会抛 `IndexError`/`ValueError`）。
调用方可用 `StreamingPreprocessor(fill_nan=0.0)` 自动填 0，或自行处理 NaN。

⚠️ `_acc_mean_1` 在常量段**不会**变 NaN（体动链对该输入不退化），这是刻意的 ——
它反映的是"没有体动"而非"信号无效"。要区分"无效"与"无体动"请结合 HRV/RRV 的 NaN 判断。

## 已知局限

1. **`150_hrv_median_nni` 的跨调用不一致（8.2e-4 相对）**
   `beats.rolling_quantile_causal` 在每个数组的前 300 s 用 `expanding`（锚点 = 数组起点），
   而流式每帧传新窗口 → 阈值锚点随窗口滑动。只有这**一个槽位**受影响，
   其余 12 个要么逐位一致、要么 ≤2.5e-12。
   彻底修需要跨调用累计 ≥1200 s 原始缓冲（+192 KB 状态、+33% 滤波耗时）。

2. **ACT 的睡前活动陷阱**（**本模块不解决**，忠实复现现状）
   分母是"当晚最大值"，受试者入睡前在床边活动会把整夜特征压废
   （实测注入 10 min 活动即让非零 epoch 掉 99.6%）。这是从 D04 继承的算法设计缺陷，
   与因果性/流式无关，离线跑同样存在。详见 `movement.py` 模块文档。

3. **拍数不足的 epoch 返回 NaN**
   离线路径是整行剔除，流式不能剔除。调用方决定填充策略
   （`StreamingPreprocessor(fill_nan=0.0)` 可自动填 0，与窗口化路径的约定一致）。

4. **HRV 口径未完全对齐 MESA**
   槽位 3 (`150_hrv_median_nni`) 用 150 s 窗口，其余 7 个逐 epoch；MESA 的 8 个
   全是逐 epoch（第 3 个是重复列）。这是本次按用户圈定采用的口径，
   迁移学习时的语义影响待评估。

5. **所有参数都在合成信号上调过**，未在真实 IR-UWB 数据上验证。

## 性能

0.72 s / 900 s 窗口（单核，Python）。SDK 每 30 s 调一次 → **2.4% 占用**。
C++ 侧会快一到两个数量级。

最耗时的两处（C++ 优化重点）：
- `apply_filter_chain` 的 FIR（61 tap，18000 点）
- `rolling_quantile_causal`（300 s 窗，逐点 `np.partition`）

## 对拍

每个模块的 docstring 里都有对拍结果表与踩过的坑。
改任何一处前请重跑对应模块的对拍（判据见"精确度"表）。
