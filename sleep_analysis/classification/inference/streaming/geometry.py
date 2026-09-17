"""流式推理的窗口与 epoch 网格几何。

**本文件是自包含的常量与换算层, 不 import 任何其他模块, 也不依赖 numpy。**
C++ 侧对应一个纯头文件 (常量 + 几个 inline 函数)。

与 SDK 接口的对应关系
---------------------
`SleepStagingInterface/include/sleep_staging_realtime.h` 定义:

  - 每次 `process` 传入**完整 900 s 窗口**: 18000 个 float (20 Hz、50 ms 均匀采样)
  - `latest_ts_ms` = 窗口内**最后一个样本**的 unix 毫秒时间戳
  - 输出**唯一一个**分期结果, 对应窗口内"倒数第 5 分钟"的 30 s 时段

**epoch 网格故意不落在整 30 s 上, 而是偏移 −1.55 s**:

    epoch k 的区间 = [k×30 − 1.55, k×30 − 1.55 + 30)   秒 (相对窗口起点)

偏移量 1.55 s = FIR 群延迟 1.5 s (61 tap @20 Hz) + 峰确认 1 个样本 0.05 s。
这样最后一个完整 epoch 的末尾落在 898.45 s, 而窗口有数据到 900 s,
**最后 1.55 s 正好够把该 epoch 内的心搏全部确认出来** —— 否则该 epoch 的
HRV 会因末段心搏未确认而缺拍 (实测 `_hrv_vlf` 可偏 600%)。

    k=1  [  28.45,   58.45)      ← 首个完整 epoch
    ...
    k=9  [ 268.45,  298.45)      ┐
    ...                          ├ 模型窗 (21 个 epoch, 居中)
    k=29 [ 868.45,  898.45)      ┘   ← 产出的分期 = 窗内第 10 个 (k=19)
    k=30 [ 898.45,  928.45)      ✗ 需要数据到 928.45, 窗口只有 900

**实际用到的数据区间 = [28.45, 900] = 871.55 s** —— 即理论最小值
(`(seq_len−1)×30 + 270 = 870` 再加末端 1.55 s)。

k 的完整范围是 k=1..29 (共 29 个 epoch); 模型窗取其中 k=9..29。

⚠️ 1.55 s 是**临界值**: 需求正好是 1.5 s 群延迟 + 0.05 s 峰确认。
   若日后改 `n_taps`, 这个偏移量必须跟着重算, 否则末 epoch 会缺拍。
"""

# ---------------------------------------------------------------------------
# 采样与窗口
# ---------------------------------------------------------------------------

FS = 20.0                    # 采样率 (Hz) —— SDK 接口约定, 不支持其他值
WINDOW_S = 900.0             # 分析窗口 15 min
WINDOW_SAMPLES = 18000       # = FS * WINDOW_S

EPOCH_S = 30.0               # 一个 epoch 30 s
EPOCH_SAMPLES = 600          # = FS * EPOCH_S

# ---------------------------------------------------------------------------
# epoch 网格
# ---------------------------------------------------------------------------

#: epoch 起点相对整 30 s 的偏移 (秒)。见模块文档 —— 这是末端 HRV 完整性的保证。
GRID_SHIFT_S = 1.55

#: 完整 epoch 的 k 范围 (1-based)。k=0 起点为负、k>=30 越出窗口, 都不完整。
FIRST_COMPLETE_K = 1
LAST_COMPLETE_K = 29

# ---------------------------------------------------------------------------
# 模型窗
# ---------------------------------------------------------------------------

#: 模型序列长度 (epoch 数), 与训练 config.json 的 `seq_len` 一致。
MODEL_SEQ_LEN = 21

#: 模型窗覆盖的 k 范围: 末尾对齐 LAST_COMPLETE_K, 向前 21 个。
MODEL_LAST_K = LAST_COMPLETE_K                                        # 29
MODEL_FIRST_K = MODEL_LAST_K - (MODEL_SEQ_LEN - 1)                    # 9

#: 产出 epoch 在模型窗内的下标 (0-based)。居中窗 → 中心位置。
RESULT_INDEX_IN_MODEL = MODEL_SEQ_LEN // 2                            # 10

#: 产出 epoch 的 k。
RESULT_K = MODEL_FIRST_K + RESULT_INDEX_IN_MODEL                      # 19

# ---------------------------------------------------------------------------
# 结果时间戳 (相对 latest_ts_ms 的偏移, 毫秒)
# ---------------------------------------------------------------------------
#
# 产出 epoch 相对窗口起点的区间 = [RESULT_K*30 - GRID_SHIFT_S, +30)
#                              = [568.45, 598.45)
# 而 latest 对应窗口起点 + 900 s, 所以:
#
#   结果起点 = latest − (900 − 568.45) = latest − 331.55 s
#   结果终点 = latest − (900 − 598.45) = latest − 301.55 s
#
# ⚠️ 接口头文件写的是 `start_ts = latest − 300000` (5 min)。本模块按**实际**
#    算出的时刻返回, 即 latest − 331550 —— 差额 31.55 s 的来源见项目说明:
#    30 s 是接口自身的问题 (居中窗要求窗尾落到 latest+30, 真实下限是结果终点
#    latest−300), 1.55 s 是我们的网格偏移。SDK 侧请**使用本模块返回的时间戳**,
#    不要按接口公式硬套。

RESULT_START_OFFSET_MS = 331550
RESULT_END_OFFSET_MS = 301550


# ---------------------------------------------------------------------------
# 换算函数
# ---------------------------------------------------------------------------


def epoch_start_s(k: int) -> float:
    """第 k 个 epoch 的起点, 相对窗口起点的秒数。"""
    return k * EPOCH_S - GRID_SHIFT_S


def epoch_start_sample(k: int) -> int:
    """第 k 个 epoch 的起点样本下标 (四舍五入)。

    ⚠️ `GRID_SHIFT_S=1.55` 在 20 Hz 下 = 31 个样本, 是整数, 所以这里没有
       舍入误差。若日后改偏移量, 必须保证 `shift * FS` 是整数, 否则 epoch
       边界会落在样本之间, C++ 侧的对齐会和 Python 侧不一致。
    """
    return int(round((k * EPOCH_S - GRID_SHIFT_S) * FS))


def model_epoch_ks() -> list:
    """模型窗覆盖的 k 列表 (21 个, 升序)。"""
    return list(range(MODEL_FIRST_K, MODEL_LAST_K + 1))


def result_tsms(latest_ts_ms: int) -> tuple:
    """由 `latest_ts_ms` 算出产出 epoch 的 `(start_ts_ms, end_ts_ms)`。"""
    return (latest_ts_ms - RESULT_START_OFFSET_MS,
            latest_ts_ms - RESULT_END_OFFSET_MS)


# ---------------------------------------------------------------------------
# 自检: 确保常量之间的推导关系没被改坏
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """导入时自检 —— 几何常量改错会静默产出错位结果, 不如直接炸。"""
    # 偏移量必须是整数个样本
    shift_samples = GRID_SHIFT_S * FS
    if abs(shift_samples - round(shift_samples)) > 1e-9:
        raise AssertionError(
            f"GRID_SHIFT_S={GRID_SHIFT_S} 在 fs={FS} 下不是整数个样本 "
            f"({shift_samples}) —— epoch 边界会落在样本之间, C++/Python 对齐会不一致")

    # 最后一个完整 epoch 的末端必须给 HRV 留下 1.55 s 余量
    last_end = epoch_start_s(LAST_COMPLETE_K) + EPOCH_S
    margin = WINDOW_S - last_end
    if abs(margin - GRID_SHIFT_S) > 1e-9:
        raise AssertionError(
            f"末 epoch 末端 {last_end:.2f}s 距窗口末端只剩 {margin:.2f}s, "
            f"而 HRV 需要 {GRID_SHIFT_S}s —— 会缺拍")

    # 模型窗末端必须正好是最后一个完整 epoch
    if MODEL_FIRST_K + MODEL_SEQ_LEN - 1 != MODEL_LAST_K:
        raise AssertionError("模型窗的 k 范围与 MODEL_SEQ_LEN 不自洽")

    # 首个模型窗 epoch 的 270 s RRV 窗必须落在窗口内
    rrv_start = epoch_start_s(MODEL_FIRST_K) + EPOCH_S - 270.0
    if rrv_start < 0:
        raise AssertionError(
            f"首个模型窗 epoch 的 RRV 270s 窗起点 {rrv_start:.2f}s 为负 —— 需要更长的窗口")


_self_check()
