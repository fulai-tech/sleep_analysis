"""ECG R 点 → RR 间期 → HR 的清洗入口。

process_rpoint 已抽到 preprocessing/rr_utils.py（MESA / SHHS 共用），
这里仅做转发以保持既有调用方（preprocess_mesa.py / inference_full.py）兼容。

2026-08-05 决策: HRV 相关处理回退原版（hrvanalysis 非因果流程, 与原版已训练模型
保持可比性; causal 实现注释保留在 rr_utils.py, 见其顶部决策说明）。
"""

from sleep_analysis.preprocessing.rr_utils import process_rpoint

__all__ = ["process_rpoint"]
