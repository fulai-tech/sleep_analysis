"""ECG R 点 → RR 间期 → HR 的清洗入口。

process_rpoint 已抽到 preprocessing/rr_utils.py（MESA / SHHS 共用, 含 causal 分支），
这里仅做转发以保持既有调用方（preprocess_mesa.py / inference_full.py）兼容。

20260804 - rdwang: 原实现的非因果差值（双向插值 + 整夜均值 + malik 前瞻一拍）
已由 rr_utils.process_rpoint 的 causal 分支替代（processing_config.causal=True 时生效）。
"""

from sleep_analysis.preprocessing.rr_utils import process_rpoint

__all__ = ["process_rpoint"]
