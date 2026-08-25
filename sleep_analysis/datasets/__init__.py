"""数据集包 — import 本包即触发各数据集模块的 @register 注册 (自注册目录)。

加新数据集: 新建模块后在此加一行 import (保持显式, 便于发现与审计)。
"""

from . import mesadataset  # noqa: F401  (注册 "MESA_Sleep")
from . import shhs_dataset  # noqa: F401  (注册 "SHHS1" / "SHHS2")
