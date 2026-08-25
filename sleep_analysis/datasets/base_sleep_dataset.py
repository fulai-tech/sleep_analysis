"""BaseSleepDataset — 数据集统一骨架 (模板方法) + 自描述策略钩子。

设计目标 (20260822): 新增数据集时**不改任何现有代码** — 新建一个模块,
继承 BaseSleepDataset, 实现下面的钩子, @register 注册, 在 datasets/__init__.py
里加一行 import 即可。训练管线只面向这些公共钩子编程, 不认识具体数据集。

子类需要覆盖的钩子:
  - ``id_pattern``:   create_index 从 features_combined 文件名提取 subj_id 的正则 (带捕获组)
  - ``processed_path``: 处理后的数据目录 (含 features_full_combined 子目录)
  - ``has_actigraphy``: 是否有体动数据 (驱动默认模态选择与 ACT 剔除)
  - ``person_pool``:   共享被试身份的数据集组名 — 同组数据集做联合人员级划分
                      (如 SHHS1/SHHS2 都叫 "shhs"; 未来 MROS1/MROS2 都叫 "mros")
  - ``modality_defaults``: 该数据集的默认模态列表
  - ``FEATURE_COLUMNS``: 模态 → 特征列名 (默认覆盖 MESA/SHHS/MrOS 命名约定;
                      命名不同的数据集如 D04 覆盖此项)
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tpcp import Dataset


class BaseSleepDataset(Dataset):
    # ------------------------------------------------------------------
    # 自描述钩子 (子类覆盖)
    # ------------------------------------------------------------------
    id_pattern: str = ""   # 如 r"(\d{4})\.csv" (MESA) / r"(\d{6})\.csv" (SHHS)
    has_actigraphy: bool = True
    person_pool: str = ""  # 空 = 独立池 (不参与联合划分)
    modality_defaults: list = ["ACT", "HRV", "RRV"]
    processed_path: Optional[Path] = None  # 处理后数据目录 (含 features_full_combined; 子类 __init__ 设置)

    # 模态 → 特征列名 (与训练侧 data_peparation 的历史选择完全一致,
    # 含 _hrv_median_nni 的重复项 — 保持输入维度与 get_num_input 的 HRV=8 对齐)
    FEATURE_COLUMNS = {
        "ACT": ["_acc_mean_1"],
        "HRV": ["_hrv_median_nni", "_hrv_ratio_sd2_sd1", "_hrv_median_nni",
                "_hrv_vlf", "_hrv_lf", "_hrv_hf", "_hrv_lf_hf_ratio", "_hrv_total_power"],
        "RRV": ["150_RRV_MedianBB", "150_RRV_LF", "270_RRV_MCVBB", "150_RRV_CVBB"],
        "EDR": ["150_EDR_MeanBB", "150_EDR_LF", "150_EDR_HF", "150_EDR_LFHF"],
    }

    @classmethod
    def feature_columns(cls, modality: str):
        """该数据集在指定模态下选用的特征列名 (策略)。"""
        return cls.FEATURE_COLUMNS.get(modality)

    # ------------------------------------------------------------------
    # 模板方法: 各数据集共用的索引构建 (从 features_full_combined 文件名提取 ID)
    # ------------------------------------------------------------------
    def create_index(self):
        path = self.processed_path.joinpath("features_full_combined").resolve()
        path_list = list(Path(path).glob("*.csv"))
        # 只从文件名提取 ID, 避免路径中的数字串 (如 processed_data_20260805 的
        # 2026/0805) 被当成伪被试 ID (20260805 修复)
        subj_id = [re.findall(self.id_pattern, f.name)[0] for f in path_list]
        return pd.DataFrame(subj_id, columns=["subj_id"])
