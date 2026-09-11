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

import os
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
    split_file: Optional[str] = None       # 本数据集自己的划分文件 (项目内相对路径, 如
                                           # "splits/split_mesa_1121_20260806.json"); 由 study_data.json
                                           # 的 "<数据集>_split_file" 键配置, 换划分版本只改配置不动代码;
                                           # 训练时按请求的数据集分别加载再组合; None = 用随机划分

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
        files = list(Path(path).glob("*.csv"))
        # ✅2026-08-27: 确定性顺序 — Path.glob 不排序, 被试顺序 = 文件系统目录序,
        # 跨机器不一致 (目录序不同 → batch 组成不同 → 同参数训练结果分叉)。
        # 默认 sorted 保证跨机器确定性; 设置 SLEEP_ORDER_MANIFEST=<每行一个被试ID的文件>
        # 则按清单顺序排列 — 精确复现历史 run (清单 = 该 run 所在机器的目录序,
        # 如 splits/order_mesa_212_20260817.txt)。
        manifest = os.environ.get("SLEEP_ORDER_MANIFEST")
        if manifest:
            want = [l.strip() for l in open(manifest) if l.strip()]
            by_id = {re.findall(self.id_pattern, f.name)[0]: f for f in files}
            matched = [s for s in want if s in by_id]
            if not matched:
                # 清单与本数据集 ID 格式完全不匹配 (如 MESA 4 位 ID 清单用于 SHHS 6 位 ID) →
                # 退回排序序, 避免整集被试被静默丢弃 (训练会崩在空 train 或静默丢数据集)
                print(f"[WARNING] 顺序清单 {manifest} 与本数据集 "
                      f"(person_pool={self.person_pool}, id_pattern={self.id_pattern}) "
                      f"完全不匹配, 退回排序序", flush=True)
                files = sorted(files)
            else:
                missing = [s for s in want if s not in by_id]
                if missing:
                    print(f"[WARNING] 顺序清单 {manifest}: {len(missing)} 个 ID 数据缺失: {missing[:10]}")
                files = [by_id[s] for s in matched]
        else:
            files = sorted(files)
        # 只从文件名提取 ID, 避免路径中的数字串 (如 processed_data_20260805 的
        # 2026/0805) 被当成伪被试 ID (20260805 修复)
        subj_id = [re.findall(self.id_pattern, f.name)[0] for f in files]
        return pd.DataFrame(subj_id, columns=["subj_id"])
