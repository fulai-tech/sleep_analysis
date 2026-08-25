"""SHHS Dataset class — 兼容 tpcp Dataset 接口，供训练管线使用。"""

import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from sleep_analysis.datasets.base_sleep_dataset import BaseSleepDataset
from sleep_analysis.datasets.registry import register


def _get_shhs_config(study: str):
    """读取 study_data.json 中与指定 SHHS 子集相关的路径。"""
    config_path = Path(__file__).parents[2] / "study_data.json"
    with open(config_path) as f:
        cfg = json.load(f)
    return {
        "processed_path": Path(cfg[f"{study}_processed_path"]),
        "data_path": Path(cfg[f"{study}_path"]),
    }


@register("SHHS1")
@register("SHHS2")
class ShhsDataset(BaseSleepDataset):
    """SHHS 数据集类，兼容 MESA 训练管线。

    Parameters
    ----------
    study : str
        "shhs1" 或 "shhs2"。
    processed_path : optional
        覆盖 study_data.json 的处理后数据目录 (测试用)。
    groupby_cols : optional
        传给 tpcp.Dataset 基类。
    subset_index : optional
        传给 tpcp.Dataset 基类。
    """

    id_pattern = r"(\d{6})\.csv"
    has_actigraphy = False
    person_pool = "shhs"   # SHHS1/SHHS2 共享 nsrrid → 联合人员级划分
    modality_defaults = ["HRV", "RRV"]

    groupby_cols: Optional[Union[List[str], str]]
    subset_index: Optional[pd.DataFrame]

    def __init__(self,
                 study: str = "shhs1",
                 *,
                 processed_path: Optional[Path] = None,
                 groupby_cols: Optional[Union[List[str], str]] = None,
                 subset_index: Optional[pd.DataFrame] = None,
                 **kwargs):
        self.study = study
        self._config = _get_shhs_config(study)
        # 注意: 已是 Path 的对象原样存储 — tpcp 克隆校验要求参数与属性同一对象
        self.processed_path = (processed_path if isinstance(processed_path, Path)
                               else Path(processed_path)) if processed_path is not None \
            else self._config["processed_path"]
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index, **kwargs)

    @property
    def ground_truth(self):
        if self.is_single(["subj_id"]):
            sid = self.index["subj_id"][0]
            p = self.processed_path / "sleep_stages" / f"sleep_stages{sid}.csv"
            df = pd.read_csv(p)
            # sleep_stages 文件列: sleep, 5stage, 4stage, 3stage, epoch
            cols = [c for c in ["sleep", "5stage", "4stage", "3stage"] if c in df.columns]
            return df[cols]
        raise ValueError("Data can only be accessed when there is only a single "
                         "recording of a single participant in the subset")

    @property
    def feature_table(self):
        if self.is_single(["subj_id"]):
            sid = self.index["subj_id"][0]
            p = self.processed_path / "features_full_combined" / f"features_combined{sid}.csv"
            return pd.read_csv(p, index_col=0)
        raise ValueError("Data can only be accessed when there is only a single "
                         "recording of a single participant in the subset")

    @property
    def information(self):
        """人口学 / 临床信息。SHHS1 和 SHHS2 字段略有差异，取可用部分。"""
        if self.is_single(["subj_id"]):
            sid = int(self.index["subj_id"][0])

            if self.study == "shhs1":
                csv_file = self._config["data_path"] / "datasets" / "shhs1-dataset-0.13.0.csv"
                df = pd.read_csv(csv_file, index_col="nsrrid").fillna(0)
                # SHHS1 中可用的列
                cols = ["gender", "race", "ahi_a0h4"]
                cols = [c for c in cols if c in df.columns]
                df.index = df.index.astype(str).str.zfill(6)
                return df[cols].loc[str(sid)]
            else:
                csv_file = self._config["data_path"] / "datasets" / "shhs2-dataset-0.13.0-utf8.csv"
                df = pd.read_csv(csv_file, index_col="nsrrid").fillna(0)
                cols = ["gender", "race", "ahi_a0h4"]
                cols = [c for c in cols if c in df.columns]
                if sid in df.index:
                    return df[cols].loc[sid]
                return pd.Series({c: 0 for c in cols})
        raise ValueError("Data can only be accessed when there is only a single "
                         "recording of a single participant in the subset")
