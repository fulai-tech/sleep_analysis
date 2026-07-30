"""SHHS Dataset class — 兼容 tpcp Dataset 接口，供训练管线使用。"""

import json
import re
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tpcp import Dataset


def _get_shhs_config(study: str):
    """读取 study_data.json 中与指定 SHHS 子集相关的路径。"""
    config_path = Path(__file__).parents[2] / "study_data.json"
    with open(config_path) as f:
        cfg = json.load(f)
    return {
        "processed_path": Path(cfg[f"{study}_processed_path"]),
        "data_path": Path(cfg[f"{study}_path"]),
    }


class ShhsDataset(Dataset):
    """SHHS 数据集类，兼容 MESA 训练管线。

    Parameters
    ----------
    study : str
        "shhs1" 或 "shhs2"。
    groupby_cols : optional
        传给 tpcp.Dataset 基类。
    subset_index : optional
        传给 tpcp.Dataset 基类。
    """

    groupby_cols: Optional[Union[List[str], str]]
    subset_index: Optional[pd.DataFrame]

    def __init__(self,
                 study: str = "shhs1",
                 *,
                 groupby_cols: Optional[Union[List[str], str]] = None,
                 subset_index: Optional[pd.DataFrame] = None,
                 **kwargs):
        self.study = study
        self._config = _get_shhs_config(study)
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index, **kwargs)

    def create_index(self):
        path = self._config["processed_path"] / "features_full_combined"
        path_list = list(path.glob("*.csv"))
        # SHHS ID 为 6 位数字
        subj_id = re.findall(r"(\d{6})", str(path_list))
        return pd.DataFrame(subj_id, columns=["subj_id"])

    @property
    def ground_truth(self):
        if self.is_single(["subj_id"]):
            sid = self.index["subj_id"][0]
            p = self._config["processed_path"] / "sleep_stages" / f"sleep_stages{sid}.csv"
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
            p = self._config["processed_path"] / "features_full_combined" / f"features_combined{sid}.csv"
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
