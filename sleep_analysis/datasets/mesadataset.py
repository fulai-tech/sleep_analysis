import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from sleep_analysis.datasets.base_sleep_dataset import BaseSleepDataset
from sleep_analysis.datasets.helper import build_base_path_processed_mesa
from sleep_analysis.datasets.registry import register


@register("MESA_Sleep")
class MesaDataset(BaseSleepDataset):
    """
    Dataset class for the MESA dataset created according to the tpcp framework (https://github.com/mad-lab-fau/tpcp)

    20260822: 改为继承 BaseSleepDataset — ID 提取/模态默认值/特征列选择走基类钩子,
    新增数据集不再需要改训练管线。
    """

    id_pattern = r"(\d{4})\.csv"
    has_actigraphy = True
    person_pool = "mesa"
    modality_defaults = ["ACT", "HRV", "RRV"]

    def __init__(self, *, processed_path: Optional[Path] = None,
                 groupby_cols=None, subset_index=None, **kwargs):
        # 注意: 已是 Path 的对象原样存储 — tpcp 克隆校验要求 getattr 返回的
        # 参数与构造器入参是同一对象 (Path(p) 会新建对象导致 clone 报错)
        self.processed_path = (processed_path if isinstance(processed_path, Path)
                               else Path(processed_path)) if processed_path is not None \
            else build_base_path_processed_mesa()
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index, **kwargs)

    @property
    def actigraph_data(self):
        if self.is_single(["subj_id"]):
            path = build_base_path_processed_mesa()

            path = path.joinpath("actigraph_data_clean").resolve()
            return pd.read_csv(path.joinpath("actigraph_data_clean" + self.index["subj_id"][0] + ".csv"))[["activity"]]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def ground_truth(self):
        if self.is_single(["subj_id"]):
            path = build_base_path_processed_mesa()

            path = path.joinpath("actigraph_data_clean").resolve()
            return pd.read_csv(path.joinpath("actigraph_data_clean" + self.index["subj_id"][0] + ".csv"))[
                ["sleep", "5stage", "4stage", "3stage"]
            ]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def feature_table(self):
        if self.is_single(["subj_id"]):
            path = build_base_path_processed_mesa()

            path = path.joinpath("features_full_combined").resolve()
            return pd.read_csv(path.joinpath("features_combined" + self.index["subj_id"][0] + ".csv"), index_col=0)

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def time(self):
        if self.is_single(["subj_id"]):
            path = build_base_path_processed_mesa()

            path = path.joinpath("actigraph_data_clean").resolve()
            return pd.read_csv(path.joinpath("actigraph_data_clean" + self.index["subj_id"][0] + ".csv"))[["linetime"]]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def information(self):
        if self.is_single(["subj_id"]):
            with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
                path_dict = json.load(f)
                path = Path(path_dict["mesa_path"])
            df = pd.read_csv(path.joinpath("datasets/mesa-sleep-dataset-0.5.0.csv"), index_col="mesaid").fillna(0)
            df.index = df.index.astype(str).str.zfill(4)
            return df[
                [
                    "race1c",
                    "gender1",
                    "overall5",
                    "whiirs5c",
                    "slpapnea5",
                    "insmnia5",
                    "rstlesslgs5",
                    "sleepage5c",
                    "ahi_a0h4",
                    "extrahrs5",
                ]
            ].loc[self.index["subj_id"][0]]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def tst(self):
        if self.is_single(["subj_id"]):
            with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
                path_dict = json.load(f)
                path = Path(path_dict["mesa_path"])
            df = pd.read_csv(
                path.joinpath("datasets/mesa-sleep-harmonized-dataset-0.5.0.csv"), index_col="mesaid"
            ).fillna(0)
            df.index = df.index.astype(str).str.zfill(4)
            return df[["nsrr_ttldursp_f1"]].loc[self.index["subj_id"][0]]

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def edf_path(self):
        if self.is_single(["subj_id"]):
            with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
                path_dict = json.load(f)
                path = Path(path_dict["mesa_path_edf"])
            filename = re.findall("mesa-sleep-" + self.index["subj_id"][0] + ".edf", str(list(path.glob("*.edf"))))[0]
            return path.joinpath(filename)

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )
