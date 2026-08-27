import json
import re
from pathlib import Path

import pandas as pd
from tpcp import Dataset

from sleep_analysis.datasets.helper import build_base_path_processed_mesa, ordered_glob_csv


class MesaDataset(Dataset):
    """
    Dataset class for the MESA dataset created according to the tpcp framework (https://github.com/mad-lab-fau/tpcp)
    """

    def create_index(self):
        path = build_base_path_processed_mesa()

        path = path.joinpath("features_full_combined").resolve()
        # ✅2026-08-27: 确定性顺序 — sorted, 或 SLEEP_ORDER_MANIFEST 清单序 (精确复现历史 run)
        path_list = ordered_glob_csv(path, r"(\d{4})\.csv")
        # 20260805: 与 merge_features 同样的修复 — 只从文件名提取 ID,
        # 避免路径中的日期数字 (如 processed_data_no_leak_20260805 的 2026/0805)
        # 被当成伪被试 ID (会读取不存在的 features_combined2026.csv 崩溃)
        subj_id = [re.findall(r"(\d{4})\.csv", f.name)[0] for f in path_list]
        return pd.DataFrame(subj_id, columns=["subj_id"])

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
