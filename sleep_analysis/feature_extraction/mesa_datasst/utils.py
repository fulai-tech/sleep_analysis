import json
import re
from pathlib import Path

import pandas as pd
import tqdm

with open(Path(__file__).parents[3].joinpath("study_data.json")) as f:
    path_dict = json.load(f)
    mesa_path = Path(path_dict["mesa_path"])
    processed_mesa_path = Path(path_dict["processed_mesa_path_local"])


def merge_features(overwrite=False):
    """
    Merges actigraphy, HRV and Respiration features in one DataFrame that is then used to train the ML models
    Within each algorithm script the right features are filtered according to the dataset_name specified when running the script

    :param overwrite: If overwrite = True, the features are calculated and overwritten. If set to false, all features that are calculated are skipped.
    """

    # 20260805: 修 bug — 原实现 re.findall("(\d{4})", str(path_list)) 在
    # 整个路径字符串上匹配, 输出目录路径里的 4 位数字串（如 processed_data_no_leak_20260805
    # 里的 2026/0805）会被当成被试 ID, 导致读取不存在的文件崩溃。
    # 现改为只从文件名提取被试 ID。
    path_list = list(processed_mesa_path.joinpath("actigraph_data_clean").glob("*.csv"))
    mesa_ids = sorted({re.findall(r"(\d{4})\.csv", p.name)[0] for p in path_list})

    with tqdm.tqdm(total=len(mesa_ids)) as progress_bar:
        for subj in mesa_ids:
            if not overwrite:  # check if file already exists
                if check_processed(processed_mesa_path.joinpath("features_full_combined"), subj):
                    progress_bar.update(1)  # update progress
                    continue

            try:
                acti_features = pd.read_csv(
                    processed_mesa_path.joinpath("actigraph_features/actigraph_features" + subj + ".csv")
                ).dropna()
                hrv_features = pd.read_csv(
                    processed_mesa_path.joinpath("hrv_features/hrv_features" + subj + ".csv")
                ).dropna()
                resp_features = pd.read_csv(
                    processed_mesa_path.joinpath("respiration_features_clean/respiration_features" + subj + ".csv")
                ).dropna()
                edr_features = pd.read_csv(
                    processed_mesa_path.joinpath("edr_features_clean/edr_features" + subj + ".csv")
                ).dropna()
            except FileNotFoundError as e:
                # 20260805: 单个被试文件缺失时跳过, 不中断整个 merge
                # (如 Step 1-5 中被跳过/失败且未生成完整文件的被试)
                print(f"  [{subj}] SKIP merge: missing file {e.filename}")
                progress_bar.update(1)
                continue

            try:
                assert (
                    acti_features.shape[0] == resp_features.shape[0] == hrv_features.shape[0] == edr_features.shape[0]
                )
            except AssertionError:
                print(subj)

            feature_full = pd.concat([acti_features, hrv_features, resp_features, edr_features], axis=1)

            feature_full.to_csv(
                processed_mesa_path.joinpath("features_full_combined/features_combined" + subj + ".csv")
            )
            progress_bar.update(1)

    print("merge finished")


def check_processed(path, mesa_id):
    path_list = list(Path(path).glob("*.csv"))
    # 20260805: 与 merge_features 同样的 bug 修复 — 只从文件名提取 ID, 避免路径数字污染
    mesa_id_list = {re.findall(r"(\d{4})\.csv", p.name)[0] for p in path_list}

    if mesa_id in mesa_id_list:
        print("Mesa_id " + mesa_id + " already extracted")
        return True
    else:
        return False
