import json
import os
import re
from pathlib import Path
import platform

import pandas as pd
import pytz
from empkins_io.sensors.emrad import EmradDataset
from sklearn.model_selection import train_test_split


def ordered_glob_csv(path: Path, id_pattern: str, manifest_env: str = "SLEEP_ORDER_MANIFEST"):
    """glob 特征 CSV 并按确定顺序返回文件列表。

    ✅2026-08-27 修复: Path.glob 不排序 → 被试顺序 = 文件系统目录序,
    跨机器不一致 (目录序不同 → batch 组成不同 → 同参数训练结果分叉,
    见 2026-08-17 run 在 212 与本机复现差异)。
    - 默认: sorted(glob) — 跨机器确定性可复现。
    - 可选: 设置 SLEEP_ORDER_MANIFEST=<每行一个被试 ID 的文件>, 按清单顺序排列,
      用于精确复现历史 run (清单 = 该 run 所在机器的目录序, 如 splits/order_mesa_212_20260817.txt)。
    """
    files = list(Path(path).glob("*.csv"))
    manifest = os.environ.get(manifest_env)
    if manifest:
        want = [l.strip() for l in open(manifest) if l.strip()]
        by_id = {re.findall(id_pattern, f.name)[0]: f for f in files}
        missing = [s for s in want if s not in by_id]
        if missing:
            print(f"[WARNING] 顺序清单 {manifest}: {len(missing)} 个 ID 数据缺失: {missing[:10]}")
        files = [by_id[s] for s in want if s in by_id]
        if len(files) != len(want):
            print(f"[WARNING] 顺序清单 {manifest}: 期望 {len(want)} 项, 实际匹配 {len(files)} 项")
    else:
        files = sorted(files)
    return files


def _load_radar_data(path):
    radar_path = list(Path(path).glob("Vp_*"))
    if len(radar_path) == 0:
        return None
    data = EmradDataset.from_hd5_file(radar_path[0])

    return data


def _build_base_path(dataset_name: str):
    with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
        path_dict = json.load(f)

        if dataset_name == "empkins_mainstudy_control":
            if platform.system() == "Linux":
                return Path(path_dict["empkins_mainstudy_control_hpc"])
            elif platform.system() == "Darwin":
                return Path(path_dict["empkins_mainstudy_control"])
        elif dataset_name == "empkins_mainstudy_control_local":
            return Path(path_dict["empkins_mainstudy_control_local"])
        elif dataset_name == "empkins_mainstudy_pd":
            if platform.system() == "Linux":
                return Path(path_dict["empkins_mainstudy_pd_hpc"])
            elif platform.system() == "Darwin":
                return Path(path_dict["empkins_mainstudy_pd_local"])
        else:
            raise ValueError("Unknown dataset_name")


def build_base_path_processed(dataset_name):

    with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
        path_dict = json.load(f)

        if dataset_name == "empkins_mainstudy_control":
            return Path(path_dict["processed_D04_hpc_control"])
        else:
            return Path(path_dict["processed_D04_hpc_pd"])


def build_base_path_processed_local():
    with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
        path_dict = json.load(f)
        return Path(path_dict["processed_D04_local"])


def build_base_path_processed_mesa():
    with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
        path_dict = json.load(f)
        # Check the operating system
        if platform.system() == "Linux":
            return Path(path_dict["processed_mesa_path_hpc"])
        elif platform.system() == "Darwin":  # Darwin is the system name for macOS
            return Path(path_dict["processed_mesa_path_testing"])
        else:
            raise EnvironmentError("Unsupported operating system")


def _load_exclusion_list(json_path):
    """Load the list of participants to exclude from the JSON file."""
    json_path = json_path.joinpath("exclusion.json")
    try:
        with open(json_path, "r") as file:
            exclusion_list = json.load(file)
        return exclusion_list.get("exclude", [])
    except FileNotFoundError:
        print(f"Warning: {json_path} not found. No participants will be excluded.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {json_path} is not a valid JSON file.")
        return []


def _exclude_data(df, exclusion_criteria, vp_id):

    if exclusion_criteria is None:
        return df

    path = _build_base_path()
    path = path.joinpath("Vp_" + vp_id)

    with open(path.joinpath("exclusion.json")) as f:
        exclusion_dict = json.load(f)

    if "EEG" in exclusion_criteria:
        start = exclusion_dict["EEG"]["start"]
        end = exclusion_dict["EEG"]["end"]

        print("Excluding EEG data from", start, "to", end)
        print(df.index[0], df.index[-1])

        if start == "":
            df = df.loc[end:]

        elif end == "":
            df = df.loc[:start]

        else:
            df = pd.concat([df.loc[:start], df.loc[end:]])

        # check if "EEG2" exists in exclusion_dict
        if "EEG2" in exclusion_dict.keys():
            start = exclusion_dict["EEG2"]["start"]
            end = exclusion_dict["EEG2"]["end"]

            print("Excluding another batch of EEG data from", start, "to", end)
            print(df.index[0], df.index[-1])

            if start == "":
                df = df.loc[end:]

            elif end == "":
                df = df.loc[:start]

            else:
                df = pd.concat([df.loc[:start], df.loc[end:]])

    return df


def get_random_split(dataset, train_size=0.8):
    train, test = train_test_split(dataset, train_size=train_size, random_state=42, shuffle=True)

    return train, test


def get_concat_dataset(dataset, modality):
    features = {}
    ground_truth = {}
    modality_str = "|".join(modality)
    mod_set = {"HRV", "ACT", "RRV", "EDR"}

    if not all({mod}.issubset(mod_set) for mod in modality):
        raise AttributeError("modality MUST be list of either HRV, ACT, RRV, EDR")

    for subj in dataset:
        features[subj.index["subj_id"][0]] = subj.feature_table.filter(regex=modality_str)
        ground_truth[subj.index["subj_id"][0]] = subj.ground_truth

    features = pd.concat(features)
    ground_truth = pd.concat(ground_truth)

    return features, ground_truth


def get_features(datapoint, modality):
    modality_str = "|".join(modality)
    mod_set = {"HRV", "ACT", "RRV", "EDR"}
    if not all({mod}.issubset(mod_set) for mod in modality):
        raise AttributeError("modality MUST be list of either HRV, ACT, RRV, EDR")

    features = datapoint.feature_table.filter(regex=modality_str)

    return features
