import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from hrvanalysis import (
    get_csi_cvi_features,
    get_frequency_domain_features,
    get_geometrical_features,
    get_poincare_plot_features,
    get_time_domain_features,
)

from sleep_analysis.feature_extraction.mesa_datasst.utils import check_processed

with open(Path(__file__).parents[3].joinpath("study_data.json")) as f:
    path_dict = json.load(f)
    mesa_path = Path(path_dict["mesa_path"])
    processed_mesa_path = Path(path_dict["processed_mesa_path"])


def extract_hrv_features(overwrite=True):
    """
    Extract HRV features from MESA dataset
    Calculates 30 different HRV features from time-domain, frequency-domain and non-linear domain.

    :param overwrite: If overwrite = True, the features are calculated and overwritten. If set to false, all features that are calculated are skipped.
    """

    path_list = list(processed_mesa_path.joinpath("actigraph_data_clean").glob("*.csv"))
    # 20260805: 只从文件名提取 ID, 避免路径中的日期数字污染
    mesa_id = [re.findall(r"(\d{4})\.csv", f.name)[0] for f in path_list]
    with tqdm.tqdm(total=len(mesa_id)) as progress_bar:
        for subj in mesa_id:
            if not overwrite:  # check if file already exists
                if check_processed(processed_mesa_path.joinpath("hrv_features"), subj):
                    progress_bar.update(1)
                    continue

            df_hr = pd.read_csv(processed_mesa_path.joinpath("ecg_data_clean/ecg_data_clean" + subj + ".csv"))

            hr_features = calc_hrv_features(df_hr)

            del hr_features["_hrv_epoch"]

            hr_features.to_csv(processed_mesa_path.joinpath("hrv_features/hrv_features" + subj + ".csv"), index=False)

            print("Features extraction of HRV of subj: " + subj + " finished!")
        progress_bar.update(1)  # update progress


def calc_hrv_features(df_hr: pd.DataFrame):
    """
    Compute HRV features from time-domain, frequency-domain and non-linear domain. This is done via the python package hrvanalysis. This function returns a total of 30 HRV features.
    :param df_hr: pd.DataFrame that contains RR-intervals
    :returns: pd.DataFrame that contains all HRV features
    """
    # ✅2026-09-24: 排除**插值补出来的拍** —— 线性插值补出的段是等差数列，
    #   会让 SDSD≈0 → SD1=sqrt(SDSD²/2)≈0 → ratio_sd2_sd1 = SD2/SD1 发散
    #   （实测 1e13~1e15，把 StandardScaler 的 mean/std 拉到 8.2e9/4.8e12，
    #   该特征在模型里恒为常数）。频域 4 个槽位同样受污染（实测偏 12~35%）。
    #   掩码由 preprocessing/rr_utils.py::process_rpoint 写入。
    if "interpolated" not in df_hr.columns:
        raise KeyError(
            "输入缺少 `interpolated` 列 —— 这是 2026-09-24 之前产出的旧 ecg_data_clean，"
            "其中插值补出来的心搏会被当成真实测量参与 HRV 计算，导致 ratio_sd2_sd1 "
            "出现 1e13~1e15 的伪值。请用新代码重新生成预处理数据"
            "（见 preprocessing/rr_utils.py::process_rpoint）。")

    hr_epoch_set = set(df_hr["epoch"].values)

    all_hr_features = {}
    for i, hr_epoch_idx in enumerate(list(hr_epoch_set)):
        tmp_hr_df = df_hr[df_hr["epoch"] == hr_epoch_idx]
        tmp_hr_df = tmp_hr_df[tmp_hr_df["interpolated"] == 0]
        # ⚠️ 下面 `tmp_hr_df.size > 3` 是个**失效的门槛**：tmp_hr_df 有 20+ 列，
        #    `.size` = 行数 × 列数，等价于「至少有 1 行」。真正保证拍数够的是上游
        #    rr_utils 的 `MIN_REAL_RR_PER_EPOCH=10` 整 epoch 剔除。
        #    保留原样不改（改它会变更语义），但别再以为这里卡了拍数。
        if tmp_hr_df.size > 3:
            rr_epoch = tmp_hr_df["RR Intervals"].values

            all_hr_features[hr_epoch_idx] = {}
            all_hr_features[hr_epoch_idx].update(get_time_domain_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update(get_frequency_domain_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update(get_poincare_plot_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update(get_csi_cvi_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update(get_geometrical_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update({"epoch": hr_epoch_idx})
    all_hr_features = pd.DataFrame(all_hr_features).T
    del all_hr_features["tinn"]

    # eventually needed for covering the issue of some samples with nan inf or -inf
    with pd.option_context("mode.use_inf_as_na", True):
        all_hr_features.fillna(0.0)
        all_hr_features.replace([np.inf, -np.inf], 0.0, inplace=True)

    all_hr_features.columns = ["_hrv_" + str(col) for col in all_hr_features.columns]

    return all_hr_features.reset_index(drop=True)
