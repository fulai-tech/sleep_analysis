import json
import re
from pathlib import Path

import biopsykit as bp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal
import tqdm
from biopsykit.utils.array_handling import sliding_window

import sleep_analysis.processing_config as pc
from sleep_analysis.feature_extraction.mesa_datasst.utils import check_processed
from sleep_analysis.preprocessing.utils import extract_edf_channel


def extract_rrv_features(overwrite=False):
    """
    Extracts RRV features from the MESA dataset.
    Calculates 62 different RRV features in sliding windows of 5 min (10 epochs), 7 min (14 epochs) and 9 min (18 epochs) with overlap of 30s (1 epoch).

    :param overwrite: If overwrite = True, the features are calculated and overwritten. If set to false, all features that are calculated are skipped.
    """

    with open(Path(__file__).parents[3].joinpath("study_data.json")) as f:
        path_dict = json.load(f)
        edf_path = Path(path_dict["mesa_path"]).joinpath("polysomnography/edfs")
        processed_mesa_path = Path(path_dict["processed_mesa_path"])

    path_list = list(Path(edf_path).glob("*.edf"))
    mesa_id = re.findall("(\d{4})", str(path_list))

    with tqdm.tqdm(total=len(mesa_id)) as progress_bar:
        for subj in mesa_id:
            if not overwrite:  # check if file already exists
                if check_processed(Path(path_dict["processed_mesa_path"]).joinpath("respiration_features_raw"), subj):
                    progress_bar.update(1)  # update progress
                    continue

            # read in .edf file and process to respiration dataframe
            # tmin and tmax are variables to check edf file streams (crops to a smaller part)
            resp_df, epochs = extract_edf_channel(edf_path, subj_id=int(subj), channel="Thor")
            resp_df, epochs = process_resp(resp_df, epochs)
            features = extract_rrv_features_helper(resp_df)
            features.to_csv(
                Path(path_dict["processed_mesa_path"]).joinpath(
                    "respiration_features_raw/respiration" + str(subj) + ".csv"
                )
            )
            progress_bar.update(1)  # update progress

            print("Features extraction of RRV of subj: " + subj + " finished!")


def extract_rrv_features_helper(resp_arr, nan_pad=1.0, sampling_rate=32):
    # 20260729 - rdwang: 注释和代码不一致，作者实际用的是5个epoch、7个epoch、9个epoch，分别对应2.5min、3.5min、4.5min，与当前注释不一致
    """
    Calculate features for sliding widows of 5 min (10 epochs), 7 min (14 epochs) and 9 min (18 epochs) with overlap of 30s (1 epoch)
    according to Fonseca et al., 2015
    zero-padding at beginning and end because of sliding window
    Feature DataFrame with prefix 150_, 210_ and 270_ for features of 5 min, 7 min and 9-min window size

    :param resp_arr: respiration datastream
    :param nan_pad: value to pad NaN values with
    """

    # resp_arr_30s = sliding_window(resp_arr, 30*32, overlap_samples=0)

    # mode=mean to prevent first epochs to be zero --> no breathing extracted --> exception

    # 20260730 - rdwang: 作者的数据处理脏代码，非致命bug。目前是首尾各补(窗长-1)/2长度的均值，更好的操作应该是直接先给原始数据padding首尾以及在尾部补足整除不足的部分，而不是现在的补nan，然后再过sliding_window。这样能保证没有nan，且首尾涉及padding的数据中，只有缺失数据段被补成了均值，而不是整个数据都是均值
    resp_arr_150s = np.nan_to_num(
        np.pad(
            sliding_window(resp_arr, 150 * sampling_rate, overlap_samples=120 * sampling_rate),
            ((2, 2), (0, 0)),
            mode="mean",
        ),
        nan=nan_pad,
    )
    resp_arr_210s = np.nan_to_num(
        np.pad(
            sliding_window(resp_arr, 210 * sampling_rate, overlap_samples=180 * sampling_rate),
            ((3, 3), (0, 0)),
            mode="mean",
        ),
        nan=nan_pad,
    )
    resp_arr_270s = np.nan_to_num(
        np.pad(
            sliding_window(resp_arr, 270 * sampling_rate, overlap_samples=240 * sampling_rate),
            ((4, 4), (0, 0)),
            mode="mean",
        ),
        nan=nan_pad,
    )

    # 20260805: 修复异常分支引用未定义变量的 bug (首个窗口异常时 features_150 未定义)
    feature_list = []
    feature_keys = None  # 首次成功时的特征列名, 供异常窗口填 0 使用
    for resp_150 in resp_arr_150s:
        try:
            peaks = extract_peaks(resp_150, sampling_rate)
            features_150 = calc_rrv_features(resp_150, peaks, sampling_rate)
            if feature_keys is None:
                feature_keys = list(features_150[0].keys())
            feature_list.append(features_150[0])
        except (ValueError, IndexError):
            print("handle peak-detection error (150s window)")
            feature_list.append(dict.fromkeys(feature_keys or [], 0))
            continue

    features = pd.DataFrame(feature_list).add_prefix("150_")

    feature_list = []
    feature_keys = None
    for resp_210 in resp_arr_210s:
        try:
            peaks = extract_peaks(resp_210, sampling_rate)
            features_210 = calc_rrv_features(resp_210, peaks, sampling_rate)
            if feature_keys is None:
                feature_keys = list(features_210[0].keys())
            feature_list.append(features_210[0])

        except (ValueError, IndexError):
            print("handle peak-detection error (210s window)")
            feature_list.append(dict.fromkeys(feature_keys or [], 0))
            continue

    features = pd.concat([features, pd.DataFrame(feature_list).add_prefix("210_")], axis=1)

    feature_list = []
    feature_keys = None
    for resp_270 in resp_arr_270s:
        try:
            peaks = extract_peaks(resp_270, sampling_rate)
            features_270 = calc_rrv_features(resp_270, peaks, sampling_rate)
            if feature_keys is None:
                feature_keys = list(features_270[0].keys())
            feature_list.append(features_270[0])

        except (ValueError, IndexError):
            print("handle peak-detection error (270s window)")
            feature_list.append(dict.fromkeys(feature_keys or [], 0))
            continue

    features = pd.concat([features, pd.DataFrame(feature_list).add_prefix("270_")], axis=1)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    features.fillna(0.0, inplace=True)

    time_axis = resp_arr.index.round("30s").drop_duplicates()[0 : features.shape[0]]
    features.index = time_axis
    features["epoch"] = np.arange(1, features.shape[0] + 1)

    if features.shape[1] == 70:
        print("length 70")
        features = features[features.columns.drop(list(features.filter(regex="210_RRV_DFA")))]

    return features


def calc_rrv_features(rsp_rate, peaks_dict, sampling_rate: int):
    """
    RRV (Respiratory Rate Variability) Features extracted via python library neurokit
    :param: rsp_rate: extracted rsp_rate from edf file
            peaks_dict: extracted peaks from rsp_signal
            sampling_rate: sampling rate of signal - commonly 32 Hz
    :return: Features compressed in a dict
    """
    rrv = nk.rsp_rrv(rsp_rate, peaks_dict, sampling_rate=sampling_rate, show=False)

    return rrv.to_dict("records")


def process_resp(resp_df, epochs, sampling_rate_in=256):
    """Downsample the respiration signal to 32 Hz and align the epoch / time index to it.

    The epoch and time index are re-mapped by sample position (``j * sampling_rate_in /
    sampling_rate_out``) rather than sliced with an integer stride. Integer-stride slicing
    (``[::step]`` with ``step = int(sampling_rate_in / 32)``) only matches the resampled
    length when ``sampling_rate_in`` is an exact multiple of 32:

        256 Hz -> 32 Hz: factor 8.0     -> stride slicing works (MESA / some SHHS2 files)
        250 Hz -> 32 Hz: factor 7.8125  -> stride 7 != 7.8125 -> length mismatch
        125 Hz -> 32 Hz: factor 3.90625 -> stride 3 != 3.90625 -> length mismatch (SHHS1)

    For non-multiple-of-32 rates the truncated stride desynchronises the index from the
    resampled data, so ``pd.DataFrame(resp_arr, index=time_index)`` raised
    "Shape of passed values ... indices imply ...". Mapping by sample position guarantees
    ``len(time_index) == len(epochs) == len(resp_arr)`` for any input sampling rate.
    """
    time_index = resp_df.index

    sampling_rate_out = 32
    resp_arr = _downsample_resp(resp_df, sampling_rate_in=sampling_rate_in, sampling_rate_out=sampling_rate_out)
    n_out = len(resp_arr)
    indices = (np.arange(n_out) * sampling_rate_in / sampling_rate_out).astype(int)
    indices = np.clip(indices, 0, len(epochs) - 1)
    epochs = epochs[indices]
    time_index = time_index[indices]

    resp_df = pd.DataFrame(resp_arr, index=time_index)
    return resp_df, epochs


def _rsp_clean_causal(resp_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """因果版呼吸信号清洗 — 与 nk.rsp_clean(method="biosppy") 同参数，但只使用已见数据。

    原版 (neurokit2 _rsp_clean_biosppy): butter(2) bandpass [0.1, 0.35] Hz + filtfilt
    （零相位双向滤波, 未来泄漏）+ detrend(0)（减整夜均值, 未来泄漏）。

    因果版: 同参数 butter(2) + lfilter（正向滤波, zi 稳态初始化）。
    去掉 detrend(0): 它减去整夜均值（未来泄漏）；0.1Hz 高通已去除基线漂移，影响极小。
    """
    b, a = scipy.signal.butter(N=2, Wn=[0.1, 0.35], btype="bandpass", fs=sampling_rate)
    zi = scipy.signal.lfilter_zi(b, a) * resp_signal[0]
    cleaned, _ = scipy.signal.lfilter(b, a, resp_signal, zi=zi)
    return cleaned


def _downsample_causal(data: np.ndarray, sampling_rate_in: int, sampling_rate_out: int) -> np.ndarray:
    """因果版降采样。

    原版 (biopsykit downsample): 整数比用 decimate（因果）; 非整数比用
    filtfilt(cheby1(8) 抗混叠) + 线性插值（双向滤波, 未来泄漏, SHHS1 125Hz / SHHS2 250Hz 会走此分支）。

    因果版: 整数比保持 decimate 但显式 zero_phase=False（scipy 1.13 默认 True=内部 filtfilt,
    非因果 — 原版 MESA 256→32 实际走的就是 filtfilt）; 非整数比将抗混叠滤波换成
    lfilter（因果, zi 稳态初始化），参数与原版完全相同 (cheby1 N=8, rp=0.05, Wn=0.8/(fs_in/fs_out))，
    插值保持线性。残余向前依赖仅 1 个原始样本（≤8ms@125Hz, ≤16ms@250Hz），远小于 30s epoch，可忽略。
    """
    if (sampling_rate_in / sampling_rate_out) % 1 == 0:
        return scipy.signal.decimate(
            data, int(sampling_rate_in / sampling_rate_out), axis=0, zero_phase=False
        )

    b, a = scipy.signal.cheby1(N=8, rp=0.05, Wn=0.8 / (sampling_rate_in / sampling_rate_out))
    zi = scipy.signal.lfilter_zi(b, a) * data[0]
    data_lp, _ = scipy.signal.lfilter(b, a, data, zi=zi)

    x_old = np.linspace(0, len(data_lp), num=len(data_lp), endpoint=False)
    x_new = np.linspace(0, len(data_lp), num=int(len(data_lp) / (sampling_rate_in / sampling_rate_out)), endpoint=False)
    return np.interp(x_new, x_old, data_lp)


def _downsample_resp(resp_df, sampling_rate_in: int, sampling_rate_out: int):
    # ✅20260729 - rdwang: 这里是全部数据直接做的双向滤波，不符合事实睡眠分期的需求，要改整个处理链路
    # ✅20260805 - rdwang: 已加 causal 分支（processing_config.causal=True 时用正向滤波 + 因果降采样）
    if pc.causal:
        # 单列 DataFrame 拉平为 1-D (原版 nk.rsp_clean 内部也是取单列)
        cleaned = _rsp_clean_causal(np.asarray(resp_df, dtype=float).ravel(), sampling_rate_in)
        return _downsample_causal(cleaned, sampling_rate_in, sampling_rate_out)

    cleaned = nk.rsp_clean(resp_df, sampling_rate=sampling_rate_in, method="biosppy")

    return bp.utils.array_handling.downsample(np.asarray(cleaned), sampling_rate_in, sampling_rate_out)


def extract_peaks(resp_df, sampling_rate: int):
    # Extract peaks
    df, peaks_dict = nk.rsp_peaks(resp_df, sampling_rate=sampling_rate, method="biosppy")
    info = nk.rsp_fixpeaks(peaks_dict)
    # formatted = nk.signal_formatpeaks(info, desired_length=len(resp_df), peak_indices=info["RSP_Peaks"])

    # candidate_peaks = nk.events_plot(peaks_dict["RSP_Peaks"], resp_df)
    # fixed_peaks = nk.events_plot(info["RSP_Peaks"], resp_df)

    return info


def _extract_rsp_rate(resp_df, peaks_dict, sampling_rate: int):
    # Extract rate
    rsp_rate = nk.rsp_rate(resp_df, peaks_dict, sampling_rate=sampling_rate)

    return rsp_rate


def _plot_rsp_rate(rsp_rate, peaks, sampling_rate: int):
    # Visualize
    nk.signal_plot(rsp_rate, sampling_rate=sampling_rate)
    plt.rcParams["figure.figsize"] = 15, 5  # Bigger images

    candidate_peaks = nk.events_plot(peaks["RSP_Peaks"], rsp_rate)
    plt.xlabel("Samples (32 Hz)")
    plt.title("Breath detection from breathing cycle acquired by respiratory band")
    plt.show()


def _plot_features(rsp_rate, peaks_dict, sampling_rate: int):
    rrv = nk.rsp_rrv(rsp_rate, peaks_dict, sampling_rate=sampling_rate, show=True)
    plt.show()
