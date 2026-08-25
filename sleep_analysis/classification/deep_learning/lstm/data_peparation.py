import biopsykit as bp
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from tpcp import Dataset

from sleep_analysis.feature_extraction.d04_main.combine_clean import _map_sleep_phases_to_num


def create_tensor(x_normalized, y_mat):
    # convert into tensor
    x_tensor_final = Variable(torch.Tensor(x_normalized))
    y_tensor = Variable(torch.Tensor(y_mat))

    # reshape it to (number of windows, window_length, number of features)
    y_tensor = torch.reshape(y_tensor, (y_tensor.shape[0], 1))

    return x_tensor_final, y_tensor


def test_to_list(subj, x_list, y_list, x_test_subj, y_test_subj):
    if subj.__class__.__name__ == "RealWorldIMUSet" or subj.__class__.__name__ == "RealWorldData":  # Deprecated
        x_list.append([x_test_subj, subj.index["subj_id"][0] + "_" + subj.index["night"][0]])
        y_list.append([y_test_subj, subj.index["subj_id"][0] + "_" + subj.index["night"][0]])
    elif subj.__class__.__name__ == "D04MainStudy":
        x_list.append([x_test_subj, subj.index["subj_id"][0]])
        y_list.append([y_test_subj, subj.index["subj_id"][0]])
    else:
        x_list.append([x_test_subj, subj.index["subj_id"][0]])
        y_list.append([y_test_subj, subj.index["subj_id"][0]])
    return x_list, y_list


def batchify(x_mat):
    return np.array_split(x_mat, 10)


class DataPreparation:
    """
    Data Preparation class:
    Returns sequential data for the corresponding input modality and a combination of those
    Supported modalities:
    *ACT:       Actigraphy
    *HRV:       Heart Rate Variability
    *RRV:       Respiration
    :param seq_len: Sequence length; Sequence that gets fed into LSTM
    :param overlap: Overlap of Sequences: Highly impacts runtime
    """

    def __init__(self, seq_len, overlap, causal=False):
        self.seq_len = seq_len
        self.overlap = overlap
        self.causal = causal   # True=只在左侧padding (实时分期), False=居中padding (原文)

    def get_sequence_data(self, features: pd.DataFrame, ground_truth: pd.DataFrame, overlap, padding=False):
        """
        Create sequential data using biopsykit
        :param features: features that should be sequential
        :param ground_truth: ground truth to crop it to same length (sliding window cuts window/s from beginning and end)
        """

        if overlap is None:
            self.overlap = None

        feature_arr = np.asarray(features)
        ground_truth_arr = np.asarray(ground_truth)

        # 20260731 - rdwang: 确定padding有数据泄漏，无法用于实时分期，具体待查
        # 2026-08-07 - rdwang: causal 分支 padding 值从整夜均值改为首值 (mode="edge")
        #   原 mode="mean" 用整夜特征均值(含未来 epoch)填充, 序列开头窗口受未来影响;
        #   mode="edge" 用第一帧特征值填充 (实时语义: 开始时只有最早到达的数据, 无泄漏)。
        #   scaler 是逐特征线性变换, 与 edge-pad 可交换 → 等价于"先 scale 再首值 pad"。
        #   非 causal 分支保持原版 (mode="mean", 复现作者路径不动)。
        if padding:
            if self.causal:
                npad = ((self.seq_len - 1, 0), (0, 0))   # 只垫历史，预测最后时刻
                feature_arr = np.pad(feature_arr, npad, mode="edge")
            else:
                npad = ((int(self.seq_len / 2), int(self.seq_len / 2)), (0, 0))  # 居中，原文方式
                feature_arr = np.pad(feature_arr, npad, mode="mean")
            y_mat = ground_truth_arr
            x_mat = bp.utils.array_handling.sliding_window(
                feature_arr.squeeze(), overlap_percent=self.overlap, window_samples=self.seq_len
            )

        if not padding:
            y_mat = bp.utils.array_handling.sliding_window(
                ground_truth_arr.squeeze(), overlap_percent=self.overlap, window_samples=self.seq_len
            )[:, int(self.seq_len - 1)][:-1]
            x_mat = bp.utils.array_handling.sliding_window(
                feature_arr.squeeze(), overlap_percent=self.overlap, window_samples=self.seq_len
            )[:-1]

        return x_mat, y_mat

    def scale_data(self, x_mat, scaler: StandardScaler):
        """
        Normalize all datasets with the scaling obtained from training set
        If train set: initialize new scaler - If val/test: take scaler from train_set obtained from parameters
        :param x_mat: sequential data of x_tensor
        :param scaler: StandardScaler if val/test set or None and new initialisation if train set
        """
        if scaler is None:
            scaler = StandardScaler()
            for data in batchify(x_mat):
                scaler.partial_fit(data.reshape(-1, data.shape[-1]))
            x_normalized = scaler.transform(x_mat.reshape(-1, x_mat.shape[-1])).reshape(x_mat.shape)
        else:
            x_normalized = scaler.transform(x_mat.reshape(-1, x_mat.shape[-1])).reshape(x_mat.shape)

        return x_normalized, scaler

    def _extract_subj_features_raw(self, subj, dataset, modality, classification_type="binary"):
        """按 modality 选择特征列并返回 (features_df, ground_truth_df)。

        ✅2026-08-11 - rdwang: 从 get_data 闭包中抽出, 供 get_data (窗口) 与
        get_frame_data (stateful 帧) 共用, 保证两路径特征选择完全一致。
        ✅20260822 - rdwang: 特征列选择改为数据集类的 feature_columns 策略
        (原类名白名单) — 新增数据集只需在数据集类里声明 FEATURE_COLUMNS,
        不再需要改这里。dataset 传入外层数据集实例 (不传 subj, 因 MixedDataset
        子集类名不可靠)。
        """
        fc = getattr(dataset, "feature_columns", None)
        if fc is None:
            raise AttributeError(
                f"Dataset not known: {dataset.__class__.__name__} (缺少 feature_columns 策略)")
        features = pd.DataFrame()
        all_features = subj.feature_table

        if "ACT" in modality:
            movement_features = all_features.filter(regex="_acc")[fc("ACT")]
            features = pd.concat([features, movement_features], axis=1)
        if "HRV" in modality:
            hrv_features = all_features.filter(regex="_hrv")[fc("HRV")]
            features = pd.concat([features, hrv_features], axis=1)
        if "RRV" in modality:
            rrv_features = all_features.filter(regex="RRV")[fc("RRV")]
            features = pd.concat([features, rrv_features], axis=1)
        if "EDR" in modality:
            edr_features = all_features.filter(regex="EDR")[fc("EDR")]
            features = pd.concat([features, edr_features], axis=1)

        if classification_type == "binary":
            ground_truth = subj.ground_truth["sleep"]
        else:
            ground_truth = subj.ground_truth
            ground_truth = ground_truth[classification_type]

        return features, ground_truth

    def get_data(
        self,
        dataset: Dataset,
        modality: list,
        scaler: StandardScaler = None,
        overlap=None,
        classification_type="binary",
        padding=False,
    ):
        """
        Returns and processes the data and brings them into the correct way to feed into the deep learning network
        :dataset: Dataset of tpcp class
        :modality: list of modalities to extract
        :scaler: Scaler to scale the respective data
        :overlap: overlap that is used in the sliding window method
        """
        # ---- 特征提取辅助函数 ----
        def _extract_subj_features(subj):
            features, ground_truth = self._extract_subj_features_raw(
                subj, dataset, modality, classification_type
            )
            return self.get_sequence_data(features, ground_truth,
                                          overlap=overlap, padding=padding)

        # ---- Pass 1: 统计总量 + fit scaler (不存全量数据) ----
        total_samples = 0
        if scaler is None:
            scaler = StandardScaler()
            for subj in dataset:
                x_mat, _ = _extract_subj_features(subj)
                total_samples += x_mat.shape[0]
                for chunk in batchify(x_mat):
                    scaler.partial_fit(chunk.reshape(-1, chunk.shape[-1]))
        else:
            for subj in dataset:
                x_mat, _ = _extract_subj_features(subj)
                total_samples += x_mat.shape[0]

        # ---- Pass 2: 逐被试 scale → 直接填入预分配 tensor (不 concat) ----
        n_features = len(scaler.mean_)
        x_tensor = torch.empty(total_samples, self.seq_len, n_features, dtype=torch.float32)
        y_tensor = torch.empty(total_samples, 1, dtype=torch.float32)

        cursor = 0
        for subj in dataset:
            x_mat, y_mat = _extract_subj_features(subj)
            x_scaled = scaler.transform(x_mat.reshape(-1, n_features)).reshape(x_mat.shape)
            n = x_scaled.shape[0]
            x_tensor[cursor:cursor + n] = torch.from_numpy(x_scaled.astype(np.float32))
            y_tensor[cursor:cursor + n, 0] = torch.from_numpy(y_mat.astype(np.float32))
            cursor += n

        return x_tensor, y_tensor, scaler

    def get_frame_data(self, dataset, scaler, modality, classification_type="binary"):
        """Stateful 路径: 逐被试原始帧 (n, F) + 标签 (n,) + subj_id。

        ✅2026-08-11 - rdwang: 有状态训练/推理使用逐帧数据 (不做滑窗)。
        返回 list of (x_s (n,F) float32, y_s (n,) float32, subj_id str)。

        ⚠️ scaler 必须复用 get_final_tensors 拟合的 (窗口/padded 数据) 训练集 scaler:
        若在帧上重拟合, 与 stateless 的缩放不一致, 会破坏 stateless↔stateful 可比性
        与 --load-weights 微调无状态 checkpoint 的兼容性。
        """
        frames = []
        for subj in dataset:
            features, ground_truth = self._extract_subj_features_raw(
                subj, dataset, modality, classification_type
            )
            feature_arr = np.asarray(features)
            gt_arr = np.asarray(ground_truth)
            n = feature_arr.shape[0]
            if n == 0:
                continue  # 空夜跳过 (正常被试 n >> seq_len)
            x_scaled = scaler.transform(feature_arr)  # (n, F), 与 get_data 同 scaler
            subj_id = subj.index["subj_id"][0]  # 与 test_to_list 一致的 id 取值 (含 MixedDataset 前缀)
            frames.append(
                (
                    torch.from_numpy(x_scaled.astype(np.float32)),
                    torch.from_numpy(gt_arr.astype(np.float32)),
                    str(subj_id),
                )
            )
        return frames

    def get_final_tensors(self, modality, train: Dataset, val: Dataset, test: Dataset, classification_type="binary", scaler=None):
        """
        Return final sequential tensors for each input modality
        This is the function that gets called in class LSTM_Optuna
        :param modality: input modality of datastream
        :param train: Training set
        :param val: Validation set
        :param test: Test set
        :param scaler: 预拟合的 StandardScaler (如 --load-weights 的配套 scaler); None 则用 train 拟合。
            ✅2026-08-13: 归一化参数是模型的一部分 — 加载 checkpoint 时必须用其配套 scaler,
            否则权重与归一化来自不同分布, 评估/微调结果静默失真
        """

        mod_set = {"HRV", "ACT", "RRV", "EDR"}
        if not all({mod}.issubset(mod_set) for mod in modality):
            raise AttributeError("modality MUST be list of either HRV, ACT, RRV, EDR")

        x_train, y_train, scaler = self.get_data(
            train,
            scaler=scaler,
            overlap=self.overlap,
            modality=modality,
            classification_type=classification_type,
            padding=True,
        )
        x_val, y_val, scaler = self.get_data(
            val,
            scaler=scaler,
            overlap=self.overlap,
            modality=modality,
            classification_type=classification_type,
            padding=True,
        )
        x_test = []
        y_test = []
        for subj in test:
            x_test_subj, y_test_subj, sc = self.get_data(
                subj,
                scaler=scaler,
                overlap=None,
                modality=modality,
                classification_type=classification_type,
                padding=True,
            )
            x_test, y_test = test_to_list(subj, x_test, y_test, x_test_subj, y_test_subj)
        return x_train, y_train, x_val, y_val, x_test, y_test, scaler
