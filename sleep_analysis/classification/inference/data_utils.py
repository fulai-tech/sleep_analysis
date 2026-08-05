"""
框架无关的数据工具：特征加载、序列构建、scaler 标准化、评分对比。
不依赖 PyTorch / ONNX Runtime，仅需 numpy, pandas, sklearn。

用于:
  - engine_torch.py  (PyTorch 推理)
  - engine_onnx.py   (ONNX 推理)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 训练产出的 artifact 加载
# ---------------------------------------------------------------------------

def load_run_config(run_dir: Path) -> dict:
    """加载训练运行目录中的 config.json。"""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(config_path) as f:
        return json.load(f)


def load_scaler(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载训练时保存的 StandardScaler 参数。

    Returns
    -------
    mean_ : np.ndarray  shape (n_features,)
    scale_ : np.ndarray shape (n_features,)
    """
    scaler_path = run_dir / "checkpoints" / "scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"scaler.json not found at {scaler_path}. "
            "请确保训练脚本已保存第一层 StandardScaler。"
        )
    with open(scaler_path) as f:
        data = json.load(f)
    return np.array(data["mean_"]), np.array(data["scale_"])


# ---------------------------------------------------------------------------
# 特征选择 — 必须与训练时 _extract_subj_features() 完全一致
# ---------------------------------------------------------------------------

# 各模态在特征大表中所选的列名
_HRV_COLUMNS = [
    "_hrv_median_nni",
    "_hrv_ratio_sd2_sd1",
    "_hrv_median_nni",  # 训练代码中该列选了两次，此处保持一致
    "_hrv_vlf",
    "_hrv_lf",
    "_hrv_hf",
    "_hrv_lf_hf_ratio",
    "_hrv_total_power",
]

_RRV_COLUMNS = [
    "150_RRV_MedianBB",
    "150_RRV_LF",
    "270_RRV_MCVBB",
    "150_RRV_CVBB",
]

_EDR_COLUMNS = [
    "150_EDR_MeanBB",
    "150_EDR_LF",
    "150_EDR_HF",
    "150_EDR_LFHF",
]

_ACT_COLUMNS = ["_acc_mean_1"]


def select_features(feature_df: pd.DataFrame, modality: List[str]) -> pd.DataFrame:
    """
    从特征大表中选择与训练时一致的列。

    Parameters
    ----------
    feature_df : pd.DataFrame
        完整的特征表 (features_combined{ID}.csv)。
    modality : list[str]
        模态列表，如 ["HRV", "RRV"]。

    Returns
    -------
    selected : pd.DataFrame
        按训练顺序排列的子特征表。
    """
    parts = []
    # 遍历 modality，顺序必须与训练时一致
    # 训练时的顺序: ACT → HRV → RRV → EDR
    for mod in ("ACT", "HRV", "RRV", "EDR"):
        if mod not in modality:
            continue
        if mod == "ACT":
            cols = [c for c in _ACT_COLUMNS if c in feature_df.columns]
            parts.append(feature_df[cols])
        elif mod == "HRV":
            # filter(regex="_hrv") then select specific columns
            hrv = feature_df.filter(regex="_hrv")
            cols = [c for c in _HRV_COLUMNS if c in hrv.columns]
            parts.append(hrv[cols])
        elif mod == "RRV":
            rrv = feature_df.filter(regex="RRV")
            cols = [c for c in _RRV_COLUMNS if c in rrv.columns]
            parts.append(rrv[cols])
        elif mod == "EDR":
            edr = feature_df.filter(regex="EDR")
            cols = [c for c in _EDR_COLUMNS if c in edr.columns]
            parts.append(edr[cols])
    return pd.concat(parts, axis=1)


def select_features_for_shhs(feature_df: pd.DataFrame, modality: List[str]) -> pd.DataFrame:
    """
    SHHS 数据集特征选择 — 与训练时 ShhsDataset._extract_subj_features 一致。

    注意: SHHS 的 HRV 列名与 MESA 相同 (带前导下划线 `_hrv_*`)，
    因此直接委托 select_features 即可。此前曾错误地对列名做 lstrip("_")
    (见 review)，导致 SHHS HRV 特征永远选不到。
    """
    return select_features(feature_df, modality)


# ---------------------------------------------------------------------------
# 序列构建 — 与 data_peparation.py::DataPreparation.get_sequence_data 一致
# ---------------------------------------------------------------------------

def build_sequences(
    features: np.ndarray,
    seq_len: int = 21,
    causal: bool = False,
) -> np.ndarray:
    """
    将 (n_epochs, n_features) 的特征矩阵转为滑动窗口序列。

    与 data_peparation.py 的 padding=True + sliding_window 逻辑一致：
      - causal=False: 居中 padding，每端各垫 seq_len/2 个均值
      - causal=True:  仅左侧 padding (实时分期)

    Parameters
    ----------
    features : np.ndarray  shape (n_epochs, n_features)
    seq_len : int  窗口长度 (epoch 数)
    causal : bool  是否实时模式 (仅历史 padding)

    Returns
    -------
    x : np.ndarray  shape (n_epochs, seq_len, n_features)
    """
    n_epochs, n_features = features.shape

    if causal:
        # 仅在左侧垫 (seq_len - 1) 个均值
        pad_left, pad_right = seq_len - 1, 0
    else:
        # 居中：左右各垫一半
        pad_left = seq_len // 2
        pad_right = seq_len // 2

    # 使用均值填充
    mean_vals = features.mean(axis=0, keepdims=True)
    pad_left_arr = np.tile(mean_vals, (pad_left, 1))
    pad_right_arr = np.tile(mean_vals, (pad_right, 1))
    padded = np.concatenate([pad_left_arr, features, pad_right_arr], axis=0)

    # 滑动窗口：与 biopsykit sliding_window(overlap_percent=None, window_samples=seq_len) 一致
    n_windows = padded.shape[0] - seq_len + 1
    x = np.empty((n_windows, seq_len, n_features), dtype=features.dtype)
    for i in range(n_windows):
        x[i] = padded[i : i + seq_len]

    return x


# ---------------------------------------------------------------------------
# Scaler 应用 — 第一层标准化
# ---------------------------------------------------------------------------

def apply_scaler(
    x: np.ndarray,
    mean_: np.ndarray,
    scale_: np.ndarray,
) -> np.ndarray:
    """
    应用 StandardScaler: (x - mean) / scale。
    保持输入 dtype 不变 (避免 float32 → float64 上溢)。

    Parameters
    ----------
    x : np.ndarray  shape (n_epochs, seq_len, n_features)
    mean_ : np.ndarray  shape (n_features,)
    scale_ : np.ndarray  shape (n_features,)

    Returns
    -------
    x_scaled : np.ndarray  same shape and dtype as x
    """
    x_dtype = x.dtype
    mean_ = mean_.astype(x_dtype)
    scale_ = scale_.astype(x_dtype)
    return (x - mean_) / (scale_ + np.finfo(x_dtype).eps)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_features(
    subject_id: str,
    processed_path: Path,
) -> pd.DataFrame:
    """
    加载指定被试的完整特征表。

    Parameters
    ----------
    subject_id : str  被试编号，如 "0001"
    processed_path : Path  MESA 预处理目录 (mesa_processed/)

    Returns
    -------
    feature_df : pd.DataFrame
    """
    feat_path = processed_path / "features_full_combined" / f"features_combined{subject_id}.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feat_path}. "
            "请先运行 preprocessing 管线 (preprocess_subset.py) 或检查被试编号。"
        )
    return pd.read_csv(feat_path, index_col=0)


def load_features_from_path(feature_path: Path) -> pd.DataFrame:
    """从任意路径加载特征 CSV。"""
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    return pd.read_csv(feature_path, index_col=0)


def resolve_data_paths(
    subject_id: str,
    processed_path: Path,
) -> Tuple[Path, Optional[Path]]:
    """
    根据被试编号和预处理目录解析特征文件与标注文件路径。

    不同数据集目录结构不同:
      - MESA: 标注在 actigraph_data_clean/actigraph_data_clean{id}.csv
      - SHHS: 标注在 sleep_stages/sleep_stages{id}.csv (无体动记录)
      - 标注缺失时返回 None (仅推理, 不评估)

    Returns
    -------
    (feature_path, ground_truth_path)
    """
    feat_path = processed_path / "features_full_combined" / f"features_combined{subject_id}.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feat_path}. "
            "请先运行预处理管线或检查被试编号。"
        )

    # MESA
    gt_path = processed_path / "actigraph_data_clean" / f"actigraph_data_clean{subject_id}.csv"
    # SHHS
    if not gt_path.exists():
        gt_path = processed_path / "sleep_stages" / f"sleep_stages{subject_id}.csv"
    if not gt_path.exists():
        gt_path = None

    return feat_path, gt_path


def load_ground_truth(
    subject_id: str,
    processed_path: Path,
) -> pd.DataFrame:
    """
    加载指定被试的睡眠分期标注 (MESA / SHHS 自适应)。

    Parameters
    ----------
    subject_id : str  被试编号
    processed_path : Path

    Returns
    -------
    gt : pd.DataFrame  包含列: sleep, 5stage, 4stage, 3stage
    """
    _, gt_path = resolve_data_paths(subject_id, processed_path)
    if gt_path is None:
        raise FileNotFoundError(
            f"Ground truth not found for {subject_id} in {processed_path} "
            "(searched actigraph_data_clean/ and sleep_stages/)"
        )
    df = pd.read_csv(gt_path)
    cols = [c for c in ["line", "sleep", "5stage", "4stage", "3stage"] if c in df.columns]
    return df[cols]


def load_ground_truth_from_path(
    gt_path: Path,
    classification_type: str = "4stage",
) -> np.ndarray:
    """
    从任意路径加载标注，返回指定分类方案的标签数组。

    Parameters
    ----------
    gt_path : Path
    classification_type : str

    Returns
    -------
    y_true : np.ndarray  shape (n_epochs,)
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    df = pd.read_csv(gt_path)

    # 尝试找到对应的列
    if classification_type in df.columns:
        return df[classification_type].values.astype(int)
    # 兼容 sleep_stage 列名
    if "sleep_stage" in df.columns:
        return df["sleep_stage"].values.astype(int)
    # fallback: 第一列
    return df.iloc[:, 0].values.astype(int)


# ---------------------------------------------------------------------------
# 评分 — 与 dl_scoring.py 一致
# ---------------------------------------------------------------------------

def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    classification_type: str = "4stage",
) -> dict:
    """
    计算分类指标。

    Parameters
    ----------
    y_pred : np.ndarray  shape (n_epochs,)  预测标签 (int)
    y_true : np.ndarray  shape (n_epochs,)  真实标签 (int)
    classification_type : str  "binary" | "3stage" | "4stage" | "5stage"

    Returns
    -------
    metrics : dict
    """
    if classification_type == "binary":
        labels = [0, 1]
    elif classification_type == "3stage":
        labels = [0, 1, 2]
    elif classification_type == "4stage":
        labels = [0, 1, 2, 3]
    elif classification_type == "5stage":
        labels = [0, 1, 2, 3, 4]
    else:
        raise ValueError(f"Unknown classification type: {classification_type}")

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": sk_metrics.accuracy_score(y_true, y_pred),
        "precision": sk_metrics.precision_score(y_true, y_pred, zero_division=0, average="weighted"),
        "recall": sk_metrics.recall_score(y_true, y_pred, zero_division=0, average="weighted"),
        "f1": sk_metrics.f1_score(y_true, y_pred, zero_division=0, average="weighted"),
        "kappa": sk_metrics.cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": conf_mat.tolist(),
    }


def _sanitize_prediction(pred: np.ndarray, classification_type: str) -> np.ndarray:
    """将模型原始输出转换为整数标签。"""
    if classification_type == "binary":
        # sigmoid → threshold
        return (1 / (1 + np.exp(-pred)) >= 0.5).astype(int).flatten()
    else:
        # softmax / argmax
        if pred.ndim == 2 and pred.shape[1] > 1:
            return np.argmax(pred, axis=1)
        return pred.flatten().astype(int)
