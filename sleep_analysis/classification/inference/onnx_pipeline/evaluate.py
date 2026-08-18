#!/usr/bin/env python3
"""
测试集批量评估 (独立版, 无 sleep_analysis 依赖)。

与 inference.py 相同的预处理流程 (含 night_norm 整夜归一化)，
对测试集清单中的每个被试推理并与标注对比，输出汇总指标。

用法:
    python evaluate.py --subjects-file test_subjects.csv

    # 与训练时的 per_subject_metrics.csv 对比
    python evaluate.py --subjects-file test_subjects.csv --reference per_subject_metrics.csv

输入:
    subjects-file : CSV, 列: dataset, subject_id, feature_path, ground_truth_path
    model.onnx    : ONNX 模型
    scaler_params.json : scaler 参数 + 推理配置

输出:
    {output_dir}/summary.json          汇总指标
    {output_dir}/per_subject_metrics.csv  逐被试指标
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SCALER_JSON = SCRIPT_DIR / "scaler_params.json"
MODEL_ONNX = SCRIPT_DIR / "model.onnx"

# onnxruntime 日志级别: 3 = warning 及以上 (默认), 显式设置保证行为确定
import onnxruntime as ort
ort.set_default_logger_severity(3)

# 与训练时一致的 12 列特征选择 (顺序重要)
_HRV_COLUMNS = [
    "_hrv_median_nni",
    "_hrv_ratio_sd2_sd1",
    "_hrv_median_nni",  # 训练时重复选取, 保持一致
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


# ---------------------------------------------------------------------------
# 工具函数 (与 inference.py 一致)
# ---------------------------------------------------------------------------

def select_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """从全量特征表选择训练时使用的 12 列。"""
    parts = []
    hrv = feature_df.filter(regex="_hrv")
    cols = [c for c in _HRV_COLUMNS if c in hrv.columns]
    parts.append(hrv[cols])
    rrv = feature_df.filter(regex="RRV")
    cols = [c for c in _RRV_COLUMNS if c in rrv.columns]
    parts.append(rrv[cols])
    return pd.concat(parts, axis=1)


def build_sequences(features: np.ndarray, seq_len: int, causal: bool) -> np.ndarray:
    """(n_epochs, n_features) → (n_epochs, seq_len, n_features)，均值居中 padding。"""
    n_epochs, n_features = features.shape
    if causal:
        pad_left, pad_right = seq_len - 1, 0
    else:
        pad_left = seq_len // 2
        pad_right = seq_len // 2
    mean_vals = features.mean(axis=0, keepdims=True)
    padded = np.concatenate([
        np.tile(mean_vals, (pad_left, 1)),
        features,
        np.tile(mean_vals, (pad_right, 1)),
    ], axis=0)
    n_windows = padded.shape[0] - seq_len + 1
    x = np.empty((n_windows, seq_len, n_features), dtype=features.dtype)
    for i in range(n_windows):
        x[i] = padded[i : i + seq_len]
    return x


def apply_scaler(x: np.ndarray, mean_: np.ndarray, scale_: np.ndarray) -> np.ndarray:
    mean_ = mean_.astype(x.dtype)
    scale_ = scale_.astype(x.dtype)
    return (x - mean_) / (scale_ + np.finfo(x.dtype).eps)


def apply_night_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """整夜归一化 (第二层 norm 的模型外实现, 统计量 = 被试整夜数据)。"""
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), ddof=1, keepdims=True) + eps
    return (x - mean) / std


# ---------------------------------------------------------------------------
# 评分 (与训练脚本 dl_scoring.py 一致的指标)
# ---------------------------------------------------------------------------

def multiclass_specificity(y_true, y_pred, labels):
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    weights, specificities = [], []
    n_total = len(y_true)
    for l, label in enumerate(labels):
        tp = conf_mat[l][l]
        tn = np.sum(conf_mat) - np.sum(conf_mat[:, l]) - np.sum(conf_mat[l, :]) + conf_mat[l][l]
        fp = np.sum(conf_mat[l, :]) - conf_mat[l][l]
        fn = np.sum(conf_mat[:, l]) - conf_mat[l][l]
        weights.append(np.sum(np.asarray(y_true) == label) / n_total)
        specificities.append(np.nan_to_num(tn / (tn + fp)))
    return float(np.sum(np.array(specificities) * np.array(weights)))


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, classification_type: str = "4stage") -> dict:
    """逐被试指标 (与训练时 dl_score 一致)。"""
    import sklearn.metrics as sk_metrics
    from sklearn.metrics import confusion_matrix, matthews_corrcoef

    labels = {
        "binary": [0, 1], "3stage": [0, 1, 2],
        "4stage": [0, 1, 2, 3], "5stage": [0, 1, 2, 3, 4],
    }[classification_type]

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": sk_metrics.accuracy_score(y_true, y_pred),
        "precision": sk_metrics.precision_score(y_true, y_pred, zero_division=0, average="weighted"),
        "recall": sk_metrics.recall_score(y_true, y_pred, zero_division=0, average="weighted"),
        "f1": sk_metrics.f1_score(y_true, y_pred, zero_division=0, average="weighted"),
        "kappa": sk_metrics.cohen_kappa_score(y_true, y_pred),
        "specificity": multiclass_specificity(y_true, y_pred, labels=labels),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": conf_mat.tolist(),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="测试集批量评估 (独立版)")
    parser.add_argument("--subjects-file", type=str, required=True,
                        help="测试集清单 CSV (dataset,subject_id,feature_path,ground_truth_path)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: 本目录 evaluate_results/)")
    parser.add_argument("--reference", type=str, default=None,
                        help="训练时的 per_subject_metrics.csv, 用于对比")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="最多评估多少被试 (调试用)")
    args = parser.parse_args()

    # 1. 配置
    with open(SCALER_JSON) as f:
        scaler_cfg = json.load(f)
    seq_len = scaler_cfg["seq_len"]
    causal = scaler_cfg.get("causal", False)
    classification = scaler_cfg["classification_type"]
    night_norm = scaler_cfg.get("night_norm", False)
    mean_ = np.array(scaler_cfg["mean_"])
    scale_ = np.array(scaler_cfg["scale_"])
    print(f"配置: seq_len={seq_len}, causal={causal}, classification={classification}, "
          f"night_norm={night_norm}")

    # 2. 测试集清单 (subject_id 保持字符串, 保留前导零)
    subjects = pd.read_csv(args.subjects_file, dtype={"subject_id": str})
    if args.max_subjects:
        subjects = subjects.head(args.max_subjects)
    print(f"测试集: {len(subjects)} 被试")

    # 3. ONNX 会话
    session = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 4. 逐被试推理 + 评分
    per_subject = {}
    for i, row in subjects.iterrows():
        label = f"{row['dataset']}@{row['subject_id']}"
        try:
            feat = pd.read_csv(row["feature_path"], index_col=0)  # 与训练读法一致
            selected = select_features(feat)
            features = selected.values.astype(np.float32)
            x = build_sequences(features, seq_len=seq_len, causal=causal)
            x = apply_scaler(x, mean_, scale_)
            if night_norm:
                x = apply_night_norm(x)
            out = session.run(None, {input_name: x})[0]
            y_pred = np.argmax(out, axis=1).astype(int)

            gt = pd.read_csv(row["ground_truth_path"])
            y_true = gt[classification].values[: len(y_pred)].astype(int)

            per_subject[label] = compute_metrics(y_pred, y_true, classification)
        except Exception as e:
            print(f"  [WARN] {label} 失败: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  进度: {i + 1}/{len(subjects)}")

    # 5. 汇总 (per-subject mean, 与训练脚本一致)
    subj_df = pd.DataFrame(per_subject)
    numeric = [c for c in subj_df.index if c != "confusion_matrix"]
    means = subj_df.loc[numeric].agg(["mean"], axis=1).T
    summary = {k: float(means[k].iloc[0]) for k in means.columns}

    num_classes = len({"binary": [0, 1], "3stage": [0, 1, 2], "4stage": [0, 1, 2, 3], "5stage": [0, 1, 2, 3, 4]}[classification])
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for label in subj_df.columns:
        conf_matrix += np.array(subj_df.loc["confusion_matrix", label])
    summary["confusion_matrix"] = conf_matrix.tolist()

    print(f"\n{'='*60}")
    print(f"评估结果 ({len(per_subject)} 被试, per-subject mean)")
    print(f"{'='*60}")
    for k in numeric:
        print(f"  {k:10s}: {summary[k]:.4f}")

    # 6. 与参考结果对比 (参考文件列名需与本清单的 label 一致, 如 subject@0036)
    comparison = None
    if args.reference:
        ref = pd.read_csv(args.reference, index_col=0)
        labels = set(f"{row['dataset']}@{row['subject_id']}" for _, row in subjects.iterrows())
        ref_cols = [c for c in ref.columns if str(c) in labels]
        ref_sub = ref[ref_cols].loc[numeric].apply(pd.to_numeric, errors="coerce")
        ref_mean = {k: float(ref_sub.loc[k].mean()) for k in ref_sub.index}
        diff = {k: round(summary[k] - ref_mean[k], 6) for k in numeric}
        comparison = {"reference": ref_mean, "evaluate": summary, "diff": diff}
        print(f"\n与参考结果对比 (evaluate - reference):")
        for k in numeric:
            d = diff[k]
            mark = "✅" if abs(d) < 0.01 else "⚠️"
            print(f"  {k:10s}: ref={ref_mean[k]:.4f}  evaluate={summary[k]:.4f}  "
                  f"diff={d:+.4f} {mark}")

    # 7. 保存
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "evaluate_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    subj_df.to_csv(output_dir / "per_subject_metrics.csv")
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "n_subjects": len(per_subject),
            "summary": summary,
            "training_comparison": comparison,
        }, f, indent=2, default=str)
    print(f"\n结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
