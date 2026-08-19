#!/usr/bin/env python3
"""4stage 模型评测: (A) 测试集类别占比 (B) 原始 4 分类评测 (C) 折叠 Wake/Sleep 二分类评测。

4stage 编码: 0=Wake, 1=Light, 2=Deep, 3=REM
折叠: 预测与真实都按 {0 → Wake(0), 1/2/3 → Sleep(1)} 映射后池化计算。
混淆矩阵均输出 数字版 和 百分比版(行归一化, 对角线=召回率)。

用法:
    python tools/eval_binary_fold.py                                  # 默认 run
    python tools/eval_binary_fold.py --run-dir exports_our/<run>      # 指定 run
    python tools/eval_binary_fold.py --run-dir <abs> --mesa-path <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as sk

_SA = Path("/home/rdwang/repo/SleepStaging/third_party/sleep_analysis")
STAGE_NAMES = ["wake", "light", "deep", "rem"]


def _parse_args():
    ap = argparse.ArgumentParser(description="4stage 模型评测 (原始 4 分类 + 折叠二分类)")
    ap.add_argument("--run-dir", type=str, default="exports_our/2026-08-19_114015",
                    help="训练 run 目录 (默认 2026-08-19_114015)")
    ap.add_argument("--mesa-path", type=str,
                    default="/srv/shared/psgdata/processed_data_causal_20260806/mesa_processed",
                    help="MESA processed 数据目录 (取 actigraph_data_clean 标签)")
    return ap.parse_args()


def _specificity(y_true, y_pred):
    """二分类特异度 TN/(TN+FP)。"""
    tn, fp, fn, tp = sk.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _weighted_specificity(y_true, y_pred, labels):
    """多分类加权特异度: 每类 specificity 按该类样本数加权。"""
    cm = sk.confusion_matrix(y_true, y_pred, labels=labels)
    total = 0.0
    weight_sum = 0.0
    for i, _ in enumerate(labels):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        w = cm[i].sum()
        total += sp * w
        weight_sum += w
    return total / weight_sum if weight_sum > 0 else 0.0


def _print_cm(title, cm, labels):
    """打印数字版 + 百分比版(行归一化)混淆矩阵。"""
    print(f"\n  {title} (数字版, rows=真实 → cols=预测):")
    header = "        " + "".join(f"{n:>8}" for n in labels)
    print("  " + header)
    for i, name in enumerate(labels):
        print(f"  {name:>8}" + "".join(f"{cm[i, j]:>8.0f}" for j in range(len(labels))))
    print(f"\n  {title} (百分比版, 行归一化, 对角线=召回率):")
    print("  " + header)
    for i, name in enumerate(labels):
        row = cm[i] / cm[i].sum() * 100 if cm[i].sum() > 0 else np.zeros(len(labels))
        print(f"  {name:>8}" + "".join(f"{row[j]:>7.1f}" for j in range(len(labels))))


def main():
    args = _parse_args()
    RUN = Path(args.run_dir) if Path(args.run_dir).is_absolute() else _SA / args.run_dir
    MESA = Path(args.mesa_path)

    cfg = json.load(open(RUN / "config.json"))
    print(f"run: {RUN.name} | {cfg['classification']} | dataset: {cfg['dataset']} | "
          f"modality: {cfg['modality']} | lookahead: {cfg.get('lookahead_min')}min")

    y_true_all, y_pred_all = [], []
    n_subj = 0
    for csv_path in sorted((RUN / "per_subject_predictions").glob("*.csv")):
        subj = csv_path.stem
        gt_path = MESA / "actigraph_data_clean" / f"actigraph_data_clean{subj}.csv"
        if not gt_path.exists():
            print(f"  [skip] {subj}: 无 ground truth")
            continue
        pred = pd.read_csv(csv_path, index_col=0).iloc[:, 0].to_numpy().astype(int)
        true = pd.read_csv(gt_path)["4stage"].to_numpy().astype(int)
        n = min(len(pred), len(true))
        y_true_all.append(true[:n])
        y_pred_all.append(pred[:n])
        n_subj += 1

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    n_total = len(y_true)
    print(f"被试数: {n_subj} | 总 epoch: {n_total}")

    # ================= A. 测试集类别占比 =================
    print("\n===== A. 测试集 4stage 类别占比 (真实标签) =====")
    counts = np.bincount(y_true, minlength=4)
    for i, name in enumerate(STAGE_NAMES):
        print(f"  {name:>5}: {counts[i]:>7} ({counts[i] / n_total * 100:5.1f}%)")
    print(f"  sleep 合计 (1/2/3): {counts[1:].sum():>7} ({(counts[1:].sum() / n_total * 100):5.1f}%)")

    # ================= B. 原始 4 分类评测 =================
    print("\n===== B. 原始 4 分类评测 =====")
    cm4 = sk.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    m4 = {
        "accuracy": sk.accuracy_score(y_true, y_pred),
        "precision": sk.precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": sk.recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": sk.f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": sk.cohen_kappa_score(y_true, y_pred),
        "specificity": _weighted_specificity(y_true, y_pred, [0, 1, 2, 3]),
        "mcc": sk.matthews_corrcoef(y_true, y_pred),
    }
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in m4.items()))
    print("\n  逐类召回/精确率:")
    for i, name in enumerate(STAGE_NAMES):
        rec = cm4[i, i] / cm4[i].sum() * 100 if cm4[i].sum() else 0.0
        prec = cm4[:, i].sum() and cm4[i, i] / cm4[:, i].sum() * 100
        print(f"    {name:>5}: 召回 {rec:5.1f}% | 精确率 {prec:5.1f}%")
    _print_cm("4stage 混淆矩阵", cm4, STAGE_NAMES)

    # ================= C. 折叠二分类评测 =================
    print("\n===== C. 折叠二分类 (4stage 预测 → Wake/Sleep) =====")
    bin_true = (y_true > 0).astype(int)
    bin_pred = (y_pred > 0).astype(int)
    m2 = {
        "accuracy": sk.accuracy_score(bin_true, bin_pred),
        "precision": sk.precision_score(bin_true, bin_pred, zero_division=0),
        "recall": sk.recall_score(bin_true, bin_pred, zero_division=0),
        "f1": sk.f1_score(bin_true, bin_pred, zero_division=0),
        "kappa": sk.cohen_kappa_score(bin_true, bin_pred),
        "specificity": _specificity(bin_true, bin_pred),
        "mcc": sk.matthews_corrcoef(bin_true, bin_pred),
    }
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in m2.items()))
    cm2 = sk.confusion_matrix(bin_true, bin_pred)
    _print_cm("二分类混淆矩阵", cm2, ["wake", "sleep"])

    print("\n===== 附加: 折叠误差的来源分解 =====")
    for i, name in enumerate(STAGE_NAMES):
        m = y_true == i
        if m.sum() == 0:
            continue
        pred_wake = (y_pred[m] == 0).mean() * 100
        print(f"  真实 {name:>5} ({m.sum():>6}): 折叠后报 wake {pred_wake:5.1f}% | 报 sleep {100 - pred_wake:5.1f}%")


if __name__ == "__main__":
    main()
