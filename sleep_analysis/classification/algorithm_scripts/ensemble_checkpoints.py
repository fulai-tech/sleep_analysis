#!/usr/bin/env python3
"""多 checkpoint 模型级集成 — log 域加权投票。

对每个被试, 从 N 个 run 目录读取每模型的"集成后 logits" ({subj}_logits.csv,
由 LSTM.py test() 保存, 已含各模型内部的 temporal ensembling) 与真实标签
({subj}_labels.csv), 在 log 域加权平均 (乘性共识, 与 _ensemble_chunked 一致),
argmax 得最终标签, 用 dl_score 计算指标。

- 单模型投票 (N=1) = 该模型自身结果 → 自洽性验证
- 多模型 = 不同上下文窗口/训练随机性的互补集成 (deep ensemble)
- binary (C=1): logits 加权平均 → sigmoid (几何 odds 平均, 与 log 域一致)

用法:
    python ensemble_checkpoints.py --runs runA runB runC \\
        --classification 4stage --weights 1 1 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from sleep_analysis.classification.deep_learning.dl_scoring import dl_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="1+ 个 run 目录 (各自含 per_subject_predictions/{subj}_logits.csv + _labels.csv)")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="各模型权重 (默认均匀, 会自动归一化; 可按 val MCC 传入)")
    parser.add_argument("-c", "--classification", default="4stage",
                        choices=["binary", "3stage", "4stage", "5stage"])
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="投票后预测 CSV 输出目录 (默认 <第一个 run>/ensemble_predictions)")
    return parser.parse_args()


def _read_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(np.float64)


def main() -> None:
    args = parse_args()
    pred_dirs = [Path(r) / "per_subject_predictions" for r in args.runs]
    for d in pred_dirs:
        if not d.exists():
            print(f"[ERROR] {d} not found", flush=True)
            sys.exit(1)

    # 各目录 logits 文件的被试交集
    subj_sets = [{f.name[: -len("_logits.csv")] for f in d.glob("*_logits.csv")} for d in pred_dirs]
    subjects = sorted(set.intersection(*subj_sets))
    if not subjects:
        print("[ERROR] no common subjects across runs", flush=True)
        sys.exit(1)

    weights = args.weights if args.weights is not None else [1.0] * len(pred_dirs)
    if len(weights) != len(pred_dirs):
        print(f"[ERROR] --weights count {len(weights)} != runs count {len(pred_dirs)}", flush=True)
        sys.exit(1)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    print(f"Subjects: {len(subjects)}, models: {len(pred_dirs)}, weights: {weights.tolist()}")

    out_dir = args.output_dir or (pred_dirs[0].parent.parent / "ensemble_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    score_dict, pred_dict = {}, {}
    per_model_scores: dict[str, list] = {}  # 单模型对照指标 (投票前各模型自身)
    total_cm = None  # 全体被试累加的 confusion matrix
    cm_labels = None

    for subj in subjects:
        logits_list = []
        for d in pred_dirs:
            lf = d / f"{subj}_logits.csv"
            if not lf.exists():
                print(f"[ERROR] missing {lf}", flush=True)
                sys.exit(1)
            logits_list.append(_read_matrix(lf))
        n_rows = {l.shape[0] for l in logits_list}
        if len(n_rows) != 1:
            print(f"[ERROR] {subj}: inconsistent epoch counts across runs: {n_rows}", flush=True)
            sys.exit(1)
        n, C = logits_list[0].shape

        labels = _read_matrix(pred_dirs[0] / f"{subj}_labels.csv").ravel()
        if len(labels) != n:
            print(f"[ERROR] {subj}: labels {len(labels)} != logits rows {n}", flush=True)
            sys.exit(1)
        y_df = pd.DataFrame(labels, columns=["sleep_stage"])

        stack = np.stack(logits_list)  # (N, n, C)
        if C == 1:
            # binary: logits 加权平均 → sigmoid
            merged = (weights[:, None, None] * stack).sum(axis=0)  # (n, 1)
            y_pred = (1 / (1 + np.exp(-merged)) >= 0.5).astype(float).ravel()
            model_preds = [(1 / (1 + np.exp(-stack[i])) >= 0.5).astype(float).ravel()
                           for i in range(len(pred_dirs))]
        else:
            # multi: log 域加权 (log_softmax + clamp 防 -inf)
            mx = stack.max(axis=-1, keepdims=True)
            log_p = stack - mx - np.log(np.exp(stack - mx).sum(axis=-1, keepdims=True))
            log_p = np.clip(log_p, -100.0, None)
            merged = (weights[:, None, None] * log_p).sum(axis=0)  # (n, C)
            # argmax 前转回 float32 — 与 eval 内部 (float32 logits) 的数值域一致,
            # 单模型自洽验证 (N=1) 时逐位一致
            y_pred = np.argmax(merged.astype(np.float32), axis=1)
            model_preds = [np.argmax(stack[i], axis=1) for i in range(len(pred_dirs))]

        pred_dict[subj] = y_pred
        score_dict[subj] = dl_score(y_pred, y_df, args.classification, subject_id=subj)
        pd.DataFrame(y_pred).to_csv(out_dir / f"{subj}.csv")
        # 整体 confusion matrix 累加 (binary 标签域 [0,1], 其余 [0, C-1])
        cm_labels = [0, 1] if C == 1 else list(range(C))
        cm = sk_confusion_matrix(labels, y_pred, labels=cm_labels)
        total_cm = cm if total_cm is None else total_cm + cm
        for i in range(len(pred_dirs)):
            per_model_scores.setdefault(i, {})[subj] = dl_score(
                model_preds[i], y_df, args.classification, subject_id=f"{subj}"
            )

    def _summary(scores: dict) -> pd.DataFrame:
        df = pd.DataFrame(scores)
        numeric_cols = [c for c in df.index if c != "confusion_matrix"]
        return df.loc[numeric_cols].agg(["mean"], axis=1).T

    print("=" * 60)
    print("ENSEMBLE (weighted log-domain vote):")
    ens_summary = _summary(score_dict)
    print(ens_summary.to_string())
    for i in range(len(pred_dirs)):
        print(f"  model {i} alone ({Path(args.runs[i]).name}):")
        print(_summary(per_model_scores[i]).to_string())
    print(f"Predictions saved to: {out_dir}")
    if total_cm is not None:
        cm_df = pd.DataFrame(total_cm, index=cm_labels, columns=cm_labels)
        print("Confusion Matrix (counts, rows=true → cols=pred):")
        print(cm_df.to_string())
        pct = total_cm / total_cm.sum(axis=1, keepdims=True) * 100
        print("Confusion Matrix (%, rows=true → cols=pred):")
        print(pd.DataFrame(np.round(pct, 1), index=cm_labels, columns=cm_labels).to_string())


if __name__ == "__main__":
    main()
