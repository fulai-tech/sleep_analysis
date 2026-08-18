#!/usr/bin/env python3
"""
测试集批量评估 — 复现训练代码的测试 pipeline，与训练时保存的结果对比。

用法:
    # 自动从 run 目录的 per_subject_predictions/ 生成测试集清单并评估
    python evaluate.py --run-dir exports_our/.../2026-08-03_202132 --backend onnx --night-norm

    # 使用现成的清单文件 (CSV: dataset,subject_id,feature_path,ground_truth_path)
    python evaluate.py --run-dir exports_our/.../2026-08-03_202132 \
        --subjects-file test_subjects.csv --backend torch --night-norm

    # 小批量调试
    python evaluate.py --run-dir ... --max-subjects 20

输出:
    {run_dir}/evaluate_results/
        summary.json        # 汇总指标 (per-subject mean) + 与训练 results.json 对比
        per_subject_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    d = _SCRIPT_DIR
    for _ in range(8):
        if (d / "study_data.json").exists():
            return d
        d = d.parent
    raise FileNotFoundError("Cannot find project root with study_data.json")


_PROJECT_ROOT = _find_project_root()


def _load_study_config():
    with open(_PROJECT_ROOT / "study_data.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 测试集清单生成 — 从训练 run 的 per_subject_predictions/ 还原
# ---------------------------------------------------------------------------

# 数据集前缀 → processed 目录键 (study_data.json)
_DATASET_KEYS = {
    "mesasleep": "processed_mesa_path",
    "shhs1": "shhs1_processed_path",
    "shhs2": "shhs2_processed_path",
}


def generate_subjects_list(run_dir: Path, study_cfg: dict) -> pd.DataFrame:
    """
    从训练 run 目录的 per_subject_predictions/ 提取测试集被试清单。
    文件名格式: "{dataset}@{subject_id}.csv" (如 mesasleep@0036.csv, shhs1@200012.csv)

    Returns
    -------
    DataFrame: dataset, subject_id, feature_path, ground_truth_path
    """
    pred_dir = run_dir / "per_subject_predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"per_subject_predictions/ not found in {run_dir}")

    rows = []
    for f in sorted(pred_dir.glob("*.csv")):
        name = f.stem
        if "@" not in name:
            continue
        dataset, subject_id = name.split("@", 1)
        key = _DATASET_KEYS.get(dataset)
        if key is None:
            print(f"  [WARN] 未知数据集前缀: {dataset}, 跳过 {name}")
            continue
        processed = Path(study_cfg[key])

        feat_path = processed / "features_full_combined" / f"features_combined{subject_id}.csv"
        gt_candidates = [
            processed / "actigraph_data_clean" / f"actigraph_data_clean{subject_id}.csv",  # MESA
            processed / "sleep_stages" / f"sleep_stages{subject_id}.csv",                  # SHHS
        ]
        gt_path = next((p for p in gt_candidates if p.exists()), None)

        rows.append({
            "dataset": dataset,
            "subject_id": subject_id,
            "feature_path": str(feat_path),
            "ground_truth_path": str(gt_path) if gt_path else "",
        })

    df = pd.DataFrame(rows)
    missing_feat = (~df["feature_path"].map(lambda p: Path(p).exists())).sum()
    missing_gt = (df["ground_truth_path"] == "").sum()
    print(f"测试集清单: {len(df)} 被试 (feature 缺失 {missing_feat}, 标注缺失 {missing_gt})")
    return df


# ---------------------------------------------------------------------------
# 推理 + 评分 (逐被试)
# ---------------------------------------------------------------------------

def evaluate_one(
    engine,
    feature_path: Path,
    ground_truth_path: Path,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """单被试推理 + 评分。"""
    y_pred, y_true = engine.predict_from_files(feature_path, ground_truth_path, label)
    from sleep_analysis.classification.inference.data_utils import compute_metrics
    metrics = compute_metrics(y_pred, y_true, engine.classification_type)
    return y_pred, y_true, metrics


def build_engine(run_dir: Path, backend: str, night_norm: Optional[bool]):
    if backend == "torch":
        from sleep_analysis.classification.inference.engine_torch import TorchInferenceEngine
        return TorchInferenceEngine(run_dir, night_norm=night_norm)
    elif backend == "onnx":
        from sleep_analysis.classification.inference.engine_onnx import OnnxInferenceEngine
        return OnnxInferenceEngine(run_dir, night_norm=night_norm)
    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

_METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "kappa", "specificity", "mcc"]


def summarize(subject_results: pd.DataFrame) -> dict:
    """与训练脚本一致的汇总方式: 逐被试指标求 mean。"""
    numeric_cols = [c for c in subject_results.index if c != "confusion_matrix"]
    means = subject_results.loc[numeric_cols].agg(["mean"], axis=1).T
    return {k: float(means[k].iloc[0]) for k in _METRIC_KEYS if k in means.columns}


def compare_with_training(summary: dict, run_dir: Path, dataset: Optional[str] = None) -> Optional[dict]:
    """
    与训练时保存的结果对比。

    无 --dataset 时用 results/results.json 的全体 mean;
    指定 --dataset (如 mesasleep) 时, 从 per_subject_metrics.csv 筛选该前缀的
    被试重新计算 mean, 得到该子集的训练参考指标。
    """
    if dataset is None:
        results_path = run_dir / "results" / "results.json"
        if not results_path.exists():
            return None
        with open(results_path) as f:
            train_res = json.load(f)
        train_mean = {k: v["mean"] for k, v in train_res["mean"].items()}
    else:
        per_subj_path = run_dir / "results" / "per_subject_metrics.csv"
        if not per_subj_path.exists():
            return None
        df = pd.read_csv(per_subj_path, index_col=0)
        cols = [c for c in df.columns if str(c).startswith(f"{dataset}@")]
        if not cols:
            print(f"  [WARN] per_subject_metrics.csv 中未找到 {dataset}@ 前缀的被试")
            return None
        sub = df[cols]
        # 数值转 float (confusion_matrix 行是字符串, 会让整列变成 object dtype)
        numeric = [c for c in sub.index if c != "confusion_matrix"]
        sub_numeric = sub.loc[numeric].apply(pd.to_numeric, errors="coerce")
        train_mean = {k: float(sub_numeric.loc[k].mean()) for k in sub_numeric.index}

    diff = {}
    for k in _METRIC_KEYS:
        if k in train_mean and k in summary:
            diff[k] = round(summary[k] - train_mean[k], 6)
    return {
        "training": {k: train_mean[k] for k in _METRIC_KEYS if k in train_mean},
        "evaluate": summary,
        "diff": diff,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="测试集批量评估 (复现训练测试 pipeline)")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="训练运行目录 (2026-08-03_202132)")
    parser.add_argument("--backend", type=str, default="onnx", choices=["torch", "onnx"],
                        help="推理后端 (default: onnx)")
    parser.add_argument("--night-norm", action="store_true",
                        help="启用模型外整夜归一化 (旧模型需要)")
    parser.add_argument("--subjects-file", type=str, default=None,
                        help="测试集清单 CSV (默认从 run_dir/per_subject_predictions 生成)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="只评估指定数据集 (如 mesasleep / shhs1 / shhs2)")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="最多评估多少被试 (调试用)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: {run_dir}/evaluate_results)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)

    study_cfg = _load_study_config()

    # 1. 测试集清单
    if args.subjects_file:
        subjects = pd.read_csv(args.subjects_file)
        print(f"加载测试集清单: {len(subjects)} 被试")
    else:
        subjects = generate_subjects_list(run_dir, study_cfg)

    if args.dataset:
        subjects = subjects[subjects["dataset"] == args.dataset]
        print(f"只评估数据集 {args.dataset}: {len(subjects)} 被试")

    if args.max_subjects:
        subjects = subjects.head(args.max_subjects)
        print(f"限制评估前 {len(subjects)} 个被试")

    # 过滤缺失文件
    subjects = subjects[subjects["feature_path"].map(lambda p: Path(p).exists())]
    valid_gt = subjects[subjects["ground_truth_path"] != ""]
    print(f"有效被试: {len(subjects)} (有标注: {len(valid_gt)})")

    # 2. 引擎
    engine = build_engine(run_dir, args.backend,
                          True if args.night_norm else None)

    # 3. 逐被试评估
    print(f"\n开始评估 {len(subjects)} 个被试 ...")
    all_preds, all_trues, per_subject = [], [], {}
    for i, row in subjects.iterrows():
        label = f"{row['dataset']}@{row['subject_id']}"
        if i % 200 == 0:
            print(f"  [{i}/{len(subjects)}]")
        try:
            y_pred, y_true, metrics = evaluate_one(
                engine, Path(row["feature_path"]), Path(row["ground_truth_path"]), label)
        except Exception as e:
            print(f"  [WARN] {label} 失败: {e}")
            continue
        per_subject[label] = metrics
        all_preds.append(y_pred)
        all_trues.append(y_true)

    # 4. 汇总
    subj_df = pd.DataFrame(per_subject)  # index=metric, columns=subject
    summary = summarize(subj_df)

    # 总体混淆矩阵 (与训练一致: 逐被试混淆矩阵求和)
    from sleep_analysis.classification.deep_learning.utils import get_num_classes
    num_classes = get_num_classes(engine.classification_type)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for label in subj_df.columns:
        conf_matrix += np.array(subj_df.loc["confusion_matrix", label])
    summary["confusion_matrix"] = conf_matrix.tolist()

    print(f"\n{'='*60}")
    print(f"评估结果 ({len(per_subject)} 被试, per-subject mean)")
    print(f"{'='*60}")
    for k in _METRIC_KEYS:
        print(f"  {k:10s}: {summary[k]:.4f}")

    # 5. 与训练结果对比
    comparison = compare_with_training(summary, run_dir, dataset=args.dataset)
    if comparison:
        print(f"\n与训练结果对比 (evaluate - training):")
        for k in _METRIC_KEYS:
            d = comparison["diff"].get(k)
            if d is not None:
                mark = "✅" if abs(d) < 0.01 else "⚠️"
                print(f"  {k:10s}: train={comparison['training'][k]:.4f}  "
                      f"evaluate={comparison['evaluate'][k]:.4f}  diff={d:+.4f} {mark}")

    # 6. 保存
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluate_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    subj_df.to_csv(output_dir / "per_subject_metrics.csv")
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "n_subjects": len(per_subject),
            "summary": summary,
            "training_comparison": comparison,
            "config": {
                "run_dir": str(run_dir),
                "backend": args.backend,
                "night_norm": args.night_norm,
            },
        }, f, indent=2, default=str)
    print(f"\n结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
