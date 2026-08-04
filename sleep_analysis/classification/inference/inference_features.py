#!/usr/bin/env python3
"""
基于预处理特征的睡眠分期推理。

与 inference_full.py 的区别：
  - 不涉及原始数据预处理，假设特征 CSV 已存在
  - 支持两种输入模式：按被试自动查找 或 直接指定文件路径

用法:
    # 按被试 + 数据集自动查找特征
    python inference_features.py \\
        --run-dir exports_our/2026-07-31_154445 \\
        --subject 0001 \\
        --dataset mesa \\
        --backend torch

    # 直接指定特征和标注文件
    python inference_features.py \\
        --run-dir exports_our/2026-07-31_154445 \\
        --features /path/to/features.csv \\
        --ground-truth /path/to/ground_truth.csv \\
        --backend onnx

    # 批量推理
    python inference_features.py \\
        --run-dir exports_our/2026-07-31_154445 \\
        --subject 0001,0002,0006 \\
        --dataset mesa \\
        --backend torch

输出:
    {output_dir}/per_subject_predictions/{subject}.csv
    {output_dir}/metrics_{subject}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

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
# 数据集路径解析
# ---------------------------------------------------------------------------

_DATASET_PROCESSED_KEYS = {
    "mesa": "processed_mesa_path",
    "shhs1": "shhs1_processed_path",
    "shhs2": "shhs2_processed_path",
}


def resolve_paths(
    subject_id: str,
    dataset: str,
    study_cfg: dict,
) -> tuple[Path, Path]:
    """
    根据被试编号和数据集名称解析特征文件和标注文件路径。

    Returns
    -------
    (feature_path, ground_truth_path)
    """
    key = _DATASET_PROCESSED_KEYS.get(dataset.lower())
    if key is None:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Supported: {list(_DATASET_PROCESSED_KEYS.keys())}"
        )

    processed = Path(study_cfg[key])

    feat_path = processed / "features_full_combined" / f"features_combined{subject_id}.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    gt_path = processed / "actigraph_data_clean" / f"actigraph_data_clean{subject_id}.csv"
    if not gt_path.exists():
        # SHHS 标注可能在别的位置，尝试 shhs_ground_truth
        gt_alt = processed / "shhs_ground_truth" / f"ground_truth_{subject_id}.csv"
        if gt_alt.exists():
            gt_path = gt_alt
        else:
            raise FileNotFoundError(
                f"Ground truth not found at {gt_path} or {gt_alt}"
            )

    return feat_path, gt_path


# ---------------------------------------------------------------------------
# 核心推理函数
# ---------------------------------------------------------------------------

def run_inference(
    run_dir: Path,
    feature_path: Path,
    ground_truth_path: Optional[Path],
    subject_id: str,
    backend: str,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    对单个被试执行推理。

    这是跨引擎的通用接口 — 根据 backend 选择引擎，其余流程一致。
    """
    if backend == "torch":
        from sleep_analysis.classification.inference.engine_torch import TorchInferenceEngine
        engine = TorchInferenceEngine(run_dir)
    elif backend == "onnx":
        from sleep_analysis.classification.inference.engine_onnx import OnnxInferenceEngine
        engine = OnnxInferenceEngine(run_dir)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # 推理
    y_pred, y_true = engine.predict_from_files(feature_path, ground_truth_path)

    # 评分
    from sleep_analysis.classification.inference.data_utils import compute_metrics
    metrics = compute_metrics(y_pred, y_true, engine.classification_type)

    result = {
        "subject_id": subject_id,
        "n_epochs": len(y_pred),
        "classification_type": engine.classification_type,
        "predictions": y_pred.tolist(),
        "ground_truth": y_true.tolist(),
        "metrics": metrics,
    }

    # 打印
    print(f"\n  === Results for {subject_id} ===")
    print(f"  Epochs: {result['n_epochs']}")
    for k in ["accuracy", "kappa", "mcc", "f1"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")
    print(f"  Confusion matrix:")
    import numpy as np
    print(f"  {np.array(metrics['confusion_matrix'])}")

    # 保存
    if output_dir is None:
        output_dir = run_dir / "inference_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    pred_dir = output_dir / "per_subject_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"prediction": result["predictions"],
         "ground_truth": result["ground_truth"]}
    ).to_csv(pred_dir / f"{subject_id}.csv", index_label="epoch")

    with open(output_dir / f"metrics_{subject_id}.json", "w") as f:
        json.dump({k: v for k, v in result.items()
                   if k not in ("predictions", "ground_truth")},
                  f, indent=2)

    print(f"  Saved to: {output_dir}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="基于预处理特征的睡眠分期推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 按被试 + 数据集
  %(prog)s --run-dir exports_our/2026-07-31_154445 --subject 0001 --dataset mesa

  # 直接指定文件
  %(prog)s --run-dir exports_our/2026-07-31_154445 \\
      --features /path/to/features.csv --ground-truth /path/to/gt.csv

  # ONNX 后端
  %(prog)s --run-dir exports_our/2026-07-31_154445 --subject 0001 --dataset mesa --backend onnx

  # 批量
  %(prog)s --run-dir exports_our/2026-07-31_154445 --subject 0001,0002,0006 --dataset mesa
        """,
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="训练运行目录 (含 config.json 和 checkpoints/)")
    parser.add_argument("--backend", type=str, default="torch",
                        choices=["torch", "onnx"],
                        help="推理后端 (default: torch)")

    # 输入模式 A: 被试 + 数据集
    parser.add_argument("--subject", type=str, default=None,
                        help="被试编号 (多个用逗号分隔, 如 0001,0002)")
    parser.add_argument("--dataset", type=str, default="mesa",
                        choices=["mesa", "shhs1", "shhs2"],
                        help="数据集名称 (default: mesa)")

    # 输入模式 B: 直接指定文件
    parser.add_argument("--features", type=str, default=None,
                        help="特征 CSV 文件路径")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="标注 CSV 文件路径")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: {run_dir}/inference_results)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "inference_results"

    # 解析被试列表
    if args.subject:
        subject_ids = [s.strip() for s in args.subject.split(",")]
    else:
        subject_ids = []

    # 模式 B: 直接文件路径
    if args.features:
        if not Path(args.features).exists():
            print(f"[ERROR] Feature file not found: {args.features}")
            sys.exit(1)
        gt_path = Path(args.ground_truth) if args.ground_truth else None
        subj_id = args.subject or Path(args.features).stem.replace("features_combined", "")
        results = [run_inference(
            run_dir=run_dir,
            feature_path=Path(args.features),
            ground_truth_path=gt_path,
            subject_id=subj_id,
            backend=args.backend,
            output_dir=output_dir,
        )]
    else:
        # 模式 A: 被试 + 数据集
        if not subject_ids:
            print("[ERROR] 请指定 --subject 或 --features")
            sys.exit(1)
        study_cfg = _load_study_config()
        results = []
        for subj in subject_ids:
            print(f"\n{'='*60}")
            print(f"Processing subject: {subj}")
            print(f"{'='*60}")
            feat_path, gt_path = resolve_paths(subj, args.dataset, study_cfg)
            results.append(run_inference(
                run_dir=run_dir,
                feature_path=feat_path,
                ground_truth_path=gt_path,
                subject_id=subj,
                backend=args.backend,
                output_dir=output_dir,
            ))

    # 汇总
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"Summary ({len(results)} subjects)")
        print(f"{'='*60}")
        for r in results:
            m = r["metrics"]
            print(f"  {r['subject_id']}: acc={m['accuracy']:.4f} "
                  f"kappa={m['kappa']:.4f} mcc={m['mcc']:.4f}")


if __name__ == "__main__":
    main()
