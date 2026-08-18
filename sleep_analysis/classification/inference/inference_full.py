#!/usr/bin/env python3
"""
端到端睡眠分期推理 — 基于预处理特征。

用法:
    # 快速路径：特征已预处理完成
    python inference_full.py \
        --run-dir exports_our/2026-07-31_154445 \
        --subject 0001 \
        --backend torch

    # 若特征尚未预处理，运行完整管线 (原始 EDF → 特征 → 推理)
    python inference_full.py \
        --run-dir exports_our/2026-07-31_154445 \
        --subject 0001 \
        --backend torch \
        --full-pipeline

    # 使用 ONNX 后端
    python inference_full.py \
        --run-dir exports_our/2026-07-31_154445 \
        --subject 0001 \
        --backend onnx

输出:
    - 逐 epoch 预测 {subject}.csv (与训练脚本 per_subject_predictions 格式一致)
    - 指标对比 metrics_{subject}.json
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

# 向上搜索找到 sleep_analysis 项目根目录 (包含 study_data.json)
def _find_project_root() -> Path:
    """从当前脚本目录向上搜索包含 study_data.json 的项目根目录。"""
    d = _SCRIPT_DIR
    for _ in range(8):
        if (d / "study_data.json").exists():
            return d
        d = d.parent
    raise FileNotFoundError("Cannot find project root with study_data.json")

_PROJECT_ROOT = _find_project_root()
_STUDY_CFG_PATH = _PROJECT_ROOT / "study_data.json"


def _load_study_config():
    with open(_STUDY_CFG_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 预处理管线 (仅当 --full-pipeline 且特征未就绪时执行)
# ---------------------------------------------------------------------------

def run_full_preprocessing(subject_id: str, study_cfg: dict) -> Path:
    """
    对单个 MESA 被试运行完整 6 步预处理管线。

    与 experiments/data_handling/preprocess_subset.py 流程一致，仅处理一个被试。

    Returns
    -------
    processed_path : Path  预处理产物根目录
    """
    import mesa_data_importer as importer
    import numpy as np
    import pandas as pd

    from sleep_analysis.preprocessing.mesa_dataset.edr import _extract_edr, process_resp
    from sleep_analysis.feature_extraction.mesa_datasst.rrv import extract_rrv_features_helper
    from sleep_analysis.preprocessing.utils import extract_edf_channel
    from sleep_analysis.preprocessing.mesa_dataset.ecg import process_rpoint
    from sleep_analysis.preprocessing.mesa_dataset.actigraphy import process_actigraphy
    from sleep_analysis.preprocessing.mesa_dataset.ground_truth import sleep_stage_convert_binary
    from sleep_analysis.preprocessing.mesa_dataset.respiration import check_resp_features
    from sleep_analysis.preprocessing.mesa_dataset.utils import (
        align_datastreams,
        clean_data_to_csv,
        match_exclusion_criteria,
    )
    from sleep_analysis.preprocessing.mesa_dataset.preprocess_mesa import _clean_data_helper
    from sleep_analysis.feature_extraction.mesa_datasst.actigraphy import calc_actigraph_features
    from sleep_analysis.feature_extraction.mesa_datasst.hrv import calc_hrv_features
    from sleep_analysis.feature_extraction.mesa_datasst.utils import merge_features

    edf_dir = Path(study_cfg["mesa_path_edf"])
    mesa_path = Path(study_cfg["mesa_path"])
    processed_path = Path(study_cfg["processed_mesa_path"])

    subj = subject_id
    subj_int = int(subj)

    # 检查质量排除标准
    dataset_info = pd.read_csv(mesa_path / "datasets/mesa-sleep-dataset-0.5.0.csv").set_index("mesaid")
    if match_exclusion_criteria(dataset_info, subj):
        raise ValueError(f"Subject {subj} excluded by quality criteria.")

    # ------------------------------------------------------------------
    # Step 1: EDR 特征
    # ------------------------------------------------------------------
    print(f"\n  [1/6] EDR features ...")
    out_edr = processed_path / f"edr_respiration_features_raw/edr_respiration{subj}.csv"
    if not out_edr.exists():
        out_edr.parent.mkdir(parents=True, exist_ok=True)
        raw_ecg, epochs = extract_edf_channel(edf_dir, subj_id=subj_int, channel="EKG")
        edr_signal = _extract_edr(raw_ecg, sampling_rate=256)
        resp_df, epochs = process_resp(edr_signal.respiratory_signal, epochs)
        features = extract_rrv_features_helper(resp_df, nan_pad=0.0, sampling_rate=32)
        features.to_csv(out_edr)
        print(f"    -> {out_edr} ({features.shape[0]} epochs)")
    else:
        print(f"    -> skip (exists)")

    # ------------------------------------------------------------------
    # Step 2: RRV 特征
    # ------------------------------------------------------------------
    print(f"  [2/6] RRV features ...")
    out_rrv = processed_path / f"respiration_features_raw/respiration{subj}.csv"
    if not out_rrv.exists():
        out_rrv.parent.mkdir(parents=True, exist_ok=True)
        resp_df, epochs = extract_edf_channel(edf_dir, subj_id=subj_int, channel="Thor")
        resp_df, epochs = process_resp(resp_df, epochs)
        features = extract_rrv_features_helper(resp_df)
        features.to_csv(out_rrv)
        print(f"    -> {out_rrv} ({features.shape[0]} epochs)")
    else:
        print(f"    -> skip (exists)")

    # ------------------------------------------------------------------
    # Step 3: MESA 数据清洗对齐
    # ------------------------------------------------------------------
    print(f"  [3/6] MESA data cleaning & alignment ...")
    overlap = pd.read_csv(mesa_path / "overlap/mesa-actigraphy-psg-overlap.csv")

    df_act = importer.load_single_actigraphy(mesa_path, subj_int)
    df_rpt = importer.load_single_r_point(mesa_path, subj_int)
    df_psg = importer.load_single_psg(mesa_path, subj_int)
    df_resp = importer.load_single_resp_features(processed_path, subj_int)
    df_edr = importer.load_single_edr_feature(processed_path, subj_int)

    _clean_data_helper(df_act, df_rpt, df_psg, df_resp, df_edr, overlap, subj_int)
    print(f"    -> actigraph_data_clean/ + ecg_data_clean/ done")

    # ------------------------------------------------------------------
    # Step 4: Actigraphy 特征
    # ------------------------------------------------------------------
    print(f"  [4/6] Actigraphy features ...")
    out_act_feat = processed_path / f"actigraph_features/actigraph_features{subj}.csv"
    if not out_act_feat.exists():
        out_act_feat.parent.mkdir(parents=True, exist_ok=True)
        act = pd.read_csv(processed_path / f"actigraph_data_clean/actigraph_data_clean{subj}.csv")
        feats = calc_actigraph_features(act["activity"])
        feats.to_csv(out_act_feat, index=False)
        print(f"    -> {out_act_feat}")
    else:
        print(f"    -> skip (exists)")

    # ------------------------------------------------------------------
    # Step 5: HRV 特征
    # ------------------------------------------------------------------
    print(f"  [5/6] HRV features ...")
    out_hrv_feat = processed_path / f"hrv_features/hrv_features{subj}.csv"
    if not out_hrv_feat.exists():
        out_hrv_feat.parent.mkdir(parents=True, exist_ok=True)
        hr = pd.read_csv(processed_path / f"ecg_data_clean/ecg_data_clean{subj}.csv")
        feats = calc_hrv_features(hr)
        feats.to_csv(out_hrv_feat, index=False)
        print(f"    -> {out_hrv_feat}")
    else:
        print(f"    -> skip (exists)")

    # ------------------------------------------------------------------
    # Step 6: 特征合并
    # ------------------------------------------------------------------
    print(f"  [6/6] Merge features ...")
    merge_features(overwrite=False)
    print(f"    -> features_full_combined/features_combined{subj}.csv")

    return processed_path


# ---------------------------------------------------------------------------
# 推理 + 评估 (特征路径)
# ---------------------------------------------------------------------------

def run_feature_inference(
    run_dir: Path,
    subject_id: str,
    backend: str,
    processed_path: Path,
    output_dir: Optional[Path] = None,
    night_norm: Optional[bool] = None,
) -> dict:
    """
    从预处理特征出发，完成推理 + 与标注对比。
    """
    if backend == "torch":
        from sleep_analysis.classification.inference.engine_torch import TorchInferenceEngine
        engine = TorchInferenceEngine(run_dir, night_norm=night_norm)
    elif backend == "onnx":
        # ONNX engine 将在 engine_onnx.py 中实现
        try:
            from sleep_analysis.classification.inference.engine_onnx import OnnxInferenceEngine
            engine = OnnxInferenceEngine(run_dir, night_norm=night_norm)
        except ImportError:
            print("[ERROR] ONNX engine not available yet. 请使用 --backend torch。")
            sys.exit(1)
    else:
        print(f"[ERROR] Unknown backend: {backend}")
        sys.exit(1)

    result = engine.evaluate(subject_id, processed_path)

    # 保存
    if output_dir is None:
        output_dir = run_dir / "inference_results"
    engine.save_results(result, output_dir)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="端到端睡眠分期推理 (从预处理特征出发)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --run-dir exports_our/2026-07-31_154445 --subject 0001
  %(prog)s --run-dir exports_our/2026-07-31_154445 --subject 0001 --full-pipeline
  %(prog)s --run-dir exports_our/2026-07-31_154445 --subject 0001 --backend onnx
        """,
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="训练运行目录 (含 config.json 和 checkpoints/)")
    parser.add_argument("--subject", type=str, required=True,
                        help="被试编号, 如 0001")
    parser.add_argument("--backend", type=str, default="torch",
                        choices=["torch", "onnx"],
                        help="推理后端 (default: torch)")
    parser.add_argument("--processed-path", type=str, default=None,
                        help="MESA 预处理产物目录 (默认从 study_data.json 读取)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: {run_dir}/inference_results)")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="若特征缺失, 运行完整 6 步预处理管线 (需 mesa_data_importer 等依赖)")
    parser.add_argument("--night-norm", action="store_true",
                        help="启用模型外整夜归一化 (复现训练代码测试 pipeline 的第二层 norm; "
                             "适用于 2026-08-07 之前训练的旧模型)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)

    # 加载 study_data.json
    study_cfg = _load_study_config()
    if args.processed_path:
        processed_path = Path(args.processed_path)
    else:
        processed_path = Path(study_cfg["processed_mesa_path"])

    subject_id = args.subject

    # 检查特征是否存在
    feat_file = processed_path / "features_full_combined" / f"features_combined{subject_id}.csv"
    if not feat_file.exists():
        if args.full_pipeline:
            print(f"[INFO] Features not found for subject {subject_id}, "
                  f"running full preprocessing pipeline ...")
            try:
                processed_path = run_full_preprocessing(subject_id, study_cfg)
            except Exception as e:
                print(f"[ERROR] Preprocessing failed: {e}")
                sys.exit(1)
        else:
            print(f"[ERROR] Feature file not found: {feat_file}")
            print(f"  请先运行预处理管线:")
            print(f"    cd {_PROJECT_ROOT}")
            print(f"    python experiments/data_handling/preprocess_subset.py N")
            print(f"  或添加 --full-pipeline 标志自动运行预处理。")
            sys.exit(1)

    # 运行推理
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "inference_results"
    result = run_feature_inference(
        run_dir=run_dir,
        subject_id=subject_id,
        backend=args.backend,
        processed_path=processed_path,
        output_dir=output_dir,
        night_norm=True if args.night_norm else None,
    )

    return result


if __name__ == "__main__":
    main()
