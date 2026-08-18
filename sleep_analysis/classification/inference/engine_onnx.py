"""
ONNX Runtime 推理引擎。
加载训练产出的 config + scaler + ONNX 模型，对单个被试进行睡眠分期推理。

独立于 PyTorch 引擎 — 外部 ONNX 用户只需关注此文件 + data_utils.py + model.onnx。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sleep_analysis.classification.deep_learning.utils import get_num_classes, get_num_input
from sleep_analysis.classification.inference.data_utils import (
    apply_scaler,
    build_sequences,
    compute_metrics,
    load_features_from_path,
    load_ground_truth_from_path,
    load_run_config,
    load_scaler,
    resolve_data_paths,
    select_features,
)


class OnnxInferenceEngine:
    """
    ONNX Runtime 推理引擎。

    用法:
        engine = OnnxInferenceEngine(Path("exports_our/2026-07-31_154445"))
        predictions, metrics = engine.evaluate(subject_id="0001",
                                                processed_path=Path(".../mesa_processed"))
    """

    def __init__(self, run_dir: Path):
        """
        Parameters
        ----------
        run_dir : Path
            训练运行目录，包含 config.json、checkpoints/model.onnx、checkpoints/scaler.json。
        """
        import onnxruntime as ort

        self.run_dir = Path(run_dir)
        self.config = load_run_config(self.run_dir)

        # 模型超参数
        self.classification_type = self.config["classification"]
        self.modality = self.config["modality"]
        self.num_classes = get_num_classes(self.classification_type)
        self.input_size = get_num_input(self.modality)
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.seq_len = self.config.get("seq_len", 21)
        self.causal = self.config.get("causal", False)  # 注意: 只控制序列 padding 方向, 与 processing_config.causal (数据生成) 无关
        self.lookahead_min = self.config.get("lookahead_min", None)  # ✅2026-08-17: 特征平移(分钟)
        self.stateful = self.config.get("stateful", False)  # ✅2026-08-11: 有状态推理 (逐帧扫描)

        # 加载第一层 scaler
        self.scaler_mean, self.scaler_scale = load_scaler(self.run_dir)
        if len(self.scaler_mean) != self.input_size:
            raise ValueError(
                f"Scaler n_features ({len(self.scaler_mean)}) != "
                f"model input_size ({self.input_size})."
            )

        # 加载 ONNX 模型
        onnx_path = self._find_onnx_model()
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        # ✅2026-08-11: 用名字映射而非 [0] 索引 — stateful 图有 4 输入 (x,h,c,buf) / 4 输出
        # (logits,h_n,c_n,buf_out); 无状态图仍是 "input"/"output", 行为不变
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"[OnnxEngine] Loaded model from {self.run_dir}")
        print(f"  classification: {self.classification_type}")
        print(f"  modality: {self.modality}")
        print(f"  input_size: {self.input_size}")
        print(f"  onnx model: {onnx_path}")
        print(f"  provider: CPUExecutionProvider")

    def _find_onnx_model(self) -> Path:
        """查找 ONNX 模型文件。"""
        # 优先 model.onnx，其次遍历 checkpoints/
        candidates = [
            self.run_dir / "checkpoints" / "model.onnx",
        ]
        checkpoints_dir = self.run_dir / "checkpoints"
        if checkpoints_dir.exists():
            for p in sorted(checkpoints_dir.glob("*.onnx"), reverse=True):
                if p not in candidates:
                    candidates.append(p)
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No .onnx file found in {checkpoints_dir}. "
            "请先运行 export_onnx.py 导出模型。"
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict(
        self,
        subject_id: str,
        processed_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对单个被试推理 (按被试编号从标准目录加载, MESA/SHHS 自适应)。"""
        feat_path, gt_path = resolve_data_paths(subject_id, processed_path)
        return self.predict_from_files(feat_path, gt_path, subject_id)

    def predict_from_files(
        self,
        feature_path: Path,
        ground_truth_path: Optional[Path] = None,
        label: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单个被试推理 (直接指定文件路径)。

        Parameters
        ----------
        feature_path : Path  特征 CSV 文件路径
        ground_truth_path : Path or None  标注 CSV 路径
        label : str  用于日志输出的标识
        """
        # ---- 1. 加载特征 ----
        feature_df = load_features_from_path(feature_path)

        # ---- 2. 特征选择 ----
        selected = select_features(feature_df, self.modality)
        n_epochs = len(selected)
        features = selected.values.astype(np.float32)
        print(f"  {label}: {n_epochs} epochs, "
              f"{features.shape[1]} features")

        # ---- 3. 构建滑动窗口 / stateful 逐帧扫描 ----
        if self.stateful:
            # ✅2026-08-11: stateful — 逐帧 session.run, 状态 (h,c,buf) 由引擎维护
            x = apply_scaler(features, self.scaler_mean, self.scaler_scale)  # (n, F)
            h = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
            c = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
            buf = np.zeros((1, self.seq_len - 1, self.hidden_size), dtype=np.float32)
            # warm-up: 首帧 seq_len-1 次 (与训练一致)
            for _ in range(self.seq_len - 1):
                _, h, c, buf = self.session.run(
                    self.output_names,
                    {"x": x[0:1], "h": h, "c": c, "buf": buf},
                )
            logits_list = []
            for t in range(n_epochs):
                out, h, c, buf = self.session.run(
                    self.output_names,
                    {"x": x[t:t + 1], "h": h, "c": c, "buf": buf},
                )
                logits_list.append(out[0])
            onnx_out = np.stack(logits_list)  # (n, num_classes)
        else:
            # ---- 3. 构建滑动窗口 ----
            x = build_sequences(features, seq_len=self.seq_len, causal=self.causal, lookahead_min=self.lookahead_min)

            # ---- 4. 第一层标准化 ----
            x = apply_scaler(x, self.scaler_mean, self.scaler_scale)

            # ---- 5. ONNX 推理 ----
            onnx_out = self.session.run(
                [self.output_names[0]],
                {self.input_names[0]: x},
            )[0]  # shape: (n_epochs, num_classes)

        # ---- 6. 解码预测 ----
        if self.classification_type == "binary":
            y_pred = (1 / (1 + np.exp(-onnx_out)) >= 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(onnx_out, axis=1).astype(int)

        # ---- 7. 加载真实标注 ----
        if ground_truth_path is not None:
            y_true = load_ground_truth_from_path(ground_truth_path, self.classification_type)
            y_true = y_true[:n_epochs]
        else:
            y_true = np.zeros(n_epochs, dtype=int)

        return y_pred, y_true

    # ------------------------------------------------------------------
    # 完整评估
    # ------------------------------------------------------------------

    def evaluate(
        self,
        subject_id: str,
        processed_path: Path,
    ) -> dict:
        """
        推理 + 与标注对比，返回完整结果。
        """
        y_pred, y_true = self.predict(subject_id, processed_path)
        metrics = compute_metrics(y_pred, y_true, self.classification_type)

        result = {
            "subject_id": subject_id,
            "n_epochs": len(y_pred),
            "classification_type": self.classification_type,
            "predictions": y_pred.tolist(),
            "ground_truth": y_true.tolist(),
            "metrics": metrics,
        }

        # 打印摘要
        print(f"\n  === Results for subject {subject_id} ===")
        print(f"  Epochs: {result['n_epochs']}")
        for k in ["accuracy", "kappa", "mcc", "f1"]:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")
        print(f"  Confusion matrix:")
        cm = np.array(metrics["confusion_matrix"])
        print(f"  {cm}")

        return result

    def save_results(
        self,
        result: dict,
        output_dir: Path,
    ) -> None:
        """保存推理结果到文件。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        subj = result["subject_id"]

        pred_dir = output_dir / "per_subject_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"prediction": result["predictions"],
             "ground_truth": result["ground_truth"]}
        ).to_csv(pred_dir / f"{subj}.csv", index_label="epoch")

        metrics_path = output_dir / f"metrics_{subj}.json"
        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in result.items()
                       if k not in ("predictions", "ground_truth")},
                      f, indent=2)

        print(f"  Predictions saved to: {pred_dir / f'{subj}.csv'}")
        print(f"  Metrics saved to:      {metrics_path}")
