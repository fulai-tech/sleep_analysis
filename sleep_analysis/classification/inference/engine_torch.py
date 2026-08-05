"""
PyTorch 推理引擎。
加载训练产出的 config + scaler + 模型权重，对单个被试进行睡眠分期推理。

独立于 ONNX 引擎 — 外部 PyTorch 用户只需关注此文件 + data_utils.py。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sleep_analysis.classification.deep_learning.lstm.model import Model
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


class TorchInferenceEngine:
    """
    PyTorch 推理引擎。

    用法:
        engine = TorchInferenceEngine(Path("exports_our/2026-07-31_154445"))
        predictions, metrics = engine.predict(subject_id="0001",
                                               processed_path=Path(".../mesa_processed"))
    """

    def __init__(self, run_dir: Path, device: str = "auto"):
        """
        Parameters
        ----------
        run_dir : Path
            训练运行目录，包含 config.json 和 checkpoints/ 子目录。
        device : str
            "auto" (有 GPU 就用), "cpu", "cuda"。
        """
        self.run_dir = Path(run_dir)
        self.config = load_run_config(self.run_dir)

        # 模型超参数
        self.classification_type = self.config["classification"]
        self.modality = self.config["modality"]
        self.num_classes = get_num_classes(self.classification_type)
        self.input_size = get_num_input(self.modality)
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.dropout = self.config["dropout"]
        self.seq_len = self.config.get("seq_len", 21)
        self.causal = self.config.get("causal", False)

        # 设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 加载第一层 scaler
        self.scaler_mean, self.scaler_scale = load_scaler(self.run_dir)
        if len(self.scaler_mean) != self.input_size:
            raise ValueError(
                f"Scaler n_features ({len(self.scaler_mean)}) != "
                f"model input_size ({self.input_size}). 请确认 modality 配置一致。"
            )

        # 构建并加载模型
        self.model = self._build_model()
        self._load_weights()

        print(f"[TorchEngine] Loaded model from {self.run_dir}")
        print(f"  classification: {self.classification_type}")
        print(f"  modality: {self.modality}")
        print(f"  input_size: {self.input_size}, hidden: {self.hidden_size}, "
              f"layers: {self.num_layers}")
        print(f"  device: {self.device}")

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------

    def _build_model(self) -> Model:
        """构建与训练时完全相同的 LSTM 模型结构。"""
        model = Model(
            num_classes=self.num_classes,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_gpu=(self.device == "cuda"),
            use_attention=False,  # 训练/测试均未使用 attention，用 mean pooling
            dataset_name=self.config.get("dataset", ""),
            modality=self.modality,
        )
        model.eval()
        return model

    def _load_weights(self) -> None:
        """加载 best_model.pt 权重。"""
        weights_path = self.run_dir / "checkpoints" / "best_model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        subject_id: str,
        processed_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对单个被试推理 (按被试编号从标准目录加载, MESA/SHHS 自适应)。"""
        feat_path, gt_path = resolve_data_paths(subject_id, processed_path)
        return self.predict_from_files(feat_path, gt_path, subject_id)

    @torch.no_grad()
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
        ground_truth_path : Path or None  标注 CSV 路径 (None 则返回 y_true 全零)
        label : str  用于日志输出的标识
        """
        # ---- 1. 加载特征 ----
        feature_df = load_features_from_path(feature_path)

        # ---- 2. 特征选择 (匹配训练时的列) ----
        selected = select_features(feature_df, self.modality)
        n_epochs = len(selected)
        features = selected.values.astype(np.float32)
        print(f"  {label}: {n_epochs} epochs, "
              f"{features.shape[1]} features")

        # ---- 3. 构建滑动窗口 ----
        x = build_sequences(features, seq_len=self.seq_len, causal=self.causal)

        # ---- 4. 第一层标准化 (训练集拟合的 scaler) ----
        x = apply_scaler(x, self.scaler_mean, self.scaler_scale)

        # ---- 5. 转为 tensor 并推理 ----
        x_tensor = torch.from_numpy(x).float().to(self.device)
        output = self.model(x_tensor)  # shape: (n_epochs, num_classes)

        # ---- 6. 解码预测 ----
        if self.classification_type == "binary":
            y_pred = (torch.sigmoid(output).cpu().numpy() >= 0.5).astype(int).flatten()
        else:
            y_pred = torch.argmax(output, dim=1).cpu().numpy().astype(int)

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

        Returns
        -------
        result : dict
            keys: "subject_id", "n_epochs", "predictions", "ground_truth",
                  "metrics", "classification_type"
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

        # 保存逐 epoch 预测 (与训练脚本 per_subject_predictions 格式一致)
        pred_dir = output_dir / "per_subject_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"prediction": result["predictions"], "ground_truth": result["ground_truth"]}
        ).to_csv(pred_dir / f"{subj}.csv", index_label="epoch")

        # 保存指标
        metrics_path = output_dir / f"metrics_{subj}.json"
        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "predictions"
                       and k != "ground_truth"}, f, indent=2)

        print(f"  Predictions saved to: {pred_dir / f'{subj}.csv'}")
        print(f"  Metrics saved to:      {metrics_path}")
