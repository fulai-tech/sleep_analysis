#!/usr/bin/env python3
"""
ONNX 睡眠分期推理脚本 (独立、无 sleep_analysis 依赖)。

用法:
    python inference.py

输入:
    input_features.csv   - 预处理后的输入特征 (N epochs × M features)
    scaler_params.json   - StandardScaler 参数 (mean_, scale_, seq_len, ...)
    model.onnx           - ONNX 模型

输出:
    predictions.csv      - 逐 epoch 预测结果
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CSV = SCRIPT_DIR / "input_features.csv"
SCALER_JSON = SCRIPT_DIR / "scaler_params.json"
MODEL_ONNX = SCRIPT_DIR / "model.onnx"
OUTPUT_CSV = SCRIPT_DIR / "predictions.csv"

# onnxruntime 日志级别: 3 = warning 及以上 (默认), 显式设置保证行为确定
import onnxruntime as ort
ort.set_default_logger_severity(3)

# 分类标签映射
STAGE_NAMES = {
    "binary":  {0: "Wake", 1: "Sleep"},
    "3stage":  {0: "Wake", 1: "NREM", 2: "REM"},
    "4stage":  {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"},
    "5stage":  {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"},
}


# ---------------------------------------------------------------------------
# 工具函数 (内联，无需外部依赖)
# ---------------------------------------------------------------------------

def build_sequences(features: np.ndarray, seq_len: int, causal: bool) -> np.ndarray:
    """将 (n_epochs, n_features) 转为滑动窗口 (n_epochs, seq_len, n_features)。"""
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
    """StandardScaler: (x - mean) / scale，保持输入 dtype。"""
    mean_ = mean_.astype(x.dtype)
    scale_ = scale_.astype(x.dtype)
    eps = np.finfo(x.dtype).eps
    return (x - mean_) / (scale_ + eps)


def apply_night_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    整夜归一化 (第二层 norm 的模型外实现)。

    该模型 (2026-08-03 训练) 的 forward 内含 per-batch 归一化:
        mean = x.mean(dim=(0,1)); std = x.std(dim=(0,1), unbiased=True) + eps
    训练代码测试 pipeline 中该 norm 的统计量 = 该被试整夜数据。
    推理时在 Python 侧用整夜数据预先计算并应用 (模型图内不含归一化),
    保证与训练代码测试 pipeline 结果一致, 且与喂入 batch 大小无关。

    eps=1e-5 与训练时一致。ddof=1 对齐 torch.std(unbiased=True)。
    """
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), ddof=1, keepdims=True) + eps
    return (x - mean) / std


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    # 1. 加载输入特征
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    import pandas as pd
    df = pd.read_csv(INPUT_CSV, index_col=0)
    features = df.values.astype(np.float32)
    print(f"[1/5] Loaded features: {features.shape[0]} epochs × {features.shape[1]} features")

    # 2. 加载 scaler 参数
    with open(SCALER_JSON) as f:
        scaler_cfg = json.load(f)
    seq_len = scaler_cfg["seq_len"]
    causal = scaler_cfg.get("causal", False)
    classification = scaler_cfg["classification_type"]
    night_norm = scaler_cfg.get("night_norm", False)
    mean_ = np.array(scaler_cfg["mean_"])
    scale_ = np.array(scaler_cfg["scale_"])
    print(f"[2/5] Scaler loaded: seq_len={seq_len}, causal={causal}, "
          f"classification={classification}, night_norm={night_norm}")

    # 3. 构建序列 + 第一层标准化 (训练集拟合 scaler)
    x = build_sequences(features, seq_len=seq_len, causal=causal)
    x = apply_scaler(x, mean_, scale_)

    # 3b. 第二层归一化 (整夜, 模型外) — 复现训练测试 pipeline
    if night_norm:
        x = apply_night_norm(x)
    print(f"[3/5] Sequences built: {x.shape}")

    # 4. ONNX 推理
    session = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: x})[0]
    print(f"[4/5] ONNX inference done: {onnx_out.shape}")

    # 5. 解码 + 保存
    if classification == "binary":
        y_pred = (1 / (1 + np.exp(-onnx_out)) >= 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(onnx_out, axis=1).astype(int)

    stage_names = STAGE_NAMES.get(classification, {})
    pred_labels = [stage_names.get(p, str(p)) for p in y_pred]

    output = pd.DataFrame({
        "epoch": range(len(y_pred)),
        "prediction": y_pred,
        "stage": pred_labels,
    })
    output.to_csv(OUTPUT_CSV, index=False)
    print(f"[5/5] Results saved to: {OUTPUT_CSV}")

    # 统计
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"\nPrediction summary ({len(y_pred)} epochs):")
    for u, c in zip(unique, counts):
        name = stage_names.get(int(u), f"class_{u}")
        print(f"  {name}: {c} ({100*c/len(y_pred):.1f}%)")


if __name__ == "__main__":
    main()
