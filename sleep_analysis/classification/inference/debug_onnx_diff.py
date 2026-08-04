#!/usr/bin/env python3
"""
PyTorch vs ONNX 逐层输出对比诊断脚本。

用法:
    python debug_onnx_diff.py --run-dir exports_our/2026-07-31_154445 --subject 0001

输出:
    - 每层的 max_diff 和 argmax_mismatch
    - 差异最大的特征维度和位置
    - 各层中间输出保存为 .npy 文件到 {run_dir}/debug_onnx/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

# 项目路径
_SCRIPT_DIR = Path(__file__).resolve().parent
for _ in range(8):
    if (_SCRIPT_DIR / "study_data.json").exists():
        break
    _SCRIPT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# PyTorch: 拦截各层输出
# ---------------------------------------------------------------------------

class HookModel(torch.nn.Module):
    """包装原始 Model，通过 hook 收集各层输出。"""

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.intermediates = {}
        self._hooks = []

        # 注册 hook 到关键子模块
        for name, module in self.model.named_modules():
            if name in ("lstm", "fc_1", "fc", ""):
                continue  # 跳过容器和顶层
            h = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

        # LSTM 需要特殊处理 (它输出 (output, (h, c)))
        h_lstm = self.model.lstm.register_forward_hook(self._make_hook("lstm"))
        self._hooks.append(h_lstm)

    def _make_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                self.intermediates[name] = out[0].detach()
            else:
                self.intermediates[name] = out.detach()
        return hook

    def forward(self, x):
        self.intermediates = {}
        return self.model(x)


# ---------------------------------------------------------------------------
# ONNX: 提取中间节点输出
# ---------------------------------------------------------------------------

def extract_onnx_intermediates(session, input_data, node_names):
    """通过逐一添加中间输出节点来提取各层 ONNX 输出。"""
    # 构建一个多输出的 session
    all_outputs = [o.name for o in session.get_outputs()] + list(node_names)
    outputs = session.run(all_outputs, {"input": input_data})
    # outputs 顺序: 先原输出，再中间节点
    num_orig = len(session.get_outputs())
    final = outputs[:num_orig]
    intermediates = dict(zip(node_names, outputs[num_orig:]))
    return final, intermediates


# ---------------------------------------------------------------------------
# 手动计算：PyTorch batch norm 用 NumPy 方式重算
# ---------------------------------------------------------------------------

def numpy_style_batch_norm(x, eps=1e-8):
    """用 NumPy 方式模拟 ONNX 的 batch norm (总体方差，无 Bessel 校正)。"""
    x = x.astype(np.float32)
    mean = x.mean(axis=(0, 1), keepdims=True)
    diff_sq = (x - mean) ** 2
    var = diff_sq.mean(axis=(0, 1), keepdims=True)
    std = np.sqrt(var) + eps
    return (x - mean) / std, mean.squeeze(), std.squeeze()


def torch_style_batch_norm(x_tensor, eps=1e-8):
    """PyTorch 原生的 batch norm (样本标准差，Bessel 校正)。"""
    mean = x_tensor.mean(dim=(0, 1), keepdim=True)
    std = x_tensor.std(dim=(0, 1), keepdim=True, unbiased=True) + eps
    return (x_tensor - mean) / std, mean.squeeze(), std.squeeze()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyTorch vs ONNX 逐层对比诊断")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--subject", type=str, default="0001")
    parser.add_argument("--dataset", type=str, default="mesa")
    parser.add_argument("--save-npy", action="store_true", default=True,
                        help="保存中间层输出为 .npy (默认开启)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # ---- 数据准备 ----
    from sleep_analysis.classification.inference.data_utils import (
        load_run_config, load_scaler, load_features,
        select_features, build_sequences, apply_scaler,
    )

    config = load_run_config(run_dir)
    mean_scaler, scale_scaler = load_scaler(run_dir)
    processed_path = Path(json.load(open(run_dir / "../../study_data.json"))
                           .get("processed_mesa_path", "/srv/shared/psgdata/mesa_processed"))

    feat = load_features(args.subject, processed_path)
    selected = select_features(feat, config["modality"])
    features = selected.values.astype(np.float32)
    x_raw = build_sequences(features, seq_len=config.get("seq_len", 21), causal=config.get("causal", False))
    x = apply_scaler(x_raw, mean_scaler.astype(np.float32), scale_scaler.astype(np.float32))
    x_t = torch.from_numpy(x).float()  # CPU tensor

    n_epochs, seq_len, n_features = x.shape
    print(f"Input: {n_epochs} epochs × {seq_len} seq_len × {n_features} features")

    # =====================================================================
    # Step 1: Batch Norm 对比 (只用数据，不涉及模型)
    # =====================================================================
    print(f"\n{'='*60}")
    print("Step 1: Batch Norm (x.mean/std over dims 0,1)")
    print(f"{'='*60}")

    eps = 1e-8
    norm_t, mean_t, std_t = torch_style_batch_norm(x_t, eps=eps)
    norm_np, mean_np, std_np = numpy_style_batch_norm(x, eps=eps)

    norm_t_np = norm_t.numpy()
    diff_norm = np.abs(norm_t_np - norm_np)
    print(f"  mean diff per feature: {np.abs(mean_t.numpy() - mean_np)}")
    print(f"  std diff per feature:  {np.abs(std_t.numpy() - std_np)}")
    print(f"  norm max diff: {diff_norm.max():.6f}")

    # 逐特征差异
    for i in range(n_features):
        d = diff_norm[:, :, i]
        print(f"    feat[{i:2d}]: std_t={std_t[i].item():.4e} std_np={std_np[i]:.4e}  "
              f"norm max_diff={d.max():.6e}")

    # =====================================================================
    # Step 2: 完整模型前向 (PyTorch CPU)
    # =====================================================================
    print(f"\n{'='*60}")
    print("Step 2: Full model forward")
    print(f"{'='*60}")

    from sleep_analysis.classification.deep_learning.lstm.model import Model
    from sleep_analysis.classification.deep_learning.utils import get_num_input, get_num_classes

    # 构建 CPU 模型
    model_cpu = Model(
        num_classes=get_num_classes(config["classification"]),
        input_size=get_num_input(config["modality"]),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_gpu=False,
        use_attention=False,
        dataset_name=config.get("dataset", ""),
        modality=config["modality"],
    )
    model_cpu.load_state_dict(torch.load(run_dir / "checkpoints" / "best_model.pt", map_location="cpu"))
    model_cpu.eval()

    # 提取各层
    layer_outputs = {}

    # 手动前向，记录每一步
    with torch.no_grad():
        # batch norm (模型内部)
        mean_m = x_t.mean(dim=(0, 1), keepdim=True)
        std_m = x_t.std(dim=(0, 1), keepdim=True, unbiased=True) + eps
        x_norm = (x_t - mean_m) / std_m
        layer_outputs["batch_norm"] = x_norm.numpy()

        # LSTM
        h0 = torch.zeros(model_cpu.num_layers, n_epochs, model_cpu.hidden_size)
        c0 = torch.zeros(model_cpu.num_layers, n_epochs, model_cpu.hidden_size)
        lstm_out, _ = model_cpu.lstm(x_norm, (h0, c0))
        layer_outputs["lstm"] = lstm_out.numpy()

        # mean pooling
        pooled = lstm_out.mean(dim=1)
        layer_outputs["pooling"] = pooled.numpy()

        # relu -> fc_1
        out = model_cpu.relu(pooled)
        out = model_cpu.fc_1(out)
        layer_outputs["fc_1"] = out.numpy()

        # dropout -> relu
        out = model_cpu.dropout(out)
        out = model_cpu.relu(out)
        layer_outputs["fc_1_relu"] = out.numpy()

        # fc (final)
        torch_out = model_cpu.fc(out).numpy()
        layer_outputs["final"] = torch_out

    # =====================================================================
    # Step 3: ONNX 完整前向
    # =====================================================================
    print("Running ONNX ...")
    onnx_path = run_dir / "checkpoints" / "model.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 获取 ONNX 模型中的关键中间节点名
    # 打印图结构找到关键节点
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    node_names = [n.name for n in onnx_model.graph.node]
    print(f"  ONNX graph has {len(node_names)} nodes")

    # 找关键节点: 用于比较的节点
    # LSTM 输出一般叫 /lstm/... 或类似名字
    # FC 输出一般叫 /fc/...
    lstm_nodes = [n for n in node_names if "lstm" in n.lower() and "output" in n.lower()]
    if not lstm_nodes:
        lstm_nodes = [n for n in node_names if "lstm" in n.lower()]

    # 直接跑完整的 ONNX
    onnx_out = session.run(None, {"input": x})[0]
    layer_outputs["onnx_final"] = onnx_out

    # =====================================================================
    # Step 4: 逐层对比
    # =====================================================================
    print(f"\n{'='*60}")
    print("Results: PyTorch vs ONNX")
    print(f"{'='*60}")

    # 4a: Batch norm 对比
    print(f"\n--- Layer: batch_norm ---")
    bn_diff = np.abs(layer_outputs["batch_norm"] - norm_np)  # PyTorch norm vs NumPy(ONNX-style) norm
    print(f"  PyTorch batch_norm vs NumPy-style batch_norm:")
    print(f"    max_diff:  {bn_diff.max():.6f}")
    print(f"    mean_diff: {bn_diff.mean():.6f}")

    # 4b: 最终输出对比
    print(f"\n--- Layer: final output ---")
    final_diff = np.abs(torch_out - onnx_out)
    print(f"  PyTorch final vs ONNX final:")
    print(f"    max_diff:        {final_diff.max():.6f}")
    print(f"    mean_diff:       {final_diff.mean():.6f}")
    torch_pred = np.argmax(torch_out, axis=1)
    onnx_pred = np.argmax(onnx_out, axis=1)
    n_mismatch = (torch_pred != onnx_pred).sum()
    print(f"    argmax mismatch: {n_mismatch}/{n_epochs} ({100*n_mismatch/n_epochs:.2f}%)")

    # 4c: 按 epoch 看差异最大的几个
    epoch_diffs = np.abs(torch_out - onnx_out).max(axis=1)
    top_epochs = np.argsort(-epoch_diffs)[:5]
    print(f"\n  Top-5 divergent epochs:")
    for rank, ei in enumerate(top_epochs):
        print(f"    epoch {ei}: diff={epoch_diffs[ei]:.4f}  "
              f"torch_pred={torch_pred[ei]} onnx_pred={onnx_pred[ei]}")

    # =====================================================================
    # Step 5: 保存
    # =====================================================================
    if args.save_npy:
        out_dir = run_dir / "debug_onnx"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in layer_outputs.items():
            np.save(out_dir / f"{name}.npy", arr.astype(np.float32))
        np.save(out_dir / "norm_pytorch.npy", norm_t_np.astype(np.float32))
        np.save(out_dir / "norm_numpy_onnx_style.npy", norm_np.astype(np.float32))
        print(f"\n  Intermediate outputs saved to: {out_dir}/")
        print(f"  Files: {sorted(f.name for f in out_dir.glob('*.npy'))}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
