#!/usr/bin/env python3
"""
PyTorch → ONNX 模型导出。

将训练好的 LSTM 模型导出为 ONNX 格式，供 engine_onnx.py 或外部 ONNX Runtime 使用。

用法:
    python export_onnx.py --run-dir exports_our/2026-07-31_154445

输出:
    {run_dir}/checkpoints/model.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# 将项目根目录加入 path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR
for _ in range(8):
    if (_PROJECT_ROOT / "study_data.json").exists():
        break
    _PROJECT_ROOT = _PROJECT_ROOT.parent

sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="训练运行目录 (含 config.json 和 checkpoints/best_model.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径 (默认: {run_dir}/checkpoints/model.onnx)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="序列长度 (默认从 config.json 读取)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset 版本 (default: 17)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # 1. 加载配置
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"[ERROR] config.json not found in {run_dir}")
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    classification_type = config["classification"]
    modality = config["modality"]
    seq_len = args.seq_len or config.get("seq_len", 21)
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]

    # 2. 计算 input_size
    from sleep_analysis.classification.deep_learning.utils import get_num_input, get_num_classes
    input_size = get_num_input(modality)
    num_classes = get_num_classes(classification_type)

    print(f"Config:")
    print(f"  classification: {classification_type}")
    print(f"  modality: {modality}")
    print(f"  input_size: {input_size}, hidden: {hidden_size}, layers: {num_layers}")
    print(f"  seq_len: {seq_len}, num_classes: {num_classes}")

    # 3. 构建模型并加载权重
    from sleep_analysis.classification.deep_learning.lstm.model import Model

    # Wrapper: 移除 forward 中的 self.to(device) — 导出时已在 CPU
    # 原模型在 forward 里调用 self.to(device)，dynamo 会把所有参数的 .to()
    # 都展开为显式算子，导致导出图爆炸。CPU 导出不需要这一步。
    class _ExportModel(Model):
        def forward(self, x):
            # 跳过 self.to(device)，直接走后续逻辑
            if len(x.shape) == 2:
                x = x.reshape(x.shape[0], x.shape[1], 1)

            mean_x = x.mean(dim=(0, 1), keepdim=True)
            std_x = x.std(dim=(0, 1), keepdim=True) + 1e-5
            x = (x - mean_x) / std_x

            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

            lstm_out, _ = self.lstm(x, (h_0, c_0))

            if self.use_attention:
                attn_out = self.attention(lstm_out)
            else:
                attn_out = lstm_out.mean(dim=1)

            out = self.relu(attn_out)
            out = self.fc_1(out)
            out = self.dropout(out)
            out = self.relu(out)
            out = self.fc(out)
            return out

    model = _ExportModel(
        num_classes=num_classes,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_gpu=False,
        use_attention=False,
        dataset_name=config.get("dataset", ""),
        modality=modality,
    )
    model.eval()

    weights_path = run_dir / "checkpoints" / "best_model.pt"
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"  Weights loaded from: {weights_path}")

    # 4. 创建 dummy input
    dummy_input = torch.randn(1, seq_len, input_size, dtype=torch.float32)
    print(f"  Dummy input shape: {dummy_input.shape}")

    # 5. 导出 ONNX
    output_path = Path(args.output) if args.output else run_dir / "checkpoints" / "model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 动态 batch 维度，固定 seq_len 和 input_size
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"\n  ONNX model saved to: {output_path}")

    # 5b. 将外部权重文件 (.onnx.data) 合并为单文件
    data_file = output_path.with_suffix(".onnx.data")
    if data_file.exists():
        import onnx
        onnx_model = onnx.load(str(output_path))
        data_file.unlink()  # 删除外部数据文件
        onnx.save(onnx_model, str(output_path))
        print(f"  External data merged → single file ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 6. 验证
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX model verified: OK")

    # 7. 快速推理对比 (可选，确认精度)
    print(f"\n  Quick sanity check (ONNX vs PyTorch):")
    import onnxruntime as ort

    # PyTorch 输出
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    # ONNX 输出
    session = ort.InferenceSession(str(output_path))
    onnx_out = session.run(None, {"input": dummy_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"  PyTorch output: {torch_out.flatten()[:5]}...")
    print(f"  ONNX output:    {onnx_out.flatten()[:5]}...")
    print(f"  Max diff: {max_diff:.6f}")
    if max_diff < 1e-4:
        print(f"  ✓ Outputs match (max diff < 1e-4)")
    else:
        print(f"  ⚠ Outputs differ (max diff = {max_diff:.6f}) — 可能需要检查 opset 版本")

    # 8. 保存导出信息
    from datetime import datetime
    export_info = {
        "exported_at": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "source_run": str(run_dir),
        "source_weights": str(weights_path),
        "onnx_model": str(output_path),
        "opset": args.opset,
        "input_shape": ["batch_size", seq_len, input_size],
        "output_shape": ["batch_size", num_classes],
        "classification_type": classification_type,
        "modality": modality,
        "seq_len": seq_len,
        "requires_scaler": f"{run_dir}/checkpoints/scaler.json",
    }
    info_path = output_path.with_suffix(".onnx_info.json")
    with open(info_path, "w") as f:
        json.dump(export_info, f, indent=2)
    print(f"\n  Export info saved to: {info_path}")


if __name__ == "__main__":
    main()
