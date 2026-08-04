# ONNX Sleep Staging Model

基于 LSTM 的睡眠分期 ONNX 模型，用于部署和模型转换。

## 文件清单

| 文件 | 说明 |
|------|------|
| `inference.py` | ONNX Runtime 推理示例 |
| `model.onnx` | ONNX 模型图（protobuf） |
| `model.onnx.data` | 模型权重（外部存储，与 `.onnx` 配套） |
| `input_features.csv` | MESA 0001 号被试的输入特征，用于验证推理 |
| `scaler_params.json` | 第一层 StandardScaler 参数 |
| `predictions.csv` | `inference.py` 运行后的输出 |
| `requirements.txt` | Python 依赖 |

## 环境要求

- Python ≥ 3.10
- `numpy`, `pandas`, `onnxruntime`（见 `requirements.txt`）

```bash
pip install -r requirements.txt
```

## 快速验证

```bash
python inference.py
```

输出 `predictions.csv`，1262 个 epoch 的推理结果。

## 模型信息

### 架构

```
Input (batch, 21, 12)
  → BatchNorm (per-batch mean/std over dims 0,1)
  → LSTM (hidden=556, layers=6, batch_first=True)
  → Mean Pooling (dim=1)
  → ReLU → FC(128) → Dropout(0.255) → ReLU → FC(4)
  → Output (batch, 4)
```

- **框架**: PyTorch 2.5 → ONNX opset 18 (dynamo 导出)
- **参数量**: ~53 MB
- **Attention**: 未启用（训练和推理均用 mean pooling）
- **输入名**: `input`，**输出名**: `output`

### 输入

| 维度 | 说明 |
|------|------|
| `batch_size` | 动态，任意值 |
| `seq_len` | 固定 21（21 个连续 30s epoch，覆盖前后各约 5 分钟上下文） |
| `n_features` | 固定 12 |

12 个特征按顺序：

| 序号 | 模态 | 特征名 |
|------|------|--------|
| 0 | HRV | `_hrv_median_nni` |
| 1 | HRV | `_hrv_ratio_sd2_sd1` |
| 2 | HRV | `_hrv_median_nni` |
| 3 | HRV | `_hrv_vlf` |
| 4 | HRV | `_hrv_lf` |
| 5 | HRV | `_hrv_hf` |
| 6 | HRV | `_hrv_lf_hf_ratio` |
| 7 | HRV | `_hrv_total_power` |
| 8 | RRV | `150_RRV_MedianBB` |
| 9 | RRV | `150_RRV_LF` |
| 10 | RRV | `270_RRV_MCVBB` |
| 11 | RRV | `150_RRV_CVBB` |

> 注：`_hrv_median_nni` 出现了两次（索引 0 和 2），这是训练时的设计，并非笔误。

### 输出

| 维度 | 说明 |
|------|------|
| `batch_size` | 与输入相同 |
| `num_classes` | 固定 4 |

4 个类别的 logits，`argmax` 解码：

| 标签 | 睡眠阶段 |
|------|---------|
| 0 | Wake |
| 1 | Light (N1+N2) |
| 2 | Deep (N3) |
| 3 | REM |

### 内置 BatchNorm

模型 `forward` 的第一步是对输入做 per-batch 标准化：

```python
mean = x.mean(dim=(0, 1))
std  = x.std(dim=(0, 1)) + 1e-8
x = (x - mean) / std
```

这意味着推理时**不需要保证输入已经零均值单位方差**——模型内部会自动处理。但输入仍需经过第一层 StandardScaler（`scaler_params.json`），该 scaler 在训练集上拟合，用于统一不同特征的量纲。

## 推理流程（完整）

```
原始特征 CSV (N epochs × 460 列)
  → 特征选择（取上述 12 列）
  → 滑动窗口构建（居中 padding，窗口长度 21）
  → 第一层 StandardScaler（scaler_params.json）
  → ONNX 模型推理
  → argmax 解码 → 睡眠阶段标签
```

`inference.py` 实现了上述完整流程（不含特征选择，因 `input_features.csv` 已经是选好的 12 列）。

## ONNX 导出方式

从 PyTorch 模型导出使用的关键参数：

```python
torch.onnx.export(
    model,
    dummy_input,                  # torch.randn(1, 21, 12)
    "model.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

- 导出时 `model.eval()`，`use_attention=False`
- 由于模型 forward 中包含 `self.to(device)` 调用，导出时使用了 wrapper 子类跳过了该行（dynamo 会展开所有参数的 `.to()` 操作）
- 权重超过阈值自动外存为 `model.onnx.data`