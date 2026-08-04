# Inference 模块

训练好的 LSTM 睡眠分期模型的推理与 ONNX 部署工具。

## 文件结构

```
inference/
├── data_utils.py              # 框架无关：特征加载、序列构建、scaler 应用、评分
├── engine_torch.py            # PyTorch 推理引擎（对外合作者：只看此文件即可）
├── engine_onnx.py             # ONNX Runtime 推理引擎（对外合作者：只看此文件即可）
├── export_onnx.py             # PyTorch → ONNX 模型导出
├── inference_features.py      # CLI：从预处理特征出发的推理
├── inference_full.py          # CLI：从原始数据出发的端到端推理（含预处理）
├── debug_onnx_diff.py         # 诊断工具：PyTorch vs ONNX 逐层对比
└── README.md                  # 本文档
```

## 快速开始

```bash
cd third_party/sleep_analysis

# 1. 导出 ONNX 模型（训练完成后执行一次）
python sleep_analysis/classification/inference/export_onnx.py \
    --run-dir exports_our/2026-08-03_202132

# 2. PyTorch 推理
python sleep_analysis/classification/inference/inference_features.py \
    --run-dir exports_our/2026-08-03_202132 \
    --subject 0001 \
    --dataset mesa \
    --backend torch \
    --output-dir exports_our/2026-08-03_202132/inference_results/torch

# 3. ONNX 推理
python sleep_analysis/classification/inference/inference_features.py \
    --run-dir exports_our/2026-08-03_202132 \
    --subject 0001 \
    --dataset mesa \
    --backend onnx \
    --output-dir exports_our/2026-08-03_202132/inference_results/onnx
```

## 命令参考

### export_onnx.py — 模型导出

将训练好的 PyTorch 模型导出为 ONNX 格式。

```bash
python export_onnx.py --run-dir <训练输出目录> [--opset 18]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run-dir` | 训练运行目录，需包含 `config.json` 和 `checkpoints/best_model.pt` | 必填 |
| `--output` | ONNX 输出路径 | `{run_dir}/checkpoints/model.onnx` |
| `--opset` | ONNX opset 版本 | 17（导出后实际为 18） |

输出：
- `{run_dir}/checkpoints/model.onnx` — ONNX 模型（单文件）
- `{run_dir}/checkpoints/model.onnx_info.json` — 导出元信息（输入输出 shape 等）

导出后会自动做 PyTorch vs ONNX 的 batch_size=1 精度对比，max_diff < 1e-4 即为通过。

### inference_features.py — 特征路径推理

从已预处理的特征 CSV 出发，加载模型完成推理并与标注对比。

```bash
python inference_features.py \
    --run-dir <训练输出目录> \
    --backend torch|onnx \
    (--subject <被试> --dataset mesa|shhs1|shhs2 | --features <文件> --ground-truth <文件>) \
    [--output-dir <输出目录>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run-dir` | 训练输出目录 | 必填 |
| `--backend` | 推理后端 | `torch` |
| `--subject` | 被试编号，多个用逗号分隔 | — |
| `--dataset` | 数据集名称 | `mesa` |
| `--features` | 直接指定特征 CSV 路径（替代 `--subject + --dataset`） | — |
| `--ground-truth` | 直接指定标注 CSV 路径 | — |
| `--output-dir` | 结果输出目录 | `{run_dir}/inference_results` |

示例：

```bash
# 单被试
python inference_features.py --run-dir <dir> --subject 0001 --dataset mesa

# 批量
python inference_features.py --run-dir <dir> --subject 0001,0002,0006 --dataset mesa

# 自定义文件
python inference_features.py --run-dir <dir> \
    --features /path/to/features.csv --ground-truth /path/to/gt.csv

# ONNX 后端
python inference_features.py --run-dir <dir> --subject 0001 --backend onnx
```

### inference_full.py — 端到端推理

从原始 EDF 数据出发，自动完成预处理 + 特征提取 + 推理 + 对比。

```bash
python inference_full.py \
    --run-dir <训练输出目录> \
    --subject <被试> \
    --backend torch|onnx \
    [--full-pipeline]
```

与 `inference_features.py` 的区别：
- 自动查找特征文件，若不存在可触发完整预处理管线（`--full-pipeline`）
- 仅支持 MESA 数据集

### debug_onnx_diff.py — 诊断工具

PyTorch 和 ONNX 推理结果不一致时，逐层提取中间输出，定位差异来源。

```bash
python debug_onnx_diff.py --run-dir <dir> --subject 0001
```

输出每层的 max_diff 以及中间张量保存为 `.npy` 文件到 `{run_dir}/debug_onnx/`。

## 输出格式

每次推理产出两个文件：

```
{output_dir}/
├── per_subject_predictions/{subject}.csv    # 逐 epoch 预测
└── metrics_{subject}.json                   # 指标汇总
```

**`{subject}.csv`**：

| epoch | prediction | ground_truth |
|-------|-----------|--------------|
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| ... | ... | ... |

sleep stage 编码：`0=Wake, 1=Light, 2=Deep, 3=REM`（4stage 分类）。

**`metrics_{subject}.json`**：

```json
{
  "subject_id": "0001",
  "n_epochs": 1262,
  "classification_type": "4stage",
  "metrics": {
    "accuracy": 0.7916,
    "precision": 0.7792,
    "recall": 0.7916,
    "f1": 0.7792,
    "kappa": 0.6209,
    "mcc": 0.6248,
    "confusion_matrix": [[516, 58, 0, 1], ...]
  }
}
```

## 对外合作者使用

### 仅需 PyTorch 推理

只需两个文件：`engine_torch.py` + `data_utils.py`。

```python
from pathlib import Path
from engine_torch import TorchInferenceEngine

engine = TorchInferenceEngine(Path("exports_our/2026-08-03_202132"))
y_pred, y_true = engine.predict("0001", Path("/path/to/mesa_processed"))
```

### 仅需 ONNX 推理

只需三个文件：`engine_onnx.py` + `data_utils.py` + `model.onnx`。

```python
from pathlib import Path
from engine_onnx import OnnxInferenceEngine

engine = OnnxInferenceEngine(Path("exports_our/2026-08-03_202132"))
y_pred, y_true = engine.predict("0001", Path("/path/to/mesa_processed"))
```

### 自定义数据

如果不使用 MESA/SHHS 的标准目录结构，直接传文件路径：

```python
y_pred, y_true = engine.predict_from_files(
    feature_path=Path("/path/to/features.csv"),
    ground_truth_path=Path("/path/to/ground_truth.csv"),
    label="my_subject",
)
```

输入特征 CSV 需包含所有模态列（HRV、RRV 等），`data_utils.select_features()` 会自动按训练时的模态筛选。

## 依赖

- **PyTorch 引擎**：`torch`, `numpy`, `pandas`, `sklearn`
- **ONNX 引擎**：`onnxruntime`, `numpy`, `pandas`, `sklearn`
- **导出**：`torch`, `onnx`, `onnxscript`, `onnxruntime`

```bash
pip install onnx onnxruntime onnxscript -i https://pypi.tuna.tsinghua.edu.cn/simple
```
