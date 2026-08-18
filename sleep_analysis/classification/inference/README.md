# Inference 模块

LSTM 睡眠分期模型的推理工具（PyTorch / ONNX Runtime 双后端）。

## 文件结构

```
inference/
├── data_utils.py              # 框架无关：特征加载、序列构建、scaler 应用、评分
├── engine_torch.py            # PyTorch 推理引擎
├── engine_onnx.py             # ONNX Runtime 推理引擎
├── inference_features.py      # CLI：从预处理特征出发的推理
├── inference_full.py          # CLI：端到端推理（含预处理）
├── evaluate.py                # 测试集批量评估
└── onnx_pipeline/             # 对外交付包（独立、无项目依赖）
```

## 快速开始

```bash
cd third_party/sleep_analysis

# PyTorch 推理
python sleep_analysis/classification/inference/inference_features.py \
    --run-dir <run-dir> \
    --subject 0001 \
    --dataset mesa \
    --backend torch

# ONNX 推理
python sleep_analysis/classification/inference/inference_features.py \
    --run-dir <run-dir> \
    --subject 0001 \
    --dataset mesa \
    --backend onnx
```

`<run-dir>` 为模型输出目录（包含 `config.json`、`checkpoints/best_model.pt`、`checkpoints/scaler.json`；ONNX 推理还需 `checkpoints/model.onnx`）。

> **旧模型（2026-08-07 之前的版本）** 的模型内部含第二层归一化，推理时需加
> `--night-norm` 以在模型外用整夜数据完成该归一化（结果与参考结果一致）。

## 命令参考

### inference_features.py — 特征路径推理

从已预处理的特征 CSV 出发，加载模型完成推理并与标注对比。

```bash
python inference_features.py \
    --run-dir <run-dir> \
    --backend torch|onnx \
    (--subject <被试> --dataset mesa|shhs1|shhs2 | --features <文件> --ground-truth <文件>) \
    [--output-dir <输出目录>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run-dir` | 模型输出目录 | 必填 |
| `--backend` | 推理后端 | `torch` |
| `--subject` | 被试编号，多个用逗号分隔 | — |
| `--dataset` | 数据集名称 | `mesa` |
| `--features` | 直接指定特征 CSV 路径（替代 `--subject + --dataset`） | — |
| `--ground-truth` | 直接指定标注 CSV 路径 | — |
| `--output-dir` | 结果输出目录 | `{run_dir}/inference_results` |
| `--night-norm` | 模型外整夜归一化（旧模型需要） | 关闭 |

示例：

```bash
# 单被试
python inference_features.py --run-dir <dir> --subject 0001 --dataset mesa

# 批量
python inference_features.py --run-dir <dir> --subject 0001,0002,0006 --dataset mesa

# 自定义文件
python inference_features.py --run-dir <dir> \
    --features /path/to/features.csv --ground-truth /path/to/gt.csv
```

### inference_full.py — 端到端推理

从原始 EDF 数据出发，自动完成预处理 + 特征提取 + 推理 + 对比。

```bash
python inference_full.py \
    --run-dir <run-dir> \
    --subject <被试> \
    --backend torch|onnx \
    [--full-pipeline]
```

与 `inference_features.py` 的区别：
- 自动查找特征文件，若不存在可触发完整预处理管线（`--full-pipeline`）
- 仅支持 MESA 数据集

### evaluate.py — 测试集批量评估

对测试集批量推理 + 评分，可与参考指标对比。

```bash
python evaluate.py --run-dir <run-dir> [--dataset mesasleep] [--backend onnx] [--night-norm]
```

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

## 引擎 API

### PyTorch

```python
from pathlib import Path
from engine_torch import TorchInferenceEngine

engine = TorchInferenceEngine(Path("<run-dir>"))
y_pred, y_true = engine.predict("0001", Path("/path/to/mesa_processed"))
```

### ONNX Runtime

```python
from pathlib import Path
from engine_onnx import OnnxInferenceEngine

engine = OnnxInferenceEngine(Path("<run-dir>"))
y_pred, y_true = engine.predict("0001", Path("/path/to/mesa_processed"))
```

### 自定义数据

不使用标准目录结构时，直接传文件路径：

```python
y_pred, y_true = engine.predict_from_files(
    feature_path=Path("/path/to/features.csv"),
    ground_truth_path=Path("/path/to/ground_truth.csv"),
    label="my_subject",
)
```

输入特征 CSV 的列需与模型要求的特征一致（引擎会自动校验）。

## 依赖

- **PyTorch 引擎**：`torch`, `numpy`, `pandas`, `sklearn`
- **ONNX 引擎**：`onnxruntime`, `numpy`, `pandas`, `sklearn`

## 对外交付

给外部合作者的独立交付包见 [onnx_pipeline/](onnx_pipeline/README.md)：
模型、scaler、示例输入、测试集、推理与评估脚本全部自包含，无项目依赖。
