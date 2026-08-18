# ONNX Sleep Staging Model (交付包)

基于 LSTM 的睡眠分期 ONNX 模型，供外部部署与模型转换使用。

模型来源：exports_our/exports_our_debug_realtime/2026-08-03_202132

训练数据：processed_data_with_leak_20260804

## 文件清单

| 文件 | 说明 |
|------|------|
| `inference.py` | ONNX Runtime 推理示例（独立，无 sleep_analysis 依赖） |
| `evaluate.py` | 测试集批量评估（推理 + 与标注对比 + 汇总指标） |
| `model.onnx` | ONNX 模型（单文件，IR v9，opset 18，~53 MB） |
| `input_features.csv` | 示例输入特征（已选好 12 列，1262 epochs），用于验证推理 |
| `scaler_params.json` | 第一层 StandardScaler 参数 + 推理配置 |
| `test_subjects.csv` | 测试集清单（225 被试，相对路径），供 `evaluate.py` 使用 |
| `test_set/features/` | 测试集特征（225 个被试，已选好 12 列，~30 MB） |
| `test_set/ground_truth/` | 测试集标注（225 个被试，~30 MB） |
| `predictions.csv` | `inference.py` 运行后的输出 |
| `requirements.txt` | Python 依赖 |

## 快速验证

```bash
# 单被试推理
python inference.py          # 输出 predictions.csv (1262 epochs)

# 测试集批量评估 (225 被试)
python evaluate.py --subjects-file test_subjects.csv
# 可选: 提供参考指标文件, 输出对比结果
python evaluate.py --subjects-file test_subjects.csv \
    --reference per_subject_metrics.csv
```

## 环境要求

- Python ≥ 3.10
- `numpy<2`, `pandas`, `scikit-learn`, `onnxruntime>=1.15`（见 `requirements.txt`）
  - **numpy 必须 <2**：onnxruntime 1.15/1.16 等版本编译自 NumPy 1.x，
    与 NumPy 2.x ABI 不兼容（报错 `numpy.core.multiarray failed to import` / 段错误）
  - `evaluate.py` 的指标计算依赖 scikit-learn；仅用 `inference.py` 单被试推理可不装
- **兼容 onnx 1.15 转换环境**（模型 IR version = 9，opset = 18；opset 18 需要 onnxruntime ≥ 1.14，1.15 完全支持）

```bash
pip install -r requirements.txt
```

## 模型输入输出

### 输入

ONNX 模型输入名 `input`，形状 `(batch_size, 21, 12)`：

| 维度 | 说明 |
|------|------|
| `batch_size` | 动态，任意值 |
| `seq_len` | 固定 21（21 个连续 30s epoch） |
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

> 注：`_hrv_median_nni` 出现两次（索引 0 和 2），这是模型的设计，并非笔误。

`input_features.csv` 与 `test_set/features/` 中的特征即为上述 12 列。

### 输出

ONNX 模型输出名 `output`，形状 `(batch_size, 4)` 的 logits，`argmax` 解码：

| 标签 | 睡眠阶段 |
|------|---------|
| 0 | Wake |
| 1 | Light (N1+N2) |
| 2 | Deep (N3) |
| 3 | REM |

## 测试集评估

`evaluate.py` 对测试集清单中的每个被试执行固定流程（推理 + 与标注对比 + 指标汇总），输出：

- `evaluate_results/summary.json` — 汇总指标（per-subject mean）
- `evaluate_results/per_subject_metrics.csv` — 逐被试指标（accuracy / precision / recall / f1 / kappa / specificity / mcc / confusion_matrix）

### 预期结果（测试集 225 被试）

| 指标 | 值 |
|------|-----|
| accuracy | 0.7483 |
| precision | 0.7684 |
| recall | 0.7483 |
| f1 | 0.7379 |
| kappa | 0.5910 |
| specificity | 0.8583 |
| mcc | 0.6041 |

> 测试集数据已随包导出（`test_set/`），`test_subjects.csv` 使用包内
> 相对路径，外部环境无需访问原始数据路径，直接运行即可复现上述结果。
