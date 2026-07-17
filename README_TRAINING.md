# LSTM 睡眠分期训练指南

## 环境准备

### 依赖安装

```bash
cd third_party/sleep_analysis
pip install poetry==1.5.1
poetry config virtualenvs.create false
poetry install
```

### PyTorch 版本要求

原 `pyproject.toml` 指定 `torch ^1.12.1`，已在本次更新中改为 `>=2.5.0`。原因是较新的 GPU 架构（Blackwell, sm_120 及以上）需要 PyTorch 2.5+。如果使用旧款 GPU，可降回 `^1.12.1`。

### 环境变量

Linux 无桌面环境（如 HPC 节点）需要在运行前设置：

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

否则 Poetry 的 keyring 会尝试调用 DBus 报错。

---

## 数据预处理

训练前必须完成 MESA 数据预处理。小规模试跑：

```bash
cd third_party/sleep_analysis
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
python experiments/data_handling/preprocess_subset.py 50   # 50 个被试，约 750 MB
```

全量处理：

```bash
python experiments/data_handling/preprocess_subset.py 2056   # 全量，约 15-20 GB
```

预处理产出目录在 `study_data.json` 中配置（`processed_mesa_path` 字段）。

---

## 训练脚本

### 主脚本

```
sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py
```

所有参数通过命令行传入，**不需要修改脚本**。查看完整参数列表：

```bash
python LSTM_paper_params.py --help
```

### 论文参数复现

以下命令假设已激活正确的 Python 环境且设置了必要的环境变量。

#### 5 分类（论文 MESA Baseline）

```bash
python LSTM_paper_params.py \
  -c 5stage \
  --modality ACT HRV RRV \
  --seq-len 21 --hidden 556 --layers 6 \
  --dropout 0.255 --lr 6.31e-5 --batch-size 512
```

| 参数 | 值 | 含义 |
|---|---|---|
| `-c 5stage` | 5 分类 | Wake / N1 / N2 / N3 / REM |
| `--seq-len 21` | 10 min | 21 epoch × 30s |
| `--hidden 556` | 隐层大小 | |
| `--layers 6` | 6 层 LSTM | |
| `--dropout 0.255` | Dropout 率 | |
| `--lr 6.31e-5` | 学习率 | Adam 自适应 |
| `--batch-size 512` | 批次大小 | |

#### 3 分类

```bash
python LSTM_paper_params.py \
  -c 3stage \
  --modality ACT HRV RRV \
  --seq-len 101 --hidden 124 --layers 3 \
  --dropout 0.363 --lr 3.44e-5
```

#### 4 分类（N1+N2 合并为浅睡）

```bash
python LSTM_paper_params.py -c 4stage
```

### 分类类型说明

| 参数 | 类别 | 说明 |
|---|---|---|
| `binary` | Wake / Sleep | 2 分类 |
| `3stage` | Wake / NREM / REM | |
| `4stage` | Wake / Light(N1+N2) / Deep(N3) / REM | |
| `5stage` | Wake / N1 / N2 / N3 / REM | AASM 标准 |

### 模态选择

论文 5-class Baseline 使用 `ACT + HRV + RRV`。`EDR` 是可选扩展。

| 参数 | 特征 | 维度 | 来源 |
|---|---|---|---|
| `ACT` | 体动均值 | 1 | 腕动计 |
| `HRV` | 心率变异性（时域+频域+非线性） | 8 | ECG R-point |
| `RRV` | 呼吸率变异性（5/9 min 窗口） | 4 | 胸腔呼吸带 |
| `EDR` | ECG 衍生呼吸率变异性 | 4 | ECG 提取的呼吸信号 |

### 快速验证

```bash
# 20 人 + 3 epoch 快速验证管线
python LSTM_paper_params.py -c 5stage --small --quick
```

### 可调超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--focal-gamma` | 2.0 | Focal Loss 聚焦参数，N3 召回低时可调大到 3.0 |
| `--grad-clip` | 0.5 | 梯度裁剪阈值 |
| `--weight-decay` | 1e-5 | L2 正则化（硬编码在 LSTM.py 里） |
| `--seed` | 42 | 随机种子 |

---

## 后台训练（screen）

```bash
screen -S lstm_5stage
cd third_party/sleep_analysis

LOG_FILE="exports_our/train_5stage_$(date +%Y-%m-%d_%H%M%S).log"

PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
  -c 5stage --modality ACT HRV RRV \
  --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 \
  2>&1 | tee "$LOG_FILE"

# Ctrl+A D 断开   |   screen -r lstm_5stage 回来看   |   screen -ls 列出所有会话
```

---

## 评测已保存的模型

每个训练产出目录结构：

```
exports_our/2026-07-16_193807/
├── config.json              # 训练参数（自文档化）
├── checkpoints/
│   ├── best_model.pt        # 最佳模型（验证 loss 最低）
│   ├── ckpt_epoch_005_acc0.5234_k0.4521_mcc0.4703.pt
│   └── ...                  # 每 5 epoch 存一个
├── per_subject_predictions/ # 每个被试逐 epoch 预测
│   ├── 0001.csv
│   └── ...
└── results/
    ├── per_subject_metrics.csv
    ├── confusion_matrix.csv              # 原始计数
    ├── confusion_matrix_percent.csv       # 百分比（对角线=召回率）
    ├── results.json                      # 所有指标汇总
    └── predictions.pickle
```

### 评估已有模型

```bash
# 自动从 config.json 恢复超参数，跳过训练直接评估
python LSTM_paper_params.py \
  --load-weights exports_our/2026-07-16_193807/checkpoints/best_model.pt \
  --eval-only

# 小规模快速验证（20 人）
python LSTM_paper_params.py \
  --load-weights exports_our/2026-07-16_193807/checkpoints/best_model.pt \
  --eval-only --small
```

`--load-weights` 会自动读取同次训练的 `config.json`，恢复 `hidden_size`、`num_layers`、`classification` 等全部超参数，无需手动指定。

---

## 训练输出说明

### 混淆矩阵

行=真实标签，列=模型预测。对角线即召回率。

```
# 5 分类示例（百分比）
         wake    n1     n2    n3    rem
wake     79.3   3.5   10.8   0.2    6.2
n1       22.0  13.4   45.9   0.2   18.5
n2        7.0   3.3   78.7   3.6    7.4
n3        1.7   0.3   76.7  19.4    1.9
rem      10.5   2.6   19.6   0.1   67.2
```

### 指标说明

| 指标 | 含义 |
|---|---|
| accuracy | 所有样本中预测正确的比例 |
| precision | 加权精确率 |
| recall | 加权召回率（多分类中 = accuracy） |
| f1 | 加权 F1 分数 |
| kappa | Cohen's Kappa，修正随机一致 |
| mcc | Matthews Correlation Coefficient，论文优化目标 |
| specificity | 加权特异度 |

---

## 注意事项

### GPU 兼容性

- RTX 5060 (Blackwell, sm_120) **必须用 PyTorch ≥ 2.5**，`sleep_analysis` 环境里的 1.13.1 不支持
- 如果 CUDA OOM，先降 batch_size（`--batch-size 128`）

### 早停

代码有 patience=5 的早停机制。验证 loss 连续 5 epoch 不降即停止。论文的 "up to 170 epochs" 是上限，实际收敛 epoch 数取决于数据和参数。

### Adam 学习率

学习率设置后不会自动衰减。Adam 通过 `m/√v` 比值自适应地自然缩小有效步长，不需要 scheduler。

### 数据划分

训练/验证/测试按被试级别划分（80%/ 20%→16%），确保同一被试的数据不会同时出现在训练集和验证/测试集中。

### 实时分期

当前模型使用 centered sliding window（窗口包含未来数据），不适合实时分期。如需实时，修改 `data_peparation.py` 的 padding 为仅左侧。

### 已知问题

- `ecg.py`: 修复了新版 pandas `value_counts().reset_index()` 列名兼容
- `hrv.py`, `rrv.py`: 修复了 `Path(__file__).parents[N]` 层级错误
- `model.py`: 正交初始化在 CPU 上执行以兼容 CUDA 13.0
- `LSTM.py` (line 354): 修了变量名 `mean_performance` → `mean_mcc`
