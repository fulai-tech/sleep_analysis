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

训练前必须完成数据预处理。处理模式由环境变量 `SLEEP_CAUSAL` 控制：

- 不设置（默认）：原版处理（复现作者/论文结果，特征含未来信息）
- `SLEEP_CAUSAL=1`：**因果处理**（实时睡眠分期用，所有处理只使用已见数据）

⚠️ 训练实时模型必须用 `SLEEP_CAUSAL=1` 生成的数据。两种模式产出的特征不同，不能混用。

### MESA

```bash
cd third_party/sleep_analysis

# 小规模试跑 (2 个被试)
SLEEP_CAUSAL=1 \
python experiments/data_handling/preprocess_subset.py 2 --no-edr --n-workers 4 \
    --output-dir /srv/shared/psgdata/xxx/mesa_processed

# 全量 (因果模式 + 屏蔽 EDR + 10 worker 并行)
SLEEP_CAUSAL=1 \
python experiments/data_handling/preprocess_subset.py 2056 --no-edr --n-workers 10 \
    --output-dir /srv/shared/psgdata/xxx/mesa_processed
```

### SHHS1 / SHHS2

```bash
SLEEP_CAUSAL=1 \
python experiments/data_handling/preprocess_shhs.py --study shhs1 --n-subjects 99999 --no-edr --n-workers 10 \
    --output-dir /srv/shared/psgdata/xxx/shhs1_processed
# shhs2 同理 (--study shhs2)
```

### 参数说明

- `SLEEP_CAUSAL=1`：因果处理（RRV 信号因果滤波/降采样 + 左对齐回顾窗口）
- `--no-edr`：屏蔽 EDR 特征提取（EDR 已弃用，占位全 0，节省计算）
- `--output-dir`：输出目录（默认 study_data.json 配置的路径）
- `--n-workers`：并行 worker 数

⚠️ **换模式必须用新的 `--output-dir`**：输出目录有 run_config.json 模式锁（`check_run_mode`），跨模式续跑会被拒绝，防止静默复用旧模式的文件。

预处理产出目录在 `study_data.json` 中配置（`processed_mesa_path` / `shhs1_processed_path` / `shhs2_processed_path`）。

---

## 训练脚本

### 主脚本

```
sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py
```

### 数据集选择

| 参数 | 数据集 | 体动 | 可用模态 | 预处理脚本 |
|---|---|---|---|---|
| `-d MESA_Sleep` | MESA | 有 | ACT+HRV+RRV | `preprocess_subset.py` |
| `-d SHHS1` | SHHS1 | 无 | HRV+RRV | `preprocess_shhs.py --study shhs1` |
| `-d SHHS2` | SHHS2 | 无 | HRV+RRV | `preprocess_shhs.py --study shhs2` |
| `-d MESA_Sleep+SHHS1+SHHS2` | 混合 | 部分 | HRV+RRV | 上述预处理分别完成后 |

不指定 `-d` 时默认为 `MESA_Sleep`。SHHS / 混合数据集不指定 `--modality` 时自动使用 `HRV RRV`（无体动数据）。如果误传 `--modality ACT` 到 SHHS，脚本会打印警告并自动移除。

EDR 已弃用（2026-08 起），不再参与训练。

所有参数通过命令行传入，**不需要修改脚本**。查看完整参数列表：

```bash
python LSTM_paper_params.py --help
```

### 实时模型训练（--causal）

实时睡眠分期要求特征数据和模型窗口都是因果的（只用已见数据）。训练实时模型：

```bash
python LSTM_paper_params.py \
  -d MESA_Sleep -c 4stage --causal \
  --modality ACT HRV RRV \
  --seq-len 21 --hidden 556 --layers 6 \
  --dropout 0.255 --lr 6.31e-5 --batch-size 512 \
  --split-file splits/split_mesa_1121_20260806.json
```

- `--causal`：序列窗口仅左侧 padding（窗口覆盖 [t-20, t]，不含未来）。config.json 记录 `causal=true`，推理引擎自动对齐
- 论文参数（无 `--causal`）为 centered 窗口（含未来 5 分钟），仅用于复现/消融对比

### 数据划分（--split-file）

训练/验证/测试按被试级别划分。为跨实验一致性，可用固定 split 名单：

```bash
# MESA 单独训练
--split-file splits/split_mesa_1121_20260806.json

# 混合训练 (MESA+SHHS1+SHHS2)
--split-file splits/split_mixed_20260808.json
```

- 单数据集格式：`{"train": [...], "val": [...], "test": [...]}`
- 多数据集格式（混合训练）：`{"mesasleep": {...}, "shhs1": {...}, "shhs2": {...}}`，按数据集前缀匹配各自名单
- SHHS1/2 的划分按 nsrrid 联合生成（同一参与者不会跨 train/val/test），由 `experiments/data_handling/make_splits.py` 生成
- 不传 `--split-file` 时用代码内自动划分（80/20 → 80/20）

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

| 参数 | 特征 | 维度 | 来源 |
|---|---|---|---|
| `ACT` | 体动均值 | 1 | 腕动计 |
| `HRV` | 心率变异性（时域+频域+非线性） | 8 | ECG R-point |
| `RRV` | 呼吸率变异性（150/270s 回顾窗口） | 4 | 胸腔呼吸带 |
| ~~`EDR`~~ | ~~ECG 衍生呼吸率变异性~~ | ~~4~~ | ~~已弃用 (2026-08)~~ |

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
| `--shuffle-mode` | `none` | 训练时打乱样本顺序的方式：`none`=不打乱；`subject`=每轮训练换一批被试的顺序（推荐）；`sample`=每轮把所有样本彻底打乱（不推荐）。详见下文「训练数据顺序」 |
| `--wake-weight` | 1.0 | 只放大 wake（类 0）的 loss 权重：最终权重 = `(1-freq)×wake_weight`，其余类不变。如 wake:睡眠=3:7 想拉平可试 7/3≈2.33。Adam 对 loss 全局缩放近似不变，一般无需降 lr，震荡明显再降 |
| `--missing-mode` | 关 | 混合训练时数据集缺某模态的处理：**关闭**=原行为（ACT 自动剔除；HRV/RRV 缺失报错，不能训练）；**开启**=缺失模态特征**填 0**（先填 0 再标准化），并为 modality 列表里**每个模态**增加 `_has_<模态>` 标志列（0/1，布局统一：`[ACT..., _has_act, HRV..., _has_hrv, RRV..., _has_rrv]`）。填 0 使缺失数据与有数据集的低活动期（MESA 的 ACT 78% 为 0）重合，数值通道不泄露数据集身份，模型只能依赖 `_has_*` 标志。⚠️ 模态缺失检测目前用硬编码正则（`_acc`/`_hrv`/`RRV`），新数据集若列名不同需同步调整 |

### 训练数据顺序（--shuffle-mode）

训练时每轮（epoch）喂给模型的样本顺序不同，最终效果也不同。三种模式：

- **`none`（默认）**：样本顺序完全固定。同样参数下两次训练结果一致，用于复现历史实验（如 08-17 基线）。
- **`subject`（推荐）**：每轮训练重新排列被试顺序，但同一个被试内部的窗口保持连续。实测整体指标最好（4stage 测试 mcc 0.591 vs 固定序 0.566），代价是 N3（deep）召回率明显下降（35% → 13%），若 N3 检测是重点需注意。
- **`sample`（不推荐）**：每轮把所有样本彻底打乱。实测模型拟合过快、很快开始过拟合，验证集效果几个 epoch 后就变差，不建议使用。

**实现说明**：打乱用 numpy 在 CPU 上完成，随机种子为 `seed + epoch`，所以同样参数下两次训练结果完全一致（换机器也一致）。不要用 `torch.randperm` 在 GPU 上打乱——不同型号 GPU 上生成的随机序列不同，会导致结果无法复现。

---

## 后台训练（screen）

```bash
screen -S lstm_5stage
cd third_party/sleep_analysis

LOG_FILE="exports_our/train_5stage_$(date +%Y-%m-%d_%H%M%S).log"

PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
python -u sleep_analysis/classification/algorithm_scripts/LSTM_paper_params.py \
  -c 5stage --modality ACT HRV RRV --causal \
  --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 \
  2>&1 | tee "$LOG_FILE"

# Ctrl+A D 断开   |   screen -r lstm_5stage 回来看   |   screen -ls 列出所有会话
```

---

## 评测已保存的模型

每个训练产出目录结构：

```
exports_our/2026-07-16_193807/
├── config.json              # 训练参数（自文档化，含 causal 标志）
├── study_data.json          # 训练时的数据路径快照（数据来源可追溯）
├── checkpoints/
│   ├── best_model.pt        # 最佳模型（验证 loss 最低）
│   ├── scaler.json          # 外部归一化参数（训练集拟合，推理必需）
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

`study_data.json` 是训练时的数据路径快照（`processed_mesa_path` / `shhs*_processed_path` 等）——训练代码会自动复制到产出目录，记录本次训练读取了哪份数据（训练后 study_data.json 路径再更改也不影响追溯）。

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

`--load-weights` 会自动读取同次训练的 `config.json`，恢复 `hidden_size`、`num_layers`、`classification`、`causal` 等全部超参数，无需手动指定。

---

## 推理引擎（新数据推理）

训练好的模型可用于对新被试推理（特征 CSV → 预测）。引擎自动从 run 目录读取 config.json（causal 等）和 checkpoints/scaler.json（归一化），无需手动传参。

```bash
# 导出 ONNX（可选，部署用）
python sleep_analysis/classification/inference/export_onnx.py \
    --run-dir exports_our/<timestamp>

# PyTorch 引擎推理
python sleep_analysis/classification/inference/inference_features.py \
    --run-dir exports_our/<timestamp> --subject <ID> --backend torch

# ONNX 引擎推理
python sleep_analysis/classification/inference/inference_features.py \
    --run-dir exports_our/<timestamp> --subject <ID> --backend onnx
```

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

训练/验证/测试按被试级别划分（80%/ 20%→16%），确保同一被试的数据不会同时出现在训练集和验证/测试集中。跨实验一致性用 `--split-file` 固定名单（见上文）。

### 实时分期

实时睡眠分期已支持（2026-08 完成改造）。**实时可用模型需同时满足两个开关**：

1. **数据因果**（`SLEEP_CAUSAL=1` 生成特征，记录在输出目录 run_config.json）：
   - RRV 信号滤波/降采样：filtfilt（零相位双向）→ lfilter（因果正向）
   - RRV 窗口：伪居中（含未来 1-2 分钟）→ 左对齐回顾窗口（epoch j 只依赖 [j-W+1, j] 的已见数据）
   - EDR 已弃用（`--no-edr`）
   - 已知保留项（团队决策）：HRV 处理（process_rpoint）与 R 点检测保持原版——HRV 特征实测差异中位 0%，保持与原版模型可比性；causal 实现在代码中注释保留（`rr_utils.py` / `ecg_rpeaks.py`）
2. **模型因果**（`--causal` 训练，config.json 的 `causal=true`）：
   - 序列窗口仅左侧 padding（不含未来）
   - 已移除模型内部整夜归一化（只保留训练集 scaler——训练/推理一致、无未来依赖、逐窗口推理可行）
   - 序列 padding 用首值填充（原为整夜均值，含未来）

两个开关相互独立：`SLEEP_CAUSAL` 控制数据生成（预处理阶段），config.json 的 `causal` 控制模型窗口（训练/推理阶段）。详见 `sleep_analysis/processing_config.py` 顶部说明。

#### stateful 模式（2026-08-11 新增）

无状态滑窗推理每预测一帧要重算整个 seq_len 窗口（~21 倍冗余算力），且模型上下文被窗口长度限制。**stateful 模式**改为逐 epoch 扫描：

- **训练**：`--stateful`（需与 `--causal` 一起，校验在 config restore 之后——`--load-weights <causal-run> --stateful` 可用）。逐帧携带 LSTM (h, c) 状态，用最近 seq_len 个 h 的滚动缓冲做均值池化；truncated BPTT 训练（`--state-chunk C`，默认 64 帧截断，detach 梯度；P = batch_size // C 个被试并行）
- **推理**：每帧 O(1)（1 步 LSTM 展开 + 缓冲池化，无窗口重算），上下文不受 seq_len 限制；config.json 的 `stateful=true` 驱动两个推理引擎自动走扫描路径
- **冷启动**：每夜开始前用首帧特征从零状态 warm-up seq_len-1 次填充缓冲——与无状态 edge-pad 语义一致（epoch 0 与无状态窗口逐位一致，epoch ≥ 1 额外携带前缀上下文，是状态化的目的）；训练/推理同一协议，无未来泄漏
- **ONNX**：导出单步 scan 图（4 输入 `x,h,c,buf` → 4 输出 `logits,h_n,c_n,buf_out`，动态 batch），引擎逐帧 `session.run` 并维护状态
- 无状态路径（默认）完全不变；旧模型与旧 ONNX 图继续可用

```bash
# 训练 stateful 模型
python LSTM_paper_params.py -c 5stage --causal --stateful \
  --seq-len 21 --hidden 556 --layers 6 --dropout 0.255 --lr 6.31e-5 \
  --split-file splits/split_mesa_1121_20260806.json

# 导出 stateful ONNX + 推理 (引擎自动按 config 走 stateful 路径)
python sleep_analysis/classification/inference/export_onnx.py --run-dir exports_our/<timestamp>
python sleep_analysis/classification/inference/inference_features.py \
    --run-dir exports_our/<timestamp> --subject <ID> --backend torch|onnx
```

### 已知问题 / 决策记录

- `rrv.py`: RRV 因果改造（滤波 + 左对齐窗口），causal 分支由 `SLEEP_CAUSAL` 控制
- `rr_utils.py`: process_rpoint 已抽为 MESA/SHHS 共用；causal 实现注释保留（HRV 决策：保持原版可比性）
- `ecg.py`: 转发 `rr_utils.process_rpoint`
- `model.py`: 已移除内部 per-batch 归一化（外部 scaler 作为唯一归一化，注释保留可回退）；正交初始化在 CPU 上执行以兼容 CUDA
- `LSTM_paper_params.py`: `--split-file` 支持单数据集/多数据集（混合训练）两种格式
- `hrv.py`, `rrv.py`: 修复了 `Path(__file__).parents[N]` 层级错误
- `model.py`: 新增 `forward_stateful`（单帧步进，显式状态，可导出）；池化必须取 (buf, h_n) 共 seq_len 个值（buf_out 差一）
- `data_peparation.py`: 特征选择抽为 `_extract_subj_features_raw`（get_data / get_frame_data 共用）；`get_frame_data` 复用窗口路径拟合的训练集 scaler（不可自拟合）
- `LSTM.py`: stateful 分支（`_stateful_groups` / `_stateful_warmup` / `_stateful_build_chunk` / `_stateful_chunk_step` / `_stateful_scan`）
- `engine_torch.py` / `engine_onnx.py`: 按 config `stateful` 分派扫描路径；ONNX 引擎改用输入/输出名映射（stateful 图 4 入 4 出）
