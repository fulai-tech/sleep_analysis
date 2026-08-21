import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from sleep_analysis.classification.deep_learning.dl_scoring import dl_score, tensor_to_performance
from sleep_analysis.classification.deep_learning.lstm.model import Model
from sleep_analysis.classification.deep_learning.utils import get_num_classes
from sleep_analysis.datasets.mesadataset import *

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with Class Weights for Multi-Class Classification.

    This loss function addresses class imbalance by down-weighting easy examples and focusing on
    hard-to-classify samples. Additionally, it incorporates class weights to balance label distributions.

    Attributes:
    - class_weights (torch.Tensor): Tensor of shape (num_classes,) containing weights for each class.
    - gamma (float): Focusing parameter to adjust loss weight based on prediction confidence (default=2.0).
    - reduction (str): Specifies the reduction method, either "mean" (default) or "sum".
    """

    def __init__(self, class_weights, gamma=2.0, reduction="mean"):
        """
        Initializes the WeightedFocalLoss function.

        Parameters:
        - class_weights (torch.Tensor): Class weights to handle imbalanced data.
        - gamma (float, optional): Focusing parameter (default=2.0).
        - reduction (str, optional): Reduction mode, either "mean" or "sum" (default="mean").
        """
        super(WeightedFocalLoss, self).__init__()

        # Ensure class weights do not contain NaN, Inf, or zero values
        self.class_weights = torch.nan_to_num(class_weights, nan=1.0, posinf=1.0, neginf=1.0)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Computes the Focal Loss with Class Weights.

        Parameters:
        - inputs (torch.Tensor): Model logits (before softmax), shape [batch_size, num_classes].
        - targets (torch.Tensor): Ground truth labels, shape [batch_size].

        Returns:
        - torch.Tensor: Computed focal loss.
        """

        # Compute log probabilities safely, avoiding log(0) which produces -inf
        log_probs = F.log_softmax(inputs, dim=-1).clamp(min=-100)

        # Compute softmax probabilities safely, avoiding exp overflow
        probs = torch.exp(log_probs).clamp(min=1e-8, max=1.0)

        # Select log probabilities and probabilities corresponding to the target labels
        log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Compute the focal weight, reducing the impact of easy-to-classify examples
        focal_weight = (1 - probs) ** self.gamma

        # Apply class weighting
        alpha_weight = self.class_weights[targets]  # Extract class weight per sample
        focal_weight = focal_weight * alpha_weight  # Multiply with class weights

        # Compute final loss
        loss = -focal_weight * log_probs

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LSTM:
    def __init__(
        self,
        num_epochs,
        learning_rate,
        input_size,
        hidden_size,
        num_layers,
        seq_len,
        dropout,
        batch_size,
        modality,
        dataset_name,
        classification_type="binary",
        output_dir=None,
        focal_gamma=2.0,
        grad_clip=0.5,
        val_sources=None,
        stateful=False,
        state_chunk=64,
        patience=5,
        inv_freq=False,
        internal_norm=False,
        chunk_len=1,
        ensemble_m=1.0,
    ):
        torch.manual_seed(seed=42)
        torch.cuda.manual_seed(seed=42)
        torch.cuda.manual_seed_all(seed=42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # parameters of LSTM
        self.num_epochs = num_epochs  # number of epochs
        self.learning_rate = learning_rate  # set learning rate
        self.input_size = input_size  # number of features
        self.hidden_size = hidden_size  # number of features in hidden state
        self.num_layers = num_layers  # number of stacked lstm layers
        self.seq_len = seq_len  # sequence length of input sequence
        self.dropout=dropout

        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.modality = modality
        self.dataset_name = dataset_name
        self.classification_type = classification_type
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parents[4] / "exports_our"
        self.focal_gamma = focal_gamma
        self.grad_clip = grad_clip
        self.val_sources = val_sources
        self.stateful = stateful          # ✅2026-08-11: 有状态模式 (逐帧扫描 + truncated BPTT)
        self.state_chunk = state_chunk    # truncated BPTT 截断长度 C; P = batch_size // C
        self.patience = patience          # ✅2026-08-13: 早停耐心 (stateful 摆动周期长时可调大)
        self.inv_freq = inv_freq          # ✅2026-08-13: 类权重 1/freq (默认 1-freq)
        self.internal_norm = internal_norm  # ✅2026-08-14: 旧版内部 per-batch 归一化 (复现 08-06 基线用)
        self.chunk_len = chunk_len        # ✅2026-08-19: action-chunking 块长度 k (k=1 退化单点)
        self.ensemble_m = ensemble_m      # ✅2026-08-19: 时间集成指数权重参数 w_j=exp(-m*j), m=0 均匀

        if self.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def train(self, x_train, y_train, x_val, y_val, retrain = False):
        # get number of classes
        num_classes = get_num_classes(self.classification_type)

        # ✅2026-08-11 - rdwang: stateful 模式下 x_train/x_val 传 frames list
        # [(x_s (n,F), y_s (n,), subj_id)], y_train/y_val 传 None
        if self.stateful:
            frames_train = x_train
            frames_val = x_val
            # 每 epoch 一标签, 与窗口计数一致
            y_count = torch.cat([y for _, y, _ in frames_train]).squeeze()
        else:
            frames_train = frames_val = None
            # ✅2026-08-19: y_train (N, k) 块标签 — 每个真实 epoch 被 k 个头重复监督,
            #   类别比例不变, 展平统计即可 (与 k=1 行为一致)
            y_count = y_train.reshape(-1)

        # load batched data
        if self.stateful:
            x_batch_train_list = y_batch_train_list = None  # chunk 调度在 epoch 循环内
        else:
            x_batch_train_list, y_batch_train_list = self.batch_loader(x_train, y_train)

        # calculate class weights
        class_labels, class_sample_count = np.unique(y_count.detach().numpy(), return_counts=True)
        class_sample_count = class_sample_count[np.argsort(class_labels)]

        print(f"class_labels_count: {class_labels}")
        print(f"class_sample_count: {class_sample_count}")
        print(f"len(y_train): {len(y_count.detach().numpy())}")

        weight_classes = class_sample_count / len(y_count.detach().numpy())
        class_weights = torch.tensor(weight_classes).float()
        if self.inv_freq:
            # ✅2026-08-13: 1/freq 类权重 (论文公式) — 少数类惩罚远重于 1-freq。
            # 动机: stateful 的"时间捷径"把深夜 deep 献祭 (预测率坍塌远超真实分布),
            # 1/freq 让抑制 deep 的代价放大一个量级 (MESA 4stage: deep 权重 0.94 → ~17)
            class_weights_cpu = 1.0 / class_weights.clamp(min=1e-6)
        else:
            class_weights_cpu = 1 - class_weights       # 先在 CPU 算好

        print(f"class_weights: {class_weights_cpu.tolist()}", flush=True)

        class_weights = class_weights_cpu.to(self.device)   # 最后才搬上 GPU

        # load empty model of our lstm class that needs to be trained
        lstm = self._load_empty_model(use_gpu=self.use_gpu, num_classes=num_classes, dropout=self.dropout, use_attention=False)

        if retrain:
            print("Finetune MESA model with Radar dataset")
            # load best model obtained from training
            self._load_best_mesa_model(lstm)
            lstm = lstm.cuda()

        if self.classification_type == "binary":
            print("set binary cross-entropy loss for binary classification", flush=True)
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            print("set focal loss for multi-class classification", flush=True)
            # If N1 & REM still underperform, tune gamma (try values like 1.5, 2.5).
            criterion = WeightedFocalLoss(class_weights=class_weights, gamma=self.focal_gamma, reduction="mean")

        if self.stateful:
            # stateful 需要逐样本 loss 做组内 active 掩码 → reduction="none"
            if self.classification_type == "binary":
                criterion_none = torch.nn.BCEWithLogitsLoss(reduction="none")
            else:
                criterion_none = WeightedFocalLoss(class_weights=class_weights, gamma=self.focal_gamma, reduction="none")

        print("learning rate", self.learning_rate)
        print("batch size", self.batch_size)
        print("num layer", self.num_layers)
        print("hidden size", self.hidden_size)


        optimizer = torch.optim.Adam(lstm.parameters(), lr=self.learning_rate, weight_decay=1e-5)


        # initialize variables to determine and save best model in val-procedure
        min_val_loss = 9999999
        max_val_performance = 0.0

        patience_threshold = self.patience
        min_epochs = 5
        patience_counter = 0


        for epoch in range(self.num_epochs):
            train_losses = []

            if self.stateful:
                # ---- stateful: 逐组逐 chunk 扫描 (truncated BPTT) ----
                lstm.train()
                for group in self._stateful_groups(frames_train):
                    # warm-up 在组起点 (带图, chunk 0 不 detach; 镜像无状态首窗口全图梯度)
                    h, c, buf = self._stateful_warmup(lstm, group)
                    max_len = max(t[0].shape[0] for t in group)
                    for chunk_start in range(0, max_len, self.state_chunk):
                        x_chunk, y_chunk, active = self._stateful_build_chunk(
                            group, chunk_start, self.device
                        )
                        h, c, buf, _ = self._stateful_chunk_step(
                            lstm, criterion_none, optimizer, x_chunk, y_chunk, active,
                            h, c, buf, detach_state=(chunk_start > 0), train_losses=train_losses,
                        )
            else:
                # iterate over all batches of training data
                for x_batch_train, y_batch_train in zip(x_batch_train_list, y_batch_train_list):
                    if self.use_gpu:
                        x_batch_train = x_batch_train.to(device=self.device)
                        y_batch_train = y_batch_train.to(device=self.device)
    
                   # for name, param in lstm.named_parameters():
                   #     if torch.isnan(param).any():
                   #         print(f"[WARNING] NaN detected in {name}, reinitializing weights.")
                   #         nn.init.uniform_(param, a=-0.1, b=0.1)  # Reinitialize to prevent training crash
                    lstm.train()
                    outputs = lstm.forward(x_batch_train)  # forward pass
                    outputs = outputs.clamp(min=-10, max=10)
    
    
                    optimizer.zero_grad()  # calculate the gradient, manually setting to 0
    
                    if torch.isnan(outputs).any():
                        print("[ERROR] NaN detected in model output! Stopping training.")
                        exit()
                    if torch.isnan(y_batch_train).any():
                        print("[ERROR] NaN detected in target labels! Stopping training.")
                        exit()
    
                    # obtain the loss function
                    # ✅2026-08-19: 块输出 (B,k,C) + 块标签 (B,k) — 展平逐位置 focal loss
                    if self.classification_type == "binary":
                        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), y_batch_train.reshape(-1, 1))
                    else:
                        loss = torch.nan_to_num(
                            criterion(outputs.reshape(-1, outputs.shape[-1]), y_batch_train.reshape(-1).long()),
                            nan=0.0, posinf=1.0, neginf=-1.0,
                        )
    
                    if torch.isnan(loss).any():
                        print("[ERROR] NaN detected in loss! Stopping training.")
                        exit()
    
                    loss.backward()  # calculates the loss of the loss function
    
                    #  Step 1: Detect and Reset NaN Gradients Before Optimizer Step
                    for name, param in lstm.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"[WARNING] NaN detected in {name} gradients, resetting to zero.")
                                param.grad = torch.zeros_like(param.grad)  # Prevent NaN propagation
    
                    #  Step 2: Add Small Gradient Noise to Attention Layer
                    for name, param in lstm.named_parameters():
                        if "attention.attention_weights" in name and param.grad is not None:
                            noise = torch.randn_like(param.grad) * 1e-3  # Tiny noise to prevent zero-variance
                            param.grad += noise
    
                    #  Step 3: Apply Even Stronger Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=self.grad_clip)  # Reduce gradient explosion
    
                    #  Step 4: Use SGD with Momentum for the Attention Layer Only
                    attention_params = [param for name, param in lstm.named_parameters() if "attention.attention_weights" in name]
                    #attention_optimizer = torch.optim.SGD(attention_params, lr=0.001, momentum=0.9)
    
                    # Update all parameters with the main optimizer
                    optimizer.step()
    
                    #  Step 4 (continued): Use SGD optimizer only for the attention layer
                    #attention_optimizer.step()
    
    
                    train_losses.append(loss.item())
    
            # change this line dependent on how often validation loss should be calculated
            if epoch % 1 == 0:
                # print train loss first
                print("-------------------------")
                print(datetime.datetime.now())
                print("-------------------------")
                print("Epoch: %d, train loss: %1.5f" % (epoch, np.mean(train_losses)))

                val_losses = []
                val_mccs = []
                val_accs = []
                val_kappas = []
                lstm.eval()

                # load validation data in batches
                if self.stateful:
                    # stateful: 逐组扫描, active 帧全量拼接后统一算 loss/指标
                    with torch.no_grad():
                        logits_val, y_val_cat = self._stateful_scan(lstm, frames_val, self.device)
                    # ✅2026-08-19: 块输出 (N,k,C) — stateful 本轮最小适配: 取第 0 头 (当前帧),
                    #   行为与改造前单头一致 (stateful 完整序列化下一轮做)
                    logits_val = logits_val[:, 0]
                    if self.classification_type == "binary":
                        val_loss = criterion_none(logits_val, y_val_cat.unsqueeze(-1)).squeeze(-1).mean()
                    else:
                        val_loss = criterion_none(logits_val, y_val_cat.long()).mean()
                    val_losses.append(val_loss.item())
                    class_performance = tensor_to_performance(y_val_cat.unsqueeze(1), logits_val, self.classification_type)
                    val_mccs.append(class_performance["mcc"])
                    val_accs.append(class_performance["accuracy"])
                    val_kappas.append(class_performance["kappa"])
                else:
                    x_batch_val_list, y_batch_val_list = self.batch_loader(x_val, y_val)

                    # iterate over all batches of validation data
                    for x_batch_val, y_batch_val in zip(x_batch_val_list, y_batch_val_list):
                        x_batch_val = x_batch_val.to(self.device)
                        y_batch_val = y_batch_val.to(self.device)

                        with torch.no_grad():
                            y_pred = lstm.forward(x_batch_val)  # (B, k, C)

                        # calculate loss of batch-wise prediction (全头逐位置, 与训练一致)
                        if self.classification_type == "binary":
                            val_loss = criterion(y_pred.reshape(-1, y_pred.shape[-1]), y_batch_val.reshape(-1, 1))
                        else:
                            val_loss = criterion(y_pred.reshape(-1, y_pred.shape[-1]), y_batch_val.reshape(-1).long())

                        val_losses.append(val_loss.item())

                        # ✅2026-08-19: 指标用第 0 头 (窗口中心 = 现状预测位置) — checkpoint 选择口径
                        class_performance = tensor_to_performance(
                            y_batch_val[:, 0:1], y_pred[:, 0], self.classification_type
                        )
                        val_mccs.append(class_performance["mcc"])
                        val_accs.append(class_performance["accuracy"])
                        val_kappas.append(class_performance["kappa"])

                mean_val_loss = np.mean(val_losses)
                mean_mcc = np.mean(val_mccs)
                mean_acc = np.mean(val_accs)
                mean_kappa = np.mean(val_kappas)

                print(f"Validation Loss: {mean_val_loss:.5f}")
                print(f"Validation Acc: {mean_acc:.4f}  Kappa: {mean_kappa:.4f}  MCC: {mean_mcc:.4f}")

                # 混合数据集：按子集计算验证指标
                if self.val_sources and len(self.val_sources) > 1:
                    lstm.eval()
                    print("  Per-source Val:")
                    import warnings
                    for src_name, src_data in sorted(self.val_sources.items()):
                        src_mccs, src_accs, src_kappas = [], [], []
                        if self.stateful:
                            # stateful: src_data 为 frames list
                            if not src_data:
                                # ✅2026-08-12: 混合划分中缺失 val 被试的源为空 frames (split 代码
                                # 显式造 ds[0:0] 空集), _stateful_scan 内 torch.cat([]) 会直接崩 —
                                # 空源无指标可算, 跳过
                                print(f"    {src_name:12s}  (empty source, skipped)")
                                continue
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                with torch.no_grad():
                                    logits_src, y_src = self._stateful_scan(lstm, src_data, self.device)
                                    # ✅2026-08-19: 块输出取第 0 头 (stateful 最小适配)
                                    perf = tensor_to_performance(
                                        y_src.unsqueeze(1), logits_src[:, 0], self.classification_type
                                    )
                            src_accs.append(perf['accuracy'])
                            src_kappas.append(perf['kappa'])
                            src_mccs.append(perf['mcc'])
                        else:
                            xs, ys = src_data
                            if xs.shape[0] == 0:
                                # ✅2026-08-13: 空源 (混合划分缺失 val 被试) — batch_loader 对空
                                # 张量仍 yield 一个空 chunk, tensor_to_performance → sklearn
                                # 空指标崩溃; 与 stateful 分支同理跳过 (既有问题, 顺手修)
                                print(f"    {src_name:12s}  (empty source, skipped)")
                                continue
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                for xb, yb in zip(*self.batch_loader(xs, ys)):
                                    xb, yb = xb.to(self.device), yb.to(self.device)
                                    with torch.no_grad():
                                        yp = lstm.forward(xb)  # (B, k, C)
                                        # ✅2026-08-19: 指标用第 0 头 (与主 val 口径一致)
                                        perf = tensor_to_performance(yb[:, 0:1], yp[:, 0], self.classification_type)
                                    src_accs.append(perf['accuracy'])
                                    src_kappas.append(perf['kappa'])
                                    src_mccs.append(perf['mcc'])
                        print(f"    {src_name:12s}  Acc={np.mean(src_accs):.4f}  "
                              f"Kappa={np.mean(src_kappas):.4f}  MCC={np.mean(src_mccs):.4f}")
                print("-------------------------")

                #  Overfitting Check: Stop if training loss is much lower than validation loss
                train_loss = np.mean(train_losses)
                if (train_loss - mean_val_loss) > 0.3:
                    print("[WARNING] Possible Overfitting Detected: Large gap between train and validation loss.")

                #  Save checkpoint every 5 epochs
                if epoch > 0 and epoch % 5 == 0:
                    ckpt_dir = self.output_dir / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_name = (
                        f"ckpt_epoch_{epoch:03d}_"
                        f"acc{mean_acc:.4f}_"
                        f"k{mean_kappa:.4f}_"
                        f"mcc{mean_mcc:.4f}.pt"
                    )
                    torch.save(lstm.state_dict(), ckpt_dir / ckpt_name)
                    print(f"[CHECKPOINT] Saved {ckpt_name}", flush=True)

                #  Improved Early Stopping
                if mean_val_loss < min_val_loss:
                    min_val_loss = mean_val_loss
                    max_val_performance = mean_mcc
                    patience_counter = 0  # Reset patience if improvement is found

                    print("*************************")
                    print(f"New Best Validation Loss: {mean_val_loss:.5f}")
                    print(f"Validation Acc: {mean_acc:.4f}  Kappa: {mean_kappa:.4f}  MCC: {mean_mcc:.4f}")
                    print("*************************", flush=True)

                    #  Save best model
                    ckpt_dir = self.output_dir / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        lstm.state_dict(),
                        ckpt_dir / "best_model.pt",
                    )

                else:
                    patience_counter += 1
                    print(f"[INFO] No Improvement. Patience: {patience_counter}/{patience_threshold}", flush=True)

                #  Define new stopping criteria
                if (
                        patience_counter >= patience_threshold  # Stop if patience runs out
                        or (epoch > min_epochs and mean_mcc <= 0.0)  # Stop if MCC is bad
                ):
                    print("[STOPPING] Training stopped due to no improvement or bad MCC.", flush=True)
                    break

        return max_val_performance

    def test(self, x_test, y_test, retrain=False):
        score_dict = {}
        pred_dict = {}

        num_classes = get_num_classes(self.classification_type)

        if self.use_gpu:
            # load "empty" lstm model --> same parameters as in training
            lstm = self._load_empty_model(use_gpu=False, num_classes=num_classes, dropout=self.dropout, use_attention=False)

            # load best model obtained from training
            self._load_best_model(lstm)
            lstm = lstm.cuda()

        else:
            lstm = self._load_empty_model(use_gpu=False, num_classes=num_classes, dropout=self.dropout, use_attention=False)

            # load best model obtained from training
            self._load_best_model(lstm)

        lstm.eval()

        if self.stateful:
            # ✅2026-08-11: stateful 测试 — 逐被试扫描 (P=1), 输出契约与无状态路径一致
            for (x_s, y_s, subj_idx) in x_test:
                subj_idx = str(subj_idx)
                x_s = x_s.to(self.device)
                h = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
                c = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
                buf = torch.zeros(1, self.seq_len - 1, self.hidden_size, device=self.device)

                with torch.no_grad():
                    # warm-up: 首帧 seq_len-1 次, 填充池化缓冲 (与无状态 edge-pad 语义一致)
                    x0 = x_s[:1]
                    for _ in range(self.seq_len - 1):
                        _, h, c, buf = lstm.forward_stateful(x0, h, c, buf)
                    # 逐帧扫描
                    logits_list = []
                    for t in range(x_s.shape[0]):
                        out, h, c, buf = lstm.forward_stateful(x_s[t:t + 1], h, c, buf)
                        logits_list.append(out)
                    y_pred = torch.cat(logits_list).to(device="cpu")
                y_pred = y_pred.detach().numpy()
                # ✅2026-08-19: stateful 本轮最小适配 — 块输出取第 0 头 (当前帧), 行为同改造前
                y_pred = y_pred[:, 0]

                # move ground truth data to cpu and convert to numpy array
                y_batch_test = y_s.cpu()
                y_batch_test = pd.DataFrame(y_batch_test.detach().numpy(), columns=["sleep_stage"])

                # determine prediction based on classification type
                if self.classification_type == "binary":
                    y_pred = (1 / (1 + np.exp(-y_pred)) >= 0.5).astype(float)
                else:
                    y_pred = np.argmax(y_pred, axis=1)

                # save predictions in dictionary
                pred_dict[subj_idx] = y_pred

                # safe sleep stage predictions with subject id to csv file for subsequent analysis
                subj_pred_dir = self.output_dir / "per_subject_predictions"
                subj_pred_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(y_pred).to_csv(subj_pred_dir / f"{subj_idx}.csv")

                # calculate classification performance for each subject
                subj_score = dl_score(
                    y_pred, y_batch_test, classification_type=self.classification_type, subject_id=subj_idx
                )
                score_dict[subj_idx] = subj_score
        else:
            # iterate over test data
            for i, (x_batch_test, y_batch_test) in enumerate(zip(x_test, y_test)):
                # obtain subject_id
                subj_idx = x_batch_test[1]
                # move data to gpu if available
                x_batch_test = x_batch_test[0].to(device=self.device)

                # apply model to test data and move to cpu and convert to numpy array
                with torch.no_grad():
                    blocks = lstm.forward(x_batch_test).to(device="cpu")
                blocks = blocks.detach().numpy()  # (W, k, C) — 每窗口一个块

                # ✅2026-08-19: 时间集成 — 对每个 epoch t, 加权合并窗口 [t-k+1, t] 的
                #   k 个深度预测 (log 域, w_j = exp(-m*j)); m=0 退化为均匀 mean log-prob
                scores = self._ensemble_chunked(blocks, self.ensemble_m)  # (W, C)

                # move ground truth data to cpu and convert to numpy array
                # 块 [i, i+k-1] 第 0 头 = epoch i → 评估用第 0 列 (与夜末尾截断对齐)
                y_batch_test = y_batch_test[0].cpu().detach().numpy()
                y_batch_test = pd.DataFrame(y_batch_test[:, 0], columns=["sleep_stage"])

                # determine prediction based on classification type
                if self.classification_type == "binary":
                    y_pred = (1 / (1 + np.exp(-scores)) >= 0.5).astype(float)
                else:
                    y_pred = np.argmax(scores, axis=1)

                # save predictions in dictionary
                pred_dict[subj_idx] = y_pred

                # safe sleep stage predictions with subject id to csv file for subsequent analysis
                subj_pred_dir = self.output_dir / "per_subject_predictions"
                subj_pred_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(y_pred).to_csv(subj_pred_dir / f"{subj_idx}.csv")

                # calculate classification performance for each subject
                subj_score = dl_score(
                    y_pred, y_batch_test, classification_type=self.classification_type, subject_id=subj_idx
                )
                score_dict[subj_idx] = subj_score

        subject_results = pd.DataFrame(score_dict)
        # 排除 no_agg 包裹的对象 (confusion_matrix)，新版 pandas 无法对其求 mean
        numeric_cols = [c for c in subject_results.index if c != "confusion_matrix"]
        score_mean = subject_results.loc[numeric_cols].agg(["mean"], axis=1).T

        return subject_results, score_mean, pred_dict

    def _ensemble_chunked(self, blocks, m):
        """✅2026-08-19: 块级时间集成 (ACT temporal ensembling 的睡眠分期移植)。

        :param blocks: 每窗口的块 logits, (W, k, C) float32 numpy
        :param m: 指数权重参数 w_j = exp(-m*j), j = 预测深度 (j=0 = 窗口中心/最新)
        :return: (W, C) 集成后分数

        对 epoch t, 预测来自窗口 w ∈ [max(0, t-k+1), t], 深度 j = t-w。
        - multi 类: log 域加权 (log_softmax 后加权平均 = 概率几何平均 = 乘性共识);
          log_softmax clamp(-100) 防极端 logits 产生 -inf → 0*-inf=NaN。
        - binary: 对 logits 加权平均 (几何 odds 平均, 与 log 域一致)。
        - m=0 → 均匀平均 (= 现状 mean log-probability); k=1 → 退化为无操作。
        - 夜末尾 epoch (t > W-k) 有效窗口数 < k, 只平均存在的 (除数按实际窗口数)。
        """
        W, k, C = blocks.shape
        if k == 1:
            return blocks[:, 0]
        weights = np.exp(-m * np.arange(k))  # (k,) — 深度 j 的权重, j=0 最新最可信
        scores = np.zeros((W, C), dtype=np.float32)
        for t in range(W):
            w_start = max(0, t - k + 1)
            ws = np.arange(w_start, t + 1)          # 覆盖 epoch t 的窗口
            js = t - ws                             # 各窗口的深度 j = t - w
            wsel = weights[js]
            wsum = wsel.sum()
            # 成对索引: 窗口 w 取块深度 (t-w) — 每个窗口的深度随 w 变化
            sel = blocks[ws, js, :]                 # (n_w, C)
            if C == 1:
                scores[t] = (wsel[:, None] * sel).sum(axis=0) / wsum
            else:
                log_p = sel.astype(np.float64)
                mx = log_p.max(axis=-1, keepdims=True)
                log_p = log_p - mx - np.log(np.exp(log_p - mx).sum(axis=-1, keepdims=True))
                log_p = np.clip(log_p, -100.0, None)
                scores[t] = (wsel[:, None] * log_p).sum(axis=0) / wsum
        return scores

    # ------------------------------------------------------------------
    # stateful 辅助 (✅2026-08-11 - rdwang: 逐帧扫描 + truncated BPTT)
    # ------------------------------------------------------------------

    def _stateful_groups(self, frames):
        """按 P = max(1, batch_size // state_chunk) 个被试一组切分 (夜长降序, 组内长度相近)。
        组内状态从零开始, 组间互不携带。返回 list of [ (x_s, y_s, subj_id), ... ]。"""
        P = max(1, self.batch_size // self.state_chunk)
        subjects = sorted(frames, key=lambda t: -t[0].shape[0])
        return [subjects[g:g + P] for g in range(0, len(subjects), P)]

    def _stateful_warmup(self, lstm, group):
        """组起点 warm-up: 各被试首帧从零状态推进 (seq_len-1) 次, 填充池化缓冲。
        - 训练噪声只加在夜起点零状态 (镜像 forward() 的 zero-init + 1e-3 noise), 携带状态不再加噪
        - 带 autograd 图 (训练时) — 镜像无状态首窗口的完整图; 调用方在 no_grad 下则为 eval
        返回 (h, c, buf)。"""
        n_g = len(group)
        h = torch.zeros(self.num_layers, n_g, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, n_g, self.hidden_size, device=self.device)
        buf = torch.zeros(n_g, self.seq_len - 1, self.hidden_size, device=self.device)
        if lstm.training:
            h += torch.randn_like(h) * 1e-3
            c += torch.randn_like(c) * 1e-3
        x0 = torch.stack([t[0][:1].to(self.device) for t in group])  # (P, 1, F)
        for _ in range(self.seq_len - 1):
            _, h, c, buf = lstm.forward_stateful(x0, h, c, buf)
        return h, c, buf

    def _stateful_build_chunk(self, group, chunk_start, device):
        """构建一个 chunk: 窗口 [chunk_start, chunk_start+C) 的 (x_chunk (C,n_g,F),
        y_chunk (C,n_g), active (C,n_g) bool)。组内夜已结束的被试在 active 中标记为 False。"""
        max_len = max(t[0].shape[0] for t in group)
        c_len = min(self.state_chunk, max_len - chunk_start)
        n_g = len(group)
        n_feat = group[0][0].shape[1]
        x_chunk = torch.zeros(c_len, n_g, n_feat, device=device)
        y_chunk = torch.zeros(c_len, n_g, device=device)
        active = torch.zeros(c_len, n_g, dtype=torch.bool, device=device)
        for t in range(c_len):
            i = chunk_start + t
            for s in range(n_g):
                if i < group[s][0].shape[0]:
                    x_chunk[t, s] = group[s][0][i]
                    y_chunk[t, s] = group[s][1][i]
                    active[t, s] = True
        return x_chunk, y_chunk, active

    def _stateful_chunk_step(self, lstm, criterion_none, optimizer, x_chunk, y_chunk, active,
                             h, c, buf, detach_state, train_losses):
        """stateful 单 chunk 训练步: 逐帧 forward_stateful + 掩码损失 + 反向。
        detach_state=True 时先 detach (h,c,buf) — truncated BPTT 截断点 (chunk 0 保留 warm-up 图)。
        返回推进后的 (h, c, buf) 与是否实际 step (全 inactive 的残留 chunk 跳过)。"""
        if detach_state:
            h, c, buf = h.detach(), c.detach(), buf.detach()
        lstm.train()
        logits_list = []
        for t in range(x_chunk.shape[0]):
            outputs, h, c, buf = lstm.forward_stateful(x_chunk[t], h, c, buf)
            logits_list.append(outputs)
        outputs = torch.stack(logits_list).clamp(min=-10, max=10)  # (C, P, k, C_out)
        # ✅2026-08-19: stateful 本轮最小适配 — 取第 0 头 (当前帧), 行为同改造前
        #   (stateful 块标签/完整序列化下一轮做)
        outputs = outputs[:, :, 0]  # (C, P, C_out)

        optimizer.zero_grad()

        if torch.isnan(outputs).any():
            print("[ERROR] NaN detected in model output! Stopping training.")
            exit()
        if torch.isnan(y_chunk).any():
            print("[ERROR] NaN detected in target labels! Stopping training.")
            exit()

        # 逐样本 loss (reduction="none"), 再按 active 掩码做均值
        if self.classification_type == "binary":
            loss = criterion_none(outputs, y_chunk.unsqueeze(-1)).squeeze(-1)  # (C, P)
        else:
            loss = torch.nan_to_num(
                criterion_none(outputs, y_chunk.long()),
                nan=0.0, posinf=1.0, neginf=-1.0,
            )  # (C, P)
        denom = active.float().sum()
        if denom.item() == 0:
            return h, c, buf, False
        loss = (loss * active.float()).sum() / denom  # 组内夜已结束的被试掩码

        if torch.isnan(loss).any():
            print("[ERROR] NaN detected in loss! Stopping training.")
            exit()

        loss.backward()

        # Step 1: Detect and Reset NaN Gradients Before Optimizer Step
        for name, param in lstm.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"[WARNING] NaN detected in {name} gradients, resetting to zero.")
                    param.grad = torch.zeros_like(param.grad)  # Prevent NaN propagation

        # Step 2: Add Small Gradient Noise to Attention Layer
        for name, param in lstm.named_parameters():
            if "attention.attention_weights" in name and param.grad is not None:
                noise = torch.randn_like(param.grad) * 1e-3  # Tiny noise to prevent zero-variance
                param.grad += noise

        # Step 3: Apply Even Stronger Gradient Clipping
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=self.grad_clip)

        optimizer.step()

        train_losses.append(loss.item())
        return h, c, buf, True

    def _stateful_scan(self, lstm, frames, device):
        """stateful eval 扫描: 逐组逐 chunk 推进状态, 只收集 active 帧的 (logits, y)。
        返回 (logits_cat (N,C), y_cat (N,))。调用方负责 eval 模式与 no_grad。"""
        logits_all, y_all = [], []
        for group in self._stateful_groups(frames):
            h, c, buf = self._stateful_warmup(lstm, group)
            max_len = max(t[0].shape[0] for t in group)
            for chunk_start in range(0, max_len, self.state_chunk):
                x_chunk, y_chunk, active = self._stateful_build_chunk(group, chunk_start, device)
                for t in range(x_chunk.shape[0]):
                    logits, h, c, buf = lstm.forward_stateful(x_chunk[t], h, c, buf)
                    if active[t].any():
                        logits_all.append(logits[active[t]])
                        y_all.append(y_chunk[t][active[t]])
        return torch.cat(logits_all), torch.cat(y_all)

    def batch_loader(self, x_train, y_train):
        return list(x_train.split(self.batch_size)), list(y_train.split(self.batch_size))

    def _load_best_model(self, model):
        """加载本次训练过程中的最佳模型权重"""
        model.load_state_dict(
            torch.load(self.output_dir / "checkpoints" / "best_model.pt")
        )
        model.eval()

    def _load_best_model_from_path(self, weights_path):
        """从指定路径加载权重 (用于恢复训练或评估历史模型)"""
        self.weights_loaded_from = weights_path
        # 这个标记会在 test() 中被 _load_best_model 使用
        # 实际上 test() 调 _load_best_model，需要 hack 一下让它用外部路径
        # 这里重写 _load_best_model 行为:
        self._load_best_model = lambda model: (
            model.load_state_dict(torch.load(weights_path)),
            model.eval()
        )

    def _load_best_mesa_model(self, model):
        model.load_state_dict(
            torch.load(
                Path(__file__)
                .parents[4]
                .joinpath(
                    "exports_our/pickle_pipelines/lstm_"
                    + "_".join(self.modality)
                    + "_"
                    + "MESA_Sleep"
                    + "_"
                    + self.classification_type
                )
            )
        )

    def _load_empty_model(self, use_gpu, num_classes, dropout, use_attention=False):
        return Model(
            num_classes=num_classes,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            use_gpu=use_gpu,
            dropout=dropout,
            dataset_name=self.dataset_name,
            modality=self.modality,
            use_attention=use_attention,
            use_internal_norm=self.internal_norm,
            chunk_len=self.chunk_len,  # ✅2026-08-19: action-chunking 块长度
        )
