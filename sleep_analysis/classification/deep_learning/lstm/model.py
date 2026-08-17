import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# Debugging Function
def debug_tensor(tensor, name):
    """Prints debug info if NaN or Inf values are detected in a tensor."""
    if torch.isnan(tensor).any():
        print(f"[DEBUG] NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"[DEBUG] Inf detected in {name}")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_size)  # Helps prevent exploding/vanishing gradients
        self.dropout = nn.Dropout(0.1)  # Prevents overfitting to a small number of features

    def forward(self, lstm_output):
        lstm_output = self.norm(lstm_output)  # Normalize before attention
        attn_scores = self.attention_weights(lstm_output).squeeze(-1)  # Shape: (batch_size, seq_length)
        attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True)[0]  # Stability trick
        attn_scores = attn_scores.clamp(min=-10, max=10)  # Clip extreme values
        attn_weights = torch.softmax(attn_scores, dim=1)  # Apply softmax
        attn_weights = self.dropout(attn_weights)  # Apply dropout to stabilize training
        attended_output = torch.sum(attn_weights.unsqueeze(-1) * lstm_output, dim=1)  # Weighted sum
        return attended_output

# Main LSTM Model
class Model(nn.Module):
    def __init__(
        self, num_classes, input_size, hidden_size, num_layers, dropout, use_gpu, use_attention=True,
        dataset_name="dataset_name", modality="acc", use_internal_norm=False
    ):
        super(Model, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.modality = modality
        self.dataset_name = dataset_name
        self.use_attention = use_attention
        self.use_gpu = use_gpu
        self.use_internal_norm = use_internal_norm  # ✅2026-08-14: 旧版内部归一化开关 (复现 08-06 基线用)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        if self.use_attention:
            self.attention = Attention(hidden_size)  # Only create attention if enabled

        # Fully connected layers
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # ✅ Move model to GPU if enabled
        if self.use_gpu:
            self.to("cuda")

        # Initialize Weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights properly to prevent NaNs"""
        for name, param in self.named_parameters():
            if "weight_ih" in name:  # Input-to-hidden weights
                nn.init.uniform_(param, a=-0.1, b=0.1)  # Uniform range prevents NaNs
            elif "weight_hh" in name:  # Hidden-to-hidden weights
                # orthogonal init uses QR decomp which can fail with some CUDA driver/lib combos.
                # Do it on CPU, then copy back to the original device.
                cpu_param = param.data.cpu()
                nn.init.orthogonal_(cpu_param)
                param.data.copy_(cpu_param)
            elif "bias" in name:
                nn.init.constant_(param, 0)  # Biases set to zero

    def forward(self, x):

        #print("GPU is enabled") if self.use_gpu else print("GPU is disabled")

        x = x.cuda() if self.use_gpu else x

        device = x.device  # Get device of input tensor

        # ✅ Ensure model parameters are on the same device
        self.to(device)

        if torch.isnan(x).any():
            print("[DEBUG] NaN detected in input batch. Skipping this batch.")
            return torch.zeros(x.shape[0], self.num_classes, device=x.device)

        # Ensure input has correct dimensions
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        # ✅20260731 - rdwang: forward的时候需要单人整夜数据的mean/std，不符合事实分期的要求，适配实时睡眠分期时要修
        # ✅2026-08-07 - rdwang: 去掉内部 per-batch 归一化（外部 scaler 作为唯一归一化）
        #   原因: 1) 外部 scaler 已做逐特征 z-score (训练集拟合, 无泄漏),
        #            内部再归一化是冗余的双重标准化 (还把个体/夜间水平信息抹掉)
        #         2) 推理时当场用整夜数据算 mean/std → 未来泄漏 + 训练-推理不一致
        #             (训练=batch混合统计 vs 推理=整夜统计)
        #         3) 去掉后 forward 不依赖 batch 统计 → 整夜 batch 与逐窗口/流式推理
        #             结果完全一致, 是实时推理的前提; ONNX 图也更干净

        # ✅2026-08-14 - rdwang: 为复现 08-06 基线 (MCC 0.564), 把旧版内部归一化做成
        #   --internal-norm 开关 (默认关闭, 行为与 08-07 之后完全一致)。
        #   恢复的代码取自 git 37a1618^ (remove additional normalization 的父版本), 一字不差。
        #   注意: 该路径重新引入 训练=batch统计 vs 推理=整夜统计 的不一致, 仅用于复现对照。
        if self.use_internal_norm:
            #  历史:
            # ✅20260803 - rdwang: 推理pipeline误差可能是1e-8导致的。AI分析: ONNX 的算子实现和 PyTorch 不同——torch.std(unbiased=True) 在 ONNX 里没有对等算子，dynamo 只能用 ReduceMean（总体方差，除以 n）来近似，在 Feature 1 std≈0 时被 eps=1e-8 放大。
            # ✅20260804 - rdwang: 提到1e-5之后问题解决。
            mean_x = x.mean(dim=(0, 1), keepdim=True)
            std_x = x.std(dim=(0, 1), keepdim=True) + 1e-5  # Avoid division by zero
            if torch.isnan(mean_x).any() or torch.isnan(std_x).any():
                print("[DEBUG] Skipping normalization due to NaN in batch statistics")
            else:
                x = (x - mean_x) / std_x
            debug_tensor(x, "Normalized Input x")

        # Initialize hidden and cell state
        if self.use_gpu:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device).cuda()
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device).cuda()
        else:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        # Add small noise to hidden states to prevent instability (training only)
        if self.training:
            h_0 += torch.randn_like(h_0) * 1e-3
            c_0 += torch.randn_like(c_0) * 1e-3

        lstm_out, _ = self.lstm(x, (h_0, c_0))  # LSTM output

        debug_tensor(lstm_out, "LSTM Output")

        if self.use_attention:
            attn_out = self.attention(lstm_out)  # Use attention if enabled
            debug_tensor(attn_out, "Attention Output")
        else:
            attn_out = lstm_out.mean(dim=1)  # Use mean pooling instead of attention
            debug_tensor(attn_out, "Mean Pooled Output (Attention Disabled)")

        out = self.relu(attn_out)
        out = self.fc_1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out)  # Final classification layer

        debug_tensor(out, "Final Model Output")

        return out

    def forward_stateful(self, x_t, h, c, buf):
        """Stateful 单帧前向 (流式分期核心)。

        :param x_t: 单帧特征, (B, F) 或 (B, 1, F)  (注意: 与 forward() 的 (B, F, 1) 不同)
        :param h:   隐状态 (num_layers, B, hidden_size)
        :param c:   细胞状态 (num_layers, B, hidden_size)
        :param buf: 最近 (seq_len-1) 个时刻的顶层 h, (B, seq_len-1, hidden_size)
        :return:    (logits (B, num_classes), h_n, c_n, buf_out (B, seq_len-1, hidden_size))

        ✅2026-08-11 - rdwang: 有状态训练/推理的单帧步进。与 forward() 的滑窗语义逐位对齐:
          每夜先对首帧 warm-up (seq_len-1) 次 (由调用方完成), 之后第 t 帧的 logits =
          MLP(mean(h_{t-seq_len+1..t})) —— 等价于无状态窗口 [t-seq_len+1..t] 的池化
          (epoch 0 逐位一致; epoch >= 1 额外携带前缀上下文, 是状态化的目的)。
          ⚠️ 池化必须取 (buf, h_n) 共 seq_len 个值, 不能取 buf_out (seq_len-1 个, 差一)。
          状态全部显式传递, 无隐藏 module 状态 → 可 ONNX 导出。
        ✅2026-08-13 - rdwang: 严禁在此调用 self.to(device) — 每帧触发 nn.LSTM._apply →
          重建扁平化权重缓冲 (~55 MB), 且 cuDNN autograd 节点把它保存到 chunk backward,
          逐帧堆积 (21 warm-up + 64 chunk ≈ 4.7 GB) 直接 OOM (MESA stateful 实测)。
          模型在构造 (Model.__init__ use_gpu) / 加载权重 (engine _load_weights, test cuda())
          时已在目标设备上, 运行中设备不变, 无需每步迁移。
        """
        # (B, F) → (B, 1, F): 帧输入的特征轴在 dim 1, 与 forward() 的 (B, F, 1) reshape 不同
        if len(x_t.shape) == 2:
            x_t = x_t.unsqueeze(1)

        if torch.isnan(x_t).any():
            # 镜像 forward():101-103 的 NaN 逃逸: logits 置零, 状态不推进 (数据错误, 帧对齐保持)
            print("[DEBUG] NaN detected in stateful input. Returning zero logits without advancing state.")
            return torch.zeros(x_t.shape[0], self.num_classes, device=x_t.device), h, c, buf

        lstm_out, (h_n, c_n) = self.lstm(x_t, (h, c))  # lstm_out: (B, 1, H)

        # 顶层隐状态 (与 forward() 的 lstm_out 取同一层)
        last_h = h_n[-1]  # (B, H)

        # 池化: 必须取 (buf, h_n) 共 seq_len 个值 (buf: 最近 S-1 个 + 当前 1 个)
        combined = torch.cat([buf, last_h.unsqueeze(1)], dim=1)  # (B, seq_len, H)
        pooled = combined.mean(dim=1)  # 与 forward() 的 lstm_out.mean(dim=1) 一致

        out = self.relu(pooled)
        out = self.fc_1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out)

        # 滚动缓冲: 丢掉最旧的, 推入当前
        buf_out = torch.cat([buf[:, 1:, :], last_h.unsqueeze(1)], dim=1)  # (B, seq_len-1, H)

        return out, h_n, c_n, buf_out
