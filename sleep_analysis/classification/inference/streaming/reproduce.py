"""路径 B：复现训练时的评测精度（离线特征表 → 整夜推理）。

**本模块与 `infer.py`（路径 A）并列**，两者共用同一个 ONNX 模型、同一份 scaler，
只在**特征来源**和**归一化统计量**上不同：

========================================  ==========================  ==========================
                                          **路径 A**（`infer.py`）     **路径 B**（本文件）
========================================  ==========================  ==========================
用途                                       部署（雷达实机）             复现训练评测
特征来源                                   **原始波形** → 流式特征       **离线特征表** CSV
特征口径                                   `prep.SLOT_NAMES`            `data_parparation` 的训练口径
                                          （slot4 = 150_hrv_*)         （slot4 = _hrv_median_nni 重复列）
一次处理的粒度                              一个 900 s 窗口              一个被试的整夜
归一化（第二层）统计量                      训练集平均基线                **该被试整夜**（= 训练时）
========================================  ==========================  ==========================

**为什么两者必须用不同口径**：路径 B 的目的是"与训练时的评测结果对上"，所以每一处都必须
照抄训练管线的做法；改动任何一列都会让它对不上。而路径 A 是部署，用的是雷达输出的特征表，
口径由 `pipeline.py` 的 13 槽位定义。

逐步对齐 `LSTM.py::test`（非 stateful 分支）
--------------------------------------------
1. 特征列：训练口径的 16 列（`[ACT..., _has_act, HRV..., _has_hrv, RRV..., _has_rrv]`），
   缺模态填 0 且标志置 0
2. `pad(mode="mean")` 按列填该列均值 → 滑窗 `seq_len=21`（步长 1）
   —— 对应 `get_sequence_data(causal=False, padding=True)`
3. scaler：**训练集**的 `mean_/scale_`（`checkpoints/scaler.json`）
4. **整夜归一化**：`x.mean(axis=(0,1))` / `x.std(axis=(0,1), ddof=1) + 1e-5`
   —— 训练时 `x_test` 是**逐被试**喂的（`LSTM.py::test` 的 `for x_batch_test in x_test`），
   一个被试就是一个 batch，所以 `internal_norm` 的统计量 = 该被试整夜全部窗口 × 时间步。
   本模块用 `apply_night_norm` 在**图外**复现同一个量（部署变体图内没有 norm）。
5. `argmax` → 分期

⚠️ **路径 B 用不了 `StreamingPreprocessor`** —— 它吃的是原始波形、按 900 s 窗口出特征；
   路径 B 吃的是别人已经算好的整夜特征表。两者的输入形态完全不同。

⚠️ **本模块只用于验证，不用于部署。** 真机上拿不到 `features_full_combined/*.csv`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

#: 训练时的特征列（来自 `base_sleep_dataset.BaseSleepDataset.FEATURE_COLUMNS`）。
#: ⚠️ HRV 的第 3 项（0-based index 2）与第 1 项**是同一个列名** `_hrv_median_nni`
#:    —— 历史上为凑 HRV=8 的输入维度留下的重复项，训练时就是这样喂的。
#:    部署路径（`prep.SLOT_NAMES`）把它换成了 `150_hrv_median_nni`，所以两边**不能互换**。
TRAIN_FEATURE_COLUMNS = {
    "ACT": ["_acc_mean_1"],
    "HRV": ["_hrv_median_nni", "_hrv_ratio_sd2_sd1", "_hrv_median_nni",
            "_hrv_vlf", "_hrv_lf", "_hrv_hf", "_hrv_lf_hf_ratio", "_hrv_total_power"],
    "RRV": ["150_RRV_MedianBB", "150_RRV_LF", "270_RRV_MCVBB", "150_RRV_CVBB"],
}

#: 模态 → 判断该被试有没有这个模态的列名关键字
_MODALITY_PROBE = {"ACT": "_acc", "HRV": "_hrv", "RRV": "RRV"}

#: 第二层归一化的 eps（与 `model.py` 的 `std_x = x.std(...) + 1e-5` 一致）
NIGHT_NORM_EPS = 1e-5


def training_input_layout(modality: List[str]) -> List[str]:
    """训练口径的 16 列布局（含 `_has_*`）。"""
    cols: List[str] = []
    for mod in ("ACT", "HRV", "RRV"):
        if mod in modality:
            cols.extend(TRAIN_FEATURE_COLUMNS[mod])
            cols.append(f"_has_{mod.lower()}")
    return cols


def build_train_matrix(df, modality: List[str]) -> np.ndarray:
    """由被试的特征表拼出训练口径的 `(n_epoch, 16)` **原始**矩阵。

    缺模态：该模态的特征列填 0、标志列填 0（与 `_extract_subj_features_raw` 一致）。
    """
    n = len(df)
    parts = []
    for mod in ("ACT", "HRV", "RRV"):
        if mod not in modality:
            continue
        present = not df.filter(regex=_MODALITY_PROBE[mod]).empty
        if present:
            parts.append(df[TRAIN_FEATURE_COLUMNS[mod]].to_numpy(dtype=np.float64))
            parts.append(np.ones((n, 1), dtype=np.float64))
        else:
            parts.append(np.zeros((n, len(TRAIN_FEATURE_COLUMNS[mod])), dtype=np.float64))
            parts.append(np.zeros((n, 1), dtype=np.float64))
    return np.concatenate(parts, axis=1)


def build_sequences_night(raw: np.ndarray, seq_len: int) -> np.ndarray:
    """`pad(mode="mean")` → 滑窗（步长 1）。对应 `get_sequence_data(causal=False, padding=True)`。

    ⚠️ `np.pad(mode="mean")` 是**按列**填该列的整夜均值（不是全局均值）。
    ⚠️ 返回 `(n_epoch, seq_len, F)` —— 窗口数 = epoch 数（padding 后滑窗，步长 1）。
    """
    half = seq_len // 2
    padded = np.pad(raw, ((half, half), (0, 0)), mode="mean")
    w = np.lib.stride_tricks.sliding_window_view(padded, seq_len, axis=0)
    return np.ascontiguousarray(np.moveaxis(w, -1, 1))       # (W, F, S) → (W, S, F)


def apply_night_norm(x: np.ndarray) -> np.ndarray:
    """整夜归一化 —— 复现训练时 `internal_norm` 在"一个被试一个 batch"下的行为。

    与 `inference/data_utils.apply_night_norm` 同构（那边是既有引擎的实现）::

        mean = x.mean(axis=(0, 1), keepdims=True)
        std  = x.std(axis=(0, 1), ddof=1, keepdims=True) + eps

    ⚠️ `ddof=1` 对齐 `torch.std(unbiased=True)`；`eps=1e-5` 对齐 `model.py`。
    """
    # ⚠️ **逐列**算 mean/std，而不是 `x.mean(axis=(0,1))` 一把梭。
    #   原因：`internal_norm` 的 `+1e-5` 会把常数列（`_has_*`）上「mean 的浮点舍入误差」
    #   放大 1e5 倍。numpy 在 3D 数组上求和的分块顺序与 torch 不同，算出的 mean 差
    #   1 ULP，归一化后就差 ~1（实测 `_has_act` 差 0.96）。逐列求和的归约顺序与
    #   torch 一致得多（实测差 ~5e-4）。
    #   ⚠️ 这仍然不是逐位一致 —— 常数列的归一化在数学上是 0/1e-5，浮点上是**噪声**。
    #      要完全复现需要与 torch 逐位相同的归约，不现实。见调用方的说明。
    out = np.empty_like(x)
    for j in range(x.shape[2]):
        col = x[:, :, j]
        out[:, :, j] = (col - col.mean()) / (col.std(ddof=1) + NIGHT_NORM_EPS)
    return out


class OvernightReproEngine:
    """整夜推理引擎（路径 B）。

    Parameters
    ----------
    run_dir : Path or str
        训练产出目录（含 `config.json`、`checkpoints/model_deploy.onnx`、`scaler.json`）。
    onnx_name : str, optional
        默认 `model_deploy.onnx`（图中无归一化，由本引擎在图外补）。
    """

    def __init__(self, run_dir, onnx_name: str = "model_deploy.onnx",
                 providers: Optional[List[str]] = None):
        import onnxruntime as ort

        self.run_dir = Path(run_dir)
        with open(self.run_dir / "config.json") as f:
            self.config = json.load(f)
        ckpt = self.run_dir / "checkpoints"

        with open(ckpt / "scaler.json") as f:
            sc = json.load(f)
        self._scaler_mean = np.asarray(sc["mean_"], dtype=np.float32)
        self._scaler_scale = np.asarray(sc["scale_"], dtype=np.float32)

        self.modality: List[str] = self.config["modality"]
        self.seq_len: int = int(self.config.get("seq_len", 21))
        self.classification: str = self.config["classification"]
        self.input_layout = training_input_layout(self.modality)

        # ⚠️ 图外归一化是本引擎的职责 —— 图里必须**没有** internal_norm，否则会归一化两次
        onnx_path = ckpt / onnx_name
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} 不存在。路径 B 需要**图外归一化**变体（否则整夜统计量进不去）：\n"
                f"  python sleep_analysis/classification/inference/export_onnx.py "
                f"--run-dir {self.run_dir} --no-internal-norm --output {onnx_path}")
        self.onnx_path = onnx_path
        info_path = ckpt / (onnx_name + "_info.json")
        if info_path.exists():
            if bool(json.loads(info_path.read_text()).get("internal_norm", False)):
                raise ValueError(
                    f"{onnx_path} 图内含 internal_norm，路径 B 再在外面套一层整夜归一化会"
                    f"归一化两次。请用 `--no-internal-norm` 导出的变体。")

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts,
            providers=providers or ort.get_available_providers())
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

        if len(self._scaler_mean) != len(self.input_layout):
            raise ValueError(f"scaler 有 {len(self._scaler_mean)} 列，训练口径推出 "
                             f"{len(self.input_layout)} 列 —— 不一致")

    # ------------------------------------------------------------------

    def predict_from_frame(self, df) -> np.ndarray:
        """对一个被试的整夜特征表做推理，返回逐 epoch 的分期。

        Parameters
        ----------
        df : pd.DataFrame
            该被试的特征表（索引为 epoch 时间轴，列含训练口径的特征列）。

        Returns
        -------
        np.ndarray, shape (n_epoch,)
            argmax 后的分期（0-based）。
        """
        raw = build_train_matrix(df, self.modality)              # (n, 16) 原始
        x = build_sequences_night(raw, self.seq_len).astype(np.float32)   # (W, 21, 16)
        eps = np.finfo(np.float32).eps
        x = (x - self._scaler_mean) / (self._scaler_scale + eps)          # 第一层
        x = apply_night_norm(x)                                           # 第二层（整夜）
        logits = self.session.run([self._output_name], {self._input_name: x})[0]
        return np.argmax(logits, axis=1)
