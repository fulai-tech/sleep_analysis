"""流式推理引擎 —— 13 维特征 → ONNX → 分期结果。

**本文件是流式链路的最后一段**：`StreamingPreprocessor` 产出 `(21, 13)` 特征，
这里补上 `--missing-mode` 的 `_has_*` 标志列、过训练集 scaler、喂 ONNX，取中心 epoch 的分期。

与 C++ 的对应关系
----------------
=====================  ==================================================
本文件                  C++ 侧
=====================  `ss_rt_process` 的后半段
`StreamingInferenceEngine.__init__`   `init` 时加载 engine + scaler 表
`build_input`          `[ACT..., _has_act, HRV..., ...]` 的缓冲拼装
`predict`              NPU 推理 + argmax
=====================  ==================================================

⚠️ **本引擎依赖 `--missing-mode` 训练出来的模型**（本项目当前的部署模型就是）。
   非 missing_mode 的模型输入是 13 列、没有标志位，`_flag_layout` 会返回空。

⚠️ **`internal_norm` 的已知语义问题（部署前必须知道）**
   09-11 这个模型的 `config.internal_norm = true`，即 `forward` 里有::

       mean_x = x.mean(dim=(0, 1), keepdim=True)     # batch × seq_len 上的均值
       std_x  = x.std(dim=(0, 1), keepdim=True) + 1e-5

   训练时 batch=512 个窗 → 统计量跨 512 个窗; 流式每次只喂 1 个窗 → 统计量只跨这 21 个 epoch。
   **两者不是同一个变换**, 模型收到的输入分布与训练时不同。
   `model.py` 的注释把这条路径标为"仅用于复现对照"（为复现 08-06 基线而恢复的）。
   模型的**数值精度**因此无法从训练指标外推 —— 上真机前应该用
   `--causal --no-internal-norm` 重训一版（见项目备忘）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .prep import SLOT_NAMES

#: `--missing-mode` 会加标志列的模态（与 `data_peparation.py` 的循环一致；EDR 不加）。
_FLAGGED_MODALITIES = ("ACT", "HRV", "RRV")

#: 4stage 的类别顺序（与训练 `_get_sleep_stage_labels("4stage")` 一致）。
STAGE_NAMES_4 = ["wake", "light", "deep", "rem"]


def _modality_of(slot: str) -> str:
    """按列名前缀判断槽位属于哪个模态（与训练侧的检测正则一致）。"""
    if slot.startswith("_acc"):
        return "ACT"
    if slot.startswith("_hrv") or "_hrv_" in slot:
        return "HRV"
    if "RRV" in slot:
        return "RRV"
    if "EDR" in slot:
        return "EDR"
    raise ValueError(f"无法判断槽位 {slot} 的模态")


def build_input_layout(slot_names: List[str], modality: List[str],
                       missing_mode: bool) -> List[str]:
    """由 13 槽位推出模型的 16 列布局。

    训练侧（`data_peparation.py::_extract_subj_features_raw`）的顺序是::

        for _mod in [("ACT","_acc"), ("HRV","_hrv"), ("RRV","RRV")]:
            features = concat([features, 该模态的列])
            if missing_mode: features[f"_has_{_mod.lower()}"] = 标志

    即**标志紧跟在它所属模态的特征之后**，且模态按 ACT → HRV → RRV 固定排序
    （与传入的 `modality` 列表顺序无关）。

    Returns
    -------
    list of str
        模型输入列名。缺失模态的标志列名形如 `_has_act`；非 missing_mode 时
        就是原槽位列表。
    """
    if not missing_mode:
        return list(slot_names)

    out: List[str] = []
    for mod in _FLAGGED_MODALITIES:
        if mod not in modality:
            continue
        out.extend([s for s in slot_names if _modality_of(s) == mod])
        out.append(f"_has_{mod.lower()}")
    # EDR 不在 _FLAGGED_MODALITIES 里（训练侧不加标志）
    out.extend([s for s in slot_names if _modality_of(s) == "EDR"])
    return out


class StreamingInferenceEngine:
    """加载一个训练 run 的 ONNX 模型，对 21×13 特征做推理。

    Parameters
    ----------
    run_dir : Path or str
        训练产出目录（含 `config.json`、`checkpoints/model.onnx`、`checkpoints/scaler.json`）。
    flag_values : dict, optional
        每个 `_has_*` 标志的取值。默认全部 1.0（IR-UWB 三个模态都真实存在）。
        ⚠️ 不要随手改成 0 —— `_has_act` 在训练集里是**有信息**的
        （mean=0.137，即只有约 14% 的训练样本有 ACT），改错会让模型按错误的
        模态可用性假设推理。
    providers : list of str, optional
        onnxruntime 执行提供者，默认让 onnxruntime 自选。
    """

    def __init__(self, run_dir, onnx_name: Optional[str] = None,
                 baseline: Optional[dict] = None,
                 flag_values: Optional[Dict[str, float]] = None,
                 providers: Optional[List[str]] = None):
        import json

        import onnxruntime as ort

        self.run_dir = Path(run_dir)
        with open(self.run_dir / "config.json") as f:
            self.config = json.load(f)

        ckpt = self.run_dir / "checkpoints"
        scaler_path = ckpt / "scaler.json"
        if not scaler_path.exists():
            raise FileNotFoundError(f"{scaler_path} 不存在")

        # ONNX 变体选择: 优先部署变体（图内不含 internal_norm, 归一化在图外）
        if onnx_name is None:
            onnx_name = ("model_deploy.onnx" if (ckpt / "model_deploy.onnx").exists()
                         else "model.onnx")
        onnx_path = ckpt / onnx_name
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} 不存在 —— 先跑 export_onnx.py --run-dir {self.run_dir}")

        # 图里到底有没有 internal_norm —— 由导出时写的 `_info.json` 判定。
        # 没有 info 时退回 config（保守：按有处理，交给下面的冲突检查报错）。
        # 导出脚本写的名字是 "<onnx 文件名>_info.json"（含 .onnx，如 model_deploy.onnx_info.json）
        info_path = ckpt / (onnx_name + "_info.json")
        # ⚠️ 缺 `internal_norm` 键时**回退到 config**, 不能默认 False ——
        #    旧版导出脚本不写这个键, 默认 False 会让「图里其实有 norm」被判成没有,
        #    于是图外再归一化一次（重复归一化, 输出全错）。
        _cfg_norm = bool(self.config.get("internal_norm", False))
        if info_path.exists():
            self.graph_internal_norm = bool(
                json.loads(info_path.read_text()).get("internal_norm", _cfg_norm))
        else:
            self.graph_internal_norm = _cfg_norm
        self.onnx_path = onnx_path

        with open(scaler_path) as f:
            sc = json.load(f)
        self._scaler_mean = np.asarray(sc["mean_"], dtype=np.float32)
        self._scaler_scale = np.asarray(sc["scale_"], dtype=np.float32)

        self.modality: List[str] = self.config["modality"]
        self.missing_mode: bool = bool(self.config.get("missing_mode", False))
        self.internal_norm: bool = bool(self.config.get("internal_norm", False))
        self.seq_len: int = int(self.config.get("seq_len", 21))
        self.classification: str = self.config["classification"]

        # 模型输入列布局（16 列，含 _has_*）
        self.input_layout = build_input_layout(SLOT_NAMES, self.modality, self.missing_mode)
        n_feat = len(self.input_layout)
        if n_feat != len(self._scaler_mean):
            raise ValueError(
                f"输入列数 {n_feat} 与 scaler 的 {len(self._scaler_mean)} 不符 —— "
                f"标位布局推导有误（布局: {self.input_layout}）")

        # 图外归一化的参数（图里没有 internal_norm 时必须提供）。
        # ⚠️ 必须在 `input_layout` 之后 —— `_resolve_norm` 用它校验列数。
        self._norm_mean, self._norm_std = self._resolve_norm(baseline, ckpt)

        flags = {"_has_act": 1.0, "_has_hrv": 1.0, "_has_rrv": 1.0}
        if flag_values:
            flags.update(flag_values)
        # 每列的取值与来源: 标志列取常数, 特征列运行时填
        self._flag_values = {c: float(flags[c]) for c in self.input_layout if c.startswith("_has_")}

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts,
            providers=providers or ort.get_available_providers())
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

        if self.graph_internal_norm:
            print("[WARN] 该 ONNX **图内**含 internal_norm —— 归一化统计量随 batch 组成变化, "
                  "训练(batch=512)与流式(batch=1)不是同一个变换, 精度无法从训练指标外推。\n"
                  "       部署请用 `export_onnx.py --no-internal-norm` 导出图外归一化变体。")
        elif self._norm_mean is None:
            print("[WARN] 图外归一化未启用（该模型原本也不含 internal_norm）")
        else:
            print(f"[INFO] 图外归一化已启用（固定基线，与本模型无关的 batch 组成无关）  "
                  f"onnx={onnx_path.name}")

    def _resolve_norm(self, baseline, ckpt: Path):
        """决定图外归一化用哪组 `(mean, std)`。

        部署策略（见项目备忘）：第一晚用**训练集基线**，之后用**被试自己的基线**
        （逐日更新）。本方法只负责把参数取出来；被试基线由调用方通过 `baseline=`
        传入。

        Raises
        ------
        FileNotFoundError
            图里没有 internal_norm，但又找不到基线参数 —— 归一化会缺失，
            输出必然错，所以直接报错而不是静默跳过。
        ValueError
            图里**已经**有 internal_norm 又传了基线 —— 会归一化两次。
        """
        import json

        if self.graph_internal_norm:
            if baseline is not None:
                raise ValueError(
                    "该 ONNX 图内已含 internal_norm，再传 baseline 会归一化两次。"
                    "部署请用 `export_onnx.py --no-internal-norm` 导出的模型变体。")
            return None, None

        if baseline is None:
            f = ckpt / "internal_norm_baseline.json"
            if not f.exists():
                raise FileNotFoundError(
                    f"ONNX 图中不含 internal_norm，但找不到图外归一化参数 {f}。\n"
                    f"先用 `python experiments/evaluation/fit_internal_norm_baseline.py "
                    f"--run-dir {self.run_dir}` 统计训练集基线。")
            baseline = json.loads(f.read_text())

        mean = np.asarray(baseline["mean_"], dtype=np.float32)
        std = np.asarray(baseline["std_"], dtype=np.float32)
        if len(mean) != self.n_features or len(std) != self.n_features:
            raise ValueError(f"基线有 {len(mean)} 列, 模型输入 {self.n_features} 列")
        return mean, std

    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self.input_layout)

    def build_input(self, features: np.ndarray) -> np.ndarray:
        """`(21, 13)` 特征 + 标志列 → `(1, 21, 16)` 模型输入（**未过 scaler**）。

        Parameters
        ----------
        features : np.ndarray
            形状 `(seq_len, 13)`，列序 = `prep.SLOT_NAMES`。

        Returns
        -------
        np.ndarray, shape (1, seq_len, n_features), dtype float32
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != len(SLOT_NAMES):
            raise ValueError(f"features 形状应为 (seq_len, {len(SLOT_NAMES)})，"
                             f"收到 {features.shape}")
        n = features.shape[0]
        out = np.empty((n, len(self.input_layout)), dtype=np.float32)
        slot_idx = {s: i for i, s in enumerate(SLOT_NAMES)}
        for j, col in enumerate(self.input_layout):
            if col in slot_idx:
                out[:, j] = features[:, slot_idx[col]]
            else:
                out[:, j] = self._flag_values[col]
        return out[None, :, :]

    def predict(self, features: np.ndarray) -> Dict[str, object]:
        """对 21×13 特征做一次推理。

        ⚠️ 输入的 21 个 epoch 必须**已经是模型窗**（居中于要产出的 epoch）——
           模型对每个窗口只给一个输出，对应窗口中心。本引擎不做窗口切分。

        Returns
        -------
        dict
            `logits` (4,) / `probs` (4,) / `stage` int / `stage_name` str / `confidence` float
        """
        x = self.build_input(features)
        # 第一层：训练集 scaler（固定常数，逐 epoch 广播）。
        eps = np.finfo(np.float32).eps
        x = (x - self._scaler_mean) / (self._scaler_scale + eps)

        # 第二层：`internal_norm` 的图外版本（图里没有时）。用固定参数替代批内统计量，
        # 第一晚是训练集基线，之后可以是该被试自己的基线。
        # ⚠️ `+1e-5` 必须与训练时一致（`model.py` 的 `std_x = x.std(...) + 1e-5`）。
        if self._norm_mean is not None:
            x = (x - self._norm_mean) / (self._norm_std + 1e-5)

        logits = self.session.run([self._output_name], {self._input_name: x})[0][0]

        # softmax（数值稳定版）
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        stage = int(np.argmax(logits))
        return {
            "logits": logits,
            "probs": probs,
            "stage": stage,
            "stage_name": STAGE_NAMES_4[stage] if self.classification == "4stage" else str(stage),
            "confidence": float(probs[stage]),
        }
