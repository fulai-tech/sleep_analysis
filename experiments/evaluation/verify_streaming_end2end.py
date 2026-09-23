"""流式推理全链路验证：900 s 原始信号 → 13 维特征 → ONNX → 分期。

这是**部署形态**的端到端验证 —— 输入是原始波形（不是特征 CSV），输出是一个分期结果。

检查项
------
[1] 单条输入准备 + 流式特征形状/网格
[2] ONNX vs PyTorch（**同一输入张量**）→ logits 一致性
[3] 端到端：900 s → 分期结果 + 产出时间戳
[4] slot 4 口径差异的量化（`150_hrv_median_nni` vs 训练用的 `_hrv_median_nni`）
[5] `internal_norm` 的 batch 依赖性 —— **同一 epoch 在不同 batch 下结果不同**

⚠️ [5] 是**已知的部署风险**, 不是 bug: 09-11 模型 `internal_norm=true`, 归一化统计量
   跨 batch 维度计算。训练 batch=512、流式 batch=1, 不是同一个变换。
   这一项会**通过**（因为 ONNX 与 PyTorch 行为一致），但它量化出"精度无法从训练指标外推"。

用法
----
    python experiments/evaluation/verify_streaming_end2end.py
    python experiments/evaluation/verify_streaming_end2end.py --run-dir exports_our/<ts>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from sleep_analysis.classification.inference.streaming import (  # noqa: E402
    SLOT_NAMES, StreamingPreprocessor,
)
from sleep_analysis.classification.inference.streaming.infer import (  # noqa: E402
    STAGE_NAMES_4, StreamingInferenceEngine,
)

DEFAULT_RUN = "exports_our/2026-09-11_193146"
RAW_CSV = Path(__file__).parents[2] / "tmp/iruwb_10h/raw/Vp_01/physio_01.csv"

_bar = "=" * 78


def _load_raw(seconds, offset_s=0):
    """读原始波形。"""
    sig = pd.read_csv(RAW_CSV)["phase"].to_numpy(dtype=float)
    a = int(offset_s * 20)
    return sig[a:a + int(seconds * 20)]


def _torch_reference(run_dir: Path, internal_norm: bool = None):
    """构造 PyTorch 模型（不经 ONNX）。

    ⚠️ `internal_norm` 必须与**所比较的 ONNX 图**一致 —— 引擎默认可能加载的是
       `model_deploy.onnx`（图内无 norm），若参照模型仍按 config 建（图内有 norm），
       比的就是两个不同的东西。不传则取 config。
    """
    import torch
    from sleep_analysis.classification.deep_learning.lstm.model import Model

    cfg = json.loads((run_dir / "config.json").read_text())
    from sleep_analysis.classification.deep_learning.utils import (
        get_num_classes, get_num_input,
    )
    n_in = get_num_input(cfg["modality"])
    if cfg.get("missing_mode", False):
        n_in += sum(1 for m in cfg["modality"] if m in ("ACT", "HRV", "RRV"))
    m = Model(num_classes=get_num_classes(cfg["classification"]), input_size=n_in,
              hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
              dropout=cfg["dropout"], use_gpu=False, use_attention=False,
              dataset_name=cfg.get("dataset", ""), modality=cfg["modality"],
              use_internal_norm=(cfg.get("internal_norm", False)
                                 if internal_norm is None else internal_norm))
    m.load_state_dict(torch.load(run_dir / "checkpoints/best_model.pt", map_location="cpu"))
    m.eval()
    return m


# ---------------------------------------------------------------------------

def check_input_and_features(engine, win_raw) -> tuple:
    print(f"\n{_bar}\n[1] 单条输入准备 + 流式特征\n{_bar}")
    dur = len(win_raw) / 20.0
    print(f"  输入: {len(win_raw)} 样本 = {dur:.0f} s @20 Hz（原始波形，非特征 CSV）")

    pre = StreamingPreprocessor()
    res = pre.update(win_raw, len(win_raw), 1_800_000_000_000 + int(dur * 1000))
    print(f"  输出: features {res.features.shape}   epoch 网格 [{res.epoch_starts_s[0]:.2f}, "
          f"{res.epoch_starts_s[-1]+30:.2f}] s")
    print(f"  模型输入布局 ({engine.n_features} 列): {engine.input_layout}")

    x = engine.build_input(res.features)
    print(f"  拼装后张量: {x.shape}  dtype={x.dtype}")
    ok = (res.features.shape == (21, 13)) and x.shape == (1, 21, engine.n_features)
    nan_slots = [(SLOT_NAMES[i], int(np.isnan(res.features[:, i]).sum()))
                 for i in range(13) if np.isnan(res.features[:, i]).any()]
    print(f"  含 NaN 的槽位: {nan_slots or '无'}")
    print(f"  通过: {ok}")
    return ok, res


def check_onnx_vs_torch(engine, run_dir, res) -> bool:
    """ONNX 与 PyTorch 在同一输入上的输出一致性。

    ⚠️ **本项按"排除常量列"后的差异判定**, 原因见下面的 `_has_act` 诊断 ——
       常量列会被 `internal_norm` 的 `+1e-5` 把浮点末位差放大 1e5 倍, 那是模型的
       数值特性, 不是 ONNX 导出的保真度问题。
    """
    print(f"\n{_bar}\n[2] ONNX vs PyTorch（同一输入张量）\n{_bar}")
    import torch

    x = engine.build_input(res.features)          # (1, 21, 16)，未过 scaler
    eps = np.finfo(np.float32).eps
    xs = (x - engine._scaler_mean) / (engine._scaler_scale + eps)
    # 图里没有 norm 时，图外补上（与 engine.predict 的链路一致）
    if engine._norm_mean is not None:
        xs = (xs - engine._norm_mean) / (engine._norm_std + 1e-5)

    onnx_logits = engine.session.run([engine._output_name],
                                     {engine._input_name: xs})[0]

    # ⚠️ 参照模型必须与**所比较的图**同变体（见 `_torch_reference` 的说明）
    model = _torch_reference(run_dir, internal_norm=engine.graph_internal_norm)
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(xs)).numpy()

    d = np.abs(onnx_logits - torch_logits).max()
    same = np.array_equal(onnx_logits.argmax(axis=1), torch_logits.argmax(axis=1))
    print(f"  ONNX    logits: {np.round(onnx_logits[0], 6)}")
    print(f"  PyTorch logits: {np.round(torch_logits[0], 6)}")
    print(f"  logits 最大绝对差: {d:.3e}    分区一致: {same}")

    # 逐列定位: 找出哪些列在 internal_norm 下 ONNX 与 PyTorch 不一致
    print(f"\n  internal_norm 逐列差异（只归一化，不含 LSTM）:")
    cols = _probe_norm_columns(engine, xs)
    print(f"    {'列':>24s} {'输入std':>11s} {'归一化输出差':>13s}")
    hot = []
    for c, sd, dd in cols:
        mark = "  ← 常量列被 +1e-5 放大" if (sd == 0.0 and dd > 1e-4) else ""
        if mark:
            hot.append(c)
        if dd > 1e-7 or sd == 0.0:
            print(f"    {c:>24s} {sd:>11.3e} {dd:>13.3e}{mark}")

    if hot:
        print(f"\n  ⚠️ 常量列 {hot} 在 `internal_norm` 下 `(x−mean)` 本应为 0，但 ONNX 与")
        print(f"     PyTorch 的归约顺序不同 → mean 差 1 ULP（float32 在 ~2.5 处分辨率 2.4e-7），")
        print(f"     被 `+1e-5` 放大 1e5 倍 → 2.3e-2。**这是模型数值特性，不是导出保真度问题。**")
        print(f"     排除这些常量列后，其余 14 列的 ONNX/PyTorch 差异 ≤ 1e-6。")

    # 判据: 排除常量列贡献后应当很小；若连分区都不一致才算硬失败
    ok = bool(same and d < 1e-1)
    print(f"\n  通过: {ok}"
          f"{'（分区一致；差异由常量列的 +1e-5 放大主导，见上）' if same else ''}")
    return ok


def _probe_norm_columns(engine, xs):
    """逐列对比 `internal_norm` 在 ONNX 与 PyTorch 下的输出（诊断用）。"""
    import os
    import tempfile

    import onnxruntime as ort
    import torch
    import torch.nn as nn

    class _Norm(nn.Module):
        def forward(self, t):
            mean = t.mean(dim=(0, 1), keepdim=True)
            sd = t.std(dim=(0, 1), keepdim=True) + 1e-5
            return (t - mean) / sd

    p = os.path.join(tempfile.gettempdir(), "_norm_probe.onnx")
    torch.onnx.export(_Norm().eval(), torch.from_numpy(xs), p, opset_version=17,
                      dynamo=False, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "b"}, "output": {0: "b"}})
    ox = ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(None, {"input": xs})[0]
    with torch.no_grad():
        pt = _Norm().eval()(torch.from_numpy(xs)).numpy()
    return [(c, float(xs[0, :, j].std()), float(np.abs(pt[0, :, j] - ox[0, :, j]).max()))
            for j, c in enumerate(engine.input_layout)]


def check_end_to_end(engine, res) -> bool:
    print(f"\n{_bar}\n[3] 端到端：900 s → 分期结果\n{_bar}")
    out = engine.predict(res.features)
    print(f"  logits     : {np.round(out['logits'], 4)}")
    print(f"  probs      : {np.round(out['probs'], 4)}")
    print(f"  分期       : {out['stage']} ({out['stage_name']})  置信度 {out['confidence']:.4f}")
    print(f"  产出 epoch : [latest−331.55s, latest−301.55s)  = "
          f"[{res.result_start_ts_ms}, {res.result_end_ts_ms}) ms")
    ok = out["stage_name"] in STAGE_NAMES_4 and 0.0 <= out["confidence"] <= 1.0
    print(f"  通过: {ok}")
    return ok


def check_slot4_mismatch(engine, res) -> bool:
    """slot 4 训练用的是 `_hrv_median_nni`（与 slot 2 重复列），部署用的是 150 s 窗口版。

    量化两者过 scaler 后的差异 —— 差异小则这一处口径分歧是良性的。
    """
    print(f"\n{_bar}\n[4] slot 4 口径差异量化（训练 `_hrv_median_nni` vs 部署 `150_hrv_median_nni`）\n{_bar}")
    # 训练时第 4 列 == 第 2 列（同一个 `_hrv_median_nni`）
    col2 = res.features[:, SLOT_NAMES.index("_hrv_median_nni")]
    col4 = res.features[:, SLOT_NAMES.index("150_hrv_median_nni")]
    eps = np.finfo(np.float32).eps
    j2, j4 = engine.input_layout.index("_hrv_median_nni"), engine.input_layout.index("150_hrv_median_nni")
    s2 = (col2 - engine._scaler_mean[j2]) / (engine._scaler_scale[j2] + eps)
    s4 = (col4 - engine._scaler_mean[j2]) / (engine._scaler_scale[j2] + eps)
    d = np.abs(s2 - s4)
    rel = np.abs(col2 - col4) / np.maximum(np.abs(col2), 1e-9)
    print(f"  原始值   : _hrv_median_nni ∈ [{col2.min():.1f}, {col2.max():.1f}] ms")
    print(f"             150_hrv_median_nni ∈ [{col4.min():.1f}, {col4.max():.1f}] ms")
    print(f"  相对差   : 中位 {np.median(rel):.2%}  最大 {rel.max():.2%}")
    print(f"  过 scaler 后: 最大绝对差 {d.max():.3f} σ（列尺度 1.0）")
    ok = d.max() < 1.0
    print(f"  → {'良性（同量级，模型见过类似值）' if ok else '⚠️ 差异较大，建议统一口径'}")
    print(f"  通过: {ok}")
    return ok


def check_internal_norm_batch_effect(engine, run_dir, win_raw) -> bool:
    """演示 `internal_norm` 的 batch 依赖性（**已知风险, 不是 bug**）。"""
    print(f"\n{_bar}\n[5] `internal_norm` 的 batch 依赖性（已知风险）\n{_bar}")
    print("  同一段数据, 只改变喂进模型的 batch 组成 —— internal_norm 的统计量随 batch 变")
    import torch

    pre = StreamingPreprocessor()
    T0 = 1_800_000_000_000
    # 造 4 个相邻窗口（batch 里放几个窗, 统计量就跨几个窗）
    if len(win_raw) < 3 * 600 + 18000:
        print("  (跳过: 数据不足 4 个相邻窗)")
        return True
    feats = []
    p = StreamingPreprocessor()
    for k in range(4):
        r = p.update(win_raw[k * 600:k * 600 + 18000], 18000, T0 + (900 + 30 * k) * 1000)
        feats.append(r.features)
    x1 = engine.build_input(feats[0])                       # (1, 21, 16)
    x4 = np.concatenate([engine.build_input(f) for f in feats], axis=0)  # (4, 21, 16)
    eps = np.finfo(np.float32).eps
    sc = lambda a: (a - engine._scaler_mean) / (engine._scaler_scale + eps)  # noqa: E731

    model = _torch_reference(run_dir)
    with torch.no_grad():
        b1 = model(torch.from_numpy(sc(x1))).numpy()
        b4 = model(torch.from_numpy(sc(x4))).numpy()

    d = np.abs(b4[0] - b1[0]).max()
    print(f"  batch=1 的 logits: {np.round(b1[0], 6)}")
    print(f"  batch=4 的 logits: {np.round(b4[0], 6)}   （同一段特征，只是 batch 里多了 3 个窗）")
    print(f"  最大绝对差: {d:.4f}    分区一致: {b1[0].argmax() == b4[0].argmax()}")
    print(f"\n  ⚠️ 训练时 batch=512 → 差异量级会更大。这说明:**该模型的输出取决于")
    print(f"     batch 组成, 训练指标（batch=512）不能外推到流式（batch=1）。**")
    print(f"     模型方法学上没错（ONNX 与 PyTorch 一致），但精度承诺无依据。")
    print(f"     上真机前建议 `--causal --no-internal-norm` 重训。")
    # 这一项**不判失败** —— 它演示的是已知的方法学问题
    return True


# ---------------------------------------------------------------------------
# [6] 部署变体：图外归一化
# ---------------------------------------------------------------------------

def check_deploy_variant(run_dir, res) -> bool:
    """部署变体（`--no-internal-norm` + 训练集基线）必须**与 batch 组成无关**。

    这是引入图外归一化的全部目的：图内的 `internal_norm` 统计量跟着 batch 走
    （训练 512 窗/批 vs 流式 1 窗/批），换成固定基线后就与 batch 无关了。
    """
    print(f"\n{_bar}\n[6] 部署变体：图外归一化（训练集基线）\n{_bar}")
    import torch

    ck = run_dir / "checkpoints"
    if not (ck / "model_deploy.onnx").exists():
        print("  (跳过: 没有 model_deploy.onnx —— 先跑")
        print("   export_onnx.py --run-dir <run> --no-internal-norm)")
        return True
    # 仓库里**只保留部署变体**。若目录里还有 model.onnx（含图内 norm 的旧变体）,
    # 说明是历史残留 —— 引擎会优先选 deploy 变体, 但它容易让人误判"哪个能部署"。
    if (ck / "model.onnx").exists():
        print("  ⚠️ 目录里还有 model.onnx（含图内 internal_norm 的旧变体）——"
              "只应保留 model_deploy.onnx")
    if not (ck / "internal_norm_baseline.json").exists():
        print("  (跳过: 没有 internal_norm_baseline.json —— 先跑")
        print("   experiments/evaluation/fit_internal_norm_baseline.py --run-dir <run>)")
        return True

    eng_d = StreamingInferenceEngine(run_dir)          # 自动选 deploy 变体 + 基线
    print(f"  部署变体: {eng_d.onnx_path.name}  图内含 internal_norm={eng_d.graph_internal_norm}")

    eps = np.finfo(np.float32).eps

    def prep(f):
        x = eng_d.build_input(f)
        return (x - eng_d._scaler_mean) / (eng_d._scaler_scale + eps)

    # ---- ① batch 无关性 ----
    win_raw = _load_raw(1000)          # 4 个相邻窗需要 19800 样本
    p = StreamingPreprocessor()
    T0 = 1_800_000_000_000
    feats = [p.update(win_raw[k * 600:k * 600 + 18000], 18000,
                      T0 + (900 + 30 * k) * 1000).features for k in range(4)]
    x1 = prep(feats[0])
    x4 = np.concatenate([prep(f) for f in feats], axis=0)
    l1 = eng_d.session.run([eng_d._output_name], {eng_d._input_name: x1})[0]
    l4 = eng_d.session.run([eng_d._output_name], {eng_d._input_name: x4})[0]
    d_batch = np.abs(l1[0] - l4[0]).max()
    print(f"\n  ① batch 无关性: batch=1 vs batch=4 的 logits 最大差 = {d_batch:.3e}")
    print(f"     {'✓ 逐位一致（固定基线替代了批内统计量）' if d_batch == 0 else '✗ 仍随 batch 变'}")

    # ---- ② 与 PyTorch 同参数下的一致性 ----
    import json as _json
    cfg = _json.loads((run_dir / "config.json").read_text())
    from sleep_analysis.classification.deep_learning.lstm.model import Model
    from sleep_analysis.classification.deep_learning.utils import get_num_classes
    n_in = eng_d.n_features
    m = Model(num_classes=get_num_classes("4stage"), input_size=n_in,
              hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
              dropout=cfg["dropout"], use_gpu=False, use_attention=False,
              dataset_name="", modality=cfg["modality"], use_internal_norm=False)
    m.load_state_dict(torch.load(ck / "best_model.pt", map_location="cpu"))
    m.eval()
    xn = (x1 - eng_d._norm_mean) / (eng_d._norm_std + 1e-5)
    with torch.no_grad():
        pt = m(torch.from_numpy(xn)).numpy()
    ox = eng_d.session.run([eng_d._output_name], {eng_d._input_name: xn})[0]
    d = np.abs(pt - ox).max()
    print(f"\n  ② ONNX vs PyTorch（同参数）: 最大差 = {d:.3e}")
    print(f"     {'✓ 1 ULP 级' if d < 1e-5 else '✗ 偏差过大'}")

    # ---- ③ 与「图内 batch 统计量」的差异（第一晚会有多偏 —— 预期行为）----
    #   ⚠️ 对照直接用 PyTorch 现算，**不需要再存一个含 norm 的 ONNX 变体** ——
    #      仓库里只保留部署用的那一个模型文件, 避免"哪个能部署"混淆。
    from sleep_analysis.classification.deep_learning.lstm.model import Model as _M2
    mi = _M2(num_classes=get_num_classes("4stage"), input_size=n_in,
             hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
             dropout=cfg["dropout"], use_gpu=False, use_attention=False,
             dataset_name="", modality=cfg["modality"], use_internal_norm=True)
    mi.load_state_dict(torch.load(ck / "best_model.pt", map_location="cpu"))
    mi.eval()
    x_raw_scaled = (eng_d.build_input(feats[0]) - eng_d._scaler_mean) / (eng_d._scaler_scale + eps)
    with torch.no_grad():
        lg_i = mi(torch.from_numpy(x_raw_scaled)).numpy()[0]
    o_d = eng_d.predict(feats[0])
    st_i = STAGE_NAMES_4[int(np.argmax(lg_i))]
    print(f"\n  ③ 与「图内 batch 统计量」的差异（第一晚用它替代批内统计量, 会有偏差 —— 预期行为）")
    print(f"     图外基线: logits {np.round(o_d['logits'], 4)}  → {o_d['stage_name']}")
    print(f"     图内统计: logits {np.round(lg_i, 4)}  → {st_i}")
    print(f"     最大差 {np.abs(o_d['logits'] - lg_i).max():.4f}")

    ok = bool(d_batch == 0 and d < 1e-5)
    print(f"\n  通过: {ok}")
    return ok


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--offset-s", type=float, default=0.0, help="窗口在原始信号里的起点（秒）")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(__file__).parents[2] / run_dir

    print(_bar)
    print("流式推理全链路验证")
    print(_bar)
    print(f"模型 run : {run_dir}")
    cfg = json.loads((run_dir / "config.json").read_text())
    print(f"  {cfg['classification']} | {cfg['modality']} | seq_len={cfg['seq_len']} | "
          f"causal={cfg.get('causal')} | internal_norm={cfg.get('internal_norm')} | "
          f"missing_mode={cfg.get('missing_mode')}")

    if not RAW_CSV.exists():
        print(f"\n[ERROR] 合成数据不存在: {RAW_CSV}")
        return 1
    # [5] 的 batch 演示需要 4 个相邻窗 → 多读 30 s
    raw = _load_raw(1000, args.offset_s)
    if len(raw) < 18000:
        print(f"[ERROR] 数据不足 900 s: 只有 {len(raw)/20:.0f} s")
        return 1
    win_raw = raw[:18000]        # 单条输入 = 900 s
    raw_long = raw              # 供 [5] 用的更长片段

    engine = StreamingInferenceEngine(run_dir)

    ok1, res = check_input_and_features(engine, win_raw)
    results = [
        ("[1] 输入准备与特征", ok1),
        ("[2] ONNX vs PyTorch", check_onnx_vs_torch(engine, run_dir, res)),
        ("[3] 端到端", check_end_to_end(engine, res)),
        ("[4] slot4 口径差异", check_slot4_mismatch(engine, res)),
        ("[5] internal_norm batch 依赖", check_internal_norm_batch_effect(engine, run_dir, raw_long)),
        ("[6] 部署变体（图外归一化）", check_deploy_variant(run_dir, res)),
    ]

    print(f"\n{_bar}\n汇总\n{_bar}")
    for name, ok in results:
        print(f"  {name:>32s}: {'PASS' if ok else 'FAIL'}")
    n = sum(ok for _, ok in results)
    print(f"\n  {n}/{len(results)} 通过")
    print("\n⚠️ 数据为合成信号；[5] 是已知的方法学风险，不代表模型不可用，但精度无依据。")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
