"""全局处理模式开关。

- ``causal=False``（默认）: 原版处理（复现作者/论文结果）
- ``causal=True``: 因果处理 — 数据处理只使用已见数据（实时睡眠分期用）

当前生效范围（2026-08-05 决策）:
  - ✅ feature_extraction/mesa_datasst/rrv.py::_downsample_resp
       呼吸信号滤波与降采样 — RRV 特征实测差异中位 11.7%, 不可忽略, 保留因果
  - ❌ preprocessing/rr_utils.py::process_rpoint — 已回退原版
       (HRV 特征实测差异中位 0%, 且保持与原版模型可比性; causal 实现注释保留)
  - ❌ experiments/data_handling/preprocess_shhs.py::generate_r_points_from_ecg
       — 已回退原版 (同上; causal 实现见 preprocessing/ecg_rpeaks.py)

注意: 本开关（数据生成阶段）与推理引擎 config.json 的 "causal" 键
(classification/inference/engine_torch.py / engine_onnx.py, 只控制 LSTM 序列窗口的
padding 方向) 是**两个独立开关**, 不要混淆。一个"实时可用"的模型需要两者同时满足:
  1. 特征数据用 SLEEP_CAUSAL=1 生成（本开关, 记录在输出目录 run_config.json）
  2. 模型用 --causal 训练（config.json 的 causal=true）

开关方式:
  1. 环境变量 ``SLEEP_CAUSAL=1|true|yes`` 启动时生效（推荐, spawn 多进程子进程自动继承）
  2. 进程内 ``processing_config.set_causal(True)`` 覆盖（比较脚本/单元测试用）

注意: 消费方必须用 ``import processing_config as pc; pc.causal`` 动态读取,
不要 ``from processing_config import causal``（那是 import 时快照, ``set_causal`` 之后不更新）。
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

causal = os.environ.get("SLEEP_CAUSAL", "0").lower() in ("1", "true", "yes")


def set_causal(value: bool) -> None:
    """进程内覆盖开关（默认由环境变量决定）。"""
    global causal
    causal = bool(value)


def check_run_mode(output_dir, mode: dict) -> dict:
    """目录级模式锁 — 校验/写入输出目录的 run_config.json。

    防止对已有输出目录用不同模式重跑时静默复用旧模式的文件
    （断点续跑只检查 out.exists(), 不区分模式）:

      - run_config.json 不存在 → 写入当前模式（新目录或旧目录认领）
      - 存在且模式一致       → 返回现有配置（允许断点续跑）
      - 存在且模式不一致     → 抛 RuntimeError（拒绝跨模式续跑, 提示用新 --output-dir）

    Parameters
    ----------
    output_dir : Path or str
        输出目录
    mode : dict
        至少含 causal / no_edr / script; 会补充 git_sha / created

    Returns
    -------
    dict
        生效的 run_config（现有或新写入）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = output_dir / "run_config.json"

    git_sha = "unknown"
    try:
        repo_root = Path(__file__).resolve().parents[1]  # third_party/sleep_analysis
        git_sha = (
            subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        pass

    record = {
        "causal": bool(mode.get("causal", False)),
        "no_edr": bool(mode.get("no_edr", False)),
        "script": mode.get("script", "unknown"),
        "git_sha": git_sha,
    }

    if cfg_file.exists():
        existing = json.loads(cfg_file.read_text())
        for key in ("causal", "no_edr", "script"):
            if existing.get(key) != record[key]:
                raise RuntimeError(
                    f"输出目录 {output_dir} 的 run_config.json 与本次运行模式不一致: "
                    f"现有 {key}={existing.get(key)} vs 当前 {key}={record[key]}。"
                    f"换模式必须使用新的 --output-dir 目录（或确认后删除 run_config.json 重试）。"
                )
        return existing

    record["created"] = datetime.now().isoformat(timespec="seconds")
    cfg_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"[run_config] {cfg_file} 写入: {record}")
    return record
