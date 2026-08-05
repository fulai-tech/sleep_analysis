"""EDR 占位表生成 — 屏蔽 EDR 处理管线时的公共工具。

屏蔽 EDR（--no-edr）时，跳过 EDR 特征提取，但下游的
check_resp_features → 列名 rename → align_datastreams → 合并 等代码完全不变，
只需在读入点用本模块的 make_edr_placeholder() 生成全 0 占位表替入。

占位值选择常数 0.0 而非 NaN/字符串的原因:
  - NaN: 会触发下游 dropna() 把整张表删空 (preprocess_mesa.py:105, merge_features 的 .dropna())，
         且 scaler 拟合出 NaN mean/std
  - 字符串: 推理端 np.float32 强转直接崩溃
  - 0.0: 经 scaler 后恒为 0，模型等价"该模态无信号"，所有 dropna/断言/交集逻辑原样通过
"""

import pandas as pd

from sleep_analysis.preprocessing.mesa_dataset.respiration import check_resp_features


def make_edr_placeholder(resp_features: pd.DataFrame) -> pd.DataFrame:
    """从 RRV 特征表生成 EDR 占位表。

    结构 = RRV 特征表经过 check_resp_features 后, 列名 RRV→EDR, 非 epoch 列全部填 0.0。

    这样:
      - 返回的表已经是最终形态, 下游的 check_resp_features / RRV→EDR rename 幂等通过
      - 行数与 epoch 集合与 RRV 表一致, align_datastreams 交集、行数断言、合并全部不变

    Parameters
    ----------
    resp_features : pd.DataFrame
        RRV 特征表 (respiration_features_raw 或 respiration_features_clean), 含 epoch 列

    Returns
    -------
    pd.DataFrame
        与 RRV 表同构的全 0 占位表, 列名为 EDR
    """
    edr = check_resp_features(resp_features.copy())
    edr.columns = [col.replace("RRV", "EDR") for col in edr.columns]
    for col in edr.columns:
        if col != "epoch":  # 保留 epoch 列, 供下游时间对齐使用
            edr[col] = 0.0
    return edr
