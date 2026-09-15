"""IR-UWB 体动特征提取。

**本文件是自包含副本, 不 import 任何 MESA/D04 模块。**

来源: ``feature_extraction/mesa_datasst/actigraphy.py::calc_actigraph_features``
（MESA 的 370 维体动特征）。复制而非 import 的原因: IR-UWB 管线要保持独立,
且需要增加 ``causal`` 参数 —— 直接改 MESA 的文件会影响已产出的 MESA 数据集。

⚠️ **与上游的差异必须保持最小**: 非因果路径（``causal=False``）应当与
``mesa_datasst/actigraphy.py`` 逐位一致 —— 这是训练侧 13 槽位对齐的前提,
也是从 MESA 预训练模型迁移权重的前提。要核对一致性可以跑:

    python -c "
    import numpy as np, pandas as pd
    from sleep_analysis.feature_extraction.mesa_datasst.actigraphy import calc_actigraph_features as mesa
    from sleep_analysis.preprocessing.iruwb.actigraphy import calc_actigraph_features as ours
    s = pd.Series(np.random.RandomState(0).rand(500))
    print(mesa(s).equals(ours(s)))   # 应为 True
    "

``causal=True`` 时 ``_centered_*`` 族改用尾部窗口。**注意列名在 causal 模式下
名不副实**（不再是中心窗口）—— 保留列名是为了让两个模式的列集合一致, 便于直接对比。
"""

import numpy as np
import pandas as pd


def calc_actigraph_features(series_actigraph: pd.Series, windows_size: int = 20,
                            causal: bool = False) -> pd.DataFrame:
    """逐窗口计算体动时序特征（370 维）。

    Parameters
    ----------
    series_actigraph : pd.Series
        逐 epoch 的体动量序列。
    windows_size : int
        最大滚动窗长（epoch 数）。窗长 1..windows_size-1 逐个计算。
    causal : bool
        True 时 ``_centered_*`` 族改用尾部窗口（只用已见数据）。
        默认 False = 与 MESA 的 ``calc_actigraph_features`` 行为一致。

    Returns
    -------
    pd.DataFrame
        370 列特征。
    """
    center_flag = not causal
    df_features = pd.DataFrame()
    for win_size in np.arange(1, windows_size):
        df_features["_acc_mean_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).mean().fillna(0.0)
        )
        df_features["_acc_mean_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=center_flag, min_periods=1).mean().fillna(0.0)
        )
        df_features["_acc_median_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).median().fillna(0.0)
        )
        df_features["_acc_median_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=center_flag, min_periods=1).median().fillna(0.0)
        )

        df_features["_acc_std_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).std().fillna(0.0)
        )
        df_features["_acc_std_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=center_flag, min_periods=1).std().fillna(0.0)
        )

        df_features["_acc_max_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).max().fillna(0.0)
        )
        df_features["_acc_max_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=center_flag, min_periods=1).max().fillna(0.0)
        )

        df_features["_acc_min_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).min().fillna(0.0)
        )
        df_features["_acc_min_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=center_flag, min_periods=1).min().fillna(0.0)
        )

        df_features["_acc_var_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).var().fillna(0.0)
        )
        df_features["_acc_var_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=center_flag, min_periods=1).var().fillna(0.0)
        )

        df_features["_acc_nat_%d" % win_size] = (
            ((series_actigraph >= 50) & (series_actigraph < 100))
            .rolling(window=win_size, center=False, min_periods=1)
            .sum()
            .fillna(0.0)
        )
        df_features["_acc_nat_centered_%d" % win_size] = (
            ((series_actigraph >= 50) & (series_actigraph < 100))
            .rolling(window=win_size, center=center_flag, min_periods=1)
            .sum()
            .fillna(0.0)
        )

        df_features["_acc_anyact_%d" % win_size] = (
            (series_actigraph > 0).rolling(window=win_size, center=False, min_periods=1).sum().fillna(0.0)
        )
        df_features["_acc_anyact_centered_%d" % win_size] = (
            (series_actigraph > 0).rolling(window=win_size, center=center_flag, min_periods=1).sum().fillna(0.0)
        )

        if win_size > 3:
            df_features["_acc_skew_%d" % win_size] = (
                series_actigraph.rolling(window=win_size, center=False, min_periods=1).skew().fillna(0.0)
            )
            df_features["_acc_skew_centered_%d" % win_size] = (
                series_actigraph.rolling(window=win_size, center=center_flag, min_periods=1).skew().fillna(0.0)
            )
            #
            df_features["_acc_kurt_%d" % win_size] = (
                series_actigraph.rolling(window=win_size, center=False, min_periods=1).kurt().fillna(0.0)
            )
            df_features["_acc_kurt_centered_%d" % win_size] = (
                series_actigraph.rolling(window=win_size, center=center_flag, min_periods=1).kurt().fillna(0.0)
            )
    df_features["_acc_Act"] = series_actigraph.fillna(0.0)
    df_features["_acc_LocAct"] = (series_actigraph + 1).apply(np.log).fillna(0.0)

    return df_features.reset_index(drop=True)
