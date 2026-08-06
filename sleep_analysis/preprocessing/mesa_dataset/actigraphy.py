def process_actigraphy(df_actigraph, df_psg, overlap, mesa_id):
    # overlap sample of Actigraphy + PSG/HR/Respiration
    # 20260805: overlap 表未覆盖部分被试 (如 6810, 全量 2055 被试中
    # overlap 表仅 1835 行), 原实现 int(overlap["line"][row]) 在空匹配时对空
    # Series 做 int() 崩溃; 改为显式检查缺失并抛明确错误 (worker 会跳过该被试)
    row = overlap[overlap["mesaid"] == mesa_id].index
    if len(row) == 0:
        raise ValueError(f"mesaid {mesa_id} not found in overlap table (cannot align actigraphy)")
    start_idx = int(overlap["line"].loc[row[0]])
    df_actigraph["line"] = df_actigraph["line"] - start_idx + 1

    # Cut actigraphy data to PSG starting point
    df_actigraph = df_actigraph.truncate(before=start_idx - 1, after=start_idx - 2 + df_psg.size).reset_index(drop=True)

    return df_actigraph[["line", "activity", "linetime"]]
