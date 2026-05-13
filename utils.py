import pandas as pd
import numpy as np

# 定义贫血判断函数
def is_anaemia(row):
    gender = row["Gender"]
    hb = row["Hb"]

    if pd.isna(hb):
        return 0  # Hb缺失，默认不判贫血（也可以改成 np.nan）

    if gender == 1 and hb < 120:
        return 1
    elif gender == 2 and hb < 110:
        return 1
    else:
        return 0
    
# 原发肿瘤T分期判断
def determine_pT(row):

    # 血管侵犯相关列
    vascular_cols = [
        "Vascular Invasion (SMA)",
        "Vascular Invasion (HA)",
        "Vascular Invasion (AA)"
    ]
    # 只要任意一个血管侵犯列为1 -> T4
    for col in vascular_cols:
        if str(row[col]).strip() == "1":
            return "T4"

    size = row["Tumor Size"]

    # 排除无效值
    if pd.isna(size) or size in [0, "/", ""]:
        return np.nan

    # 按肿瘤大小分期
    if size <= 2:
        return "T1"
    elif size <= 4:
        return "T2"
    else:
        return "T3"