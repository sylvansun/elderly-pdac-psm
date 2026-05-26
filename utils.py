import pandas as pd
import numpy as np


# SMD函数
def smd(df, col):
    t = df[df["treat"] == 1][col]
    c = df[df["treat"] == 0][col]
    pooled = np.sqrt((t.var() + c.var()) / 2)
    return abs((t.mean() - c.mean()) / pooled)


def smd_continuous(df, var, treat_col="treat"):
    g1 = df[df[treat_col] == 1][var]
    g0 = df[df[treat_col] == 0][var]

    mean1, mean0 = g1.mean(), g0.mean()
    std1, std0 = g1.std(), g0.std()

    pooled = np.sqrt((std1**2 + std0**2) / 2)

    if pooled == 0:
        return 0

    return (mean1 - mean0) / pooled


def smd_categorical(df, var, treat_col="treat"):
    levels = df[var].dropna().unique()

    smd_list = []

    for lv in levels:

        g1 = df[df[treat_col] == 1]
        g0 = df[df[treat_col] == 0]

        p1 = np.mean(g1[var] == lv)
        p0 = np.mean(g0[var] == lv)

        p = (p1 + p0) / 2

        if p == 0:
            continue

        smd = (p1 - p0) / np.sqrt(p * (1 - p))

        smd_list.append(abs(smd))

    return max(smd_list) if smd_list else 0


# 贫血判断函数
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


# 原发肿瘤T分期判断函数
def determine_pT(row):

    # 血管侵犯相关列
    vascular_cols = [
        "Vascular Invasion (SMA)",
        "Vascular Invasion (HA)",
        "Vascular Invasion (AA)",
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
