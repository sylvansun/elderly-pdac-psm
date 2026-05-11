import pandas as pd

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