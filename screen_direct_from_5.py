import pandas as pd
import numpy as np
import utils

df = pd.read_excel("5总_75岁及以上.xlsx")

# TODO: 最后需要两个文件，5总_更新和5总_并发症，分别用于分析所有的病例，和异常死亡的病例
output_file = "5总_并发症.xlsx"

# Hb to numeric and check anaemia
df["Hb"] = pd.to_numeric(df["Hb"], errors="coerce")
df["Anaemia"] = df.apply(utils.is_anaemia, axis=1)

# postoperative column one-hot encoding
df["Postoperative Chemotherapy Regimen"] = (
    df["Postoperative Chemotherapy Regimen"]
    .fillna("")
    .str.strip()
    .ne("")
    .astype(int)
)

df.to_excel(output_file, index=False)