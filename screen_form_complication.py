import pandas as pd

df = pd.read_excel("4总_75岁及以上.xlsx")

keep_values = ["术后并发症死亡", "出院后死亡", "院内死亡？"]

df_filtered = df[
    df["Delete"].isna() |
    (df["Delete"].astype(str).str.strip() == "") |
    (df["Delete"].isin(keep_values))
]

df_filtered.to_excel("5总_并发症.xlsx", index=False)