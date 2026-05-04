import pandas as pd

# “4总_75岁及以上_手动补全”中“Delete”列添加到“4总_75岁及以上”中，匹配的依据是“住院号”这一列
df_manual = pd.read_excel("4总_75岁及以上_手动补全.xlsx")
df_combined = pd.read_excel("4总_75岁及以上.xlsx")
# 创建一个以“住院号”为索引的 lookup 表
lookup = df_manual.set_index("住院号")["Delete"].to_dict()
# 将“Delete”列添加到 df_combined 中
df_combined["Delete"] = df_combined["住院号"].map(lookup)
# 保存修改后的文件
df_combined.to_excel("4总_75岁及以上.xlsx", index=False)
print("已将 Delete 列添加到 4总_75岁及以上.xlsx 文件中，并根据 住院号 进行匹配。")

# “4总_75岁及以上_手动补全”中的“In-hospital Death”、“ASA Score”、”OS“、”Survival Status“覆盖到“4总_75岁及以上”中同名列，根据“住院号”进行匹配
df_manual = pd.read_excel("4总_75岁及以上_手动补全.xlsx")
df_combined = pd.read_excel("4总_75岁及以上.xlsx")
# 创建一个以“住院号”为索引的 lookup 表
death_lookup = df_manual.set_index("住院号")["In-hospital Death"].to_dict()
asa_lookup = df_manual.set_index("住院号")["ASA Score"].to_dict()
os_lookup = df_manual.set_index("住院号")["OS"].to_dict()
survival_lookup = df_manual.set_index("住院号")["Survival Status"].to_dict()
# 将“In-hospital Death”和“ASA Score”列覆盖到 df_combined 中
df_combined["In-hospital Death"] = (
    df_combined["住院号"].map(death_lookup).combine_first(df_combined["In-hospital Death"])
)
df_combined["ASA Score"] = df_combined["住院号"].map(asa_lookup).combine_first(df_combined["ASA Score"])
df_combined["OS"] = df_combined["住院号"].map(os_lookup).combine_first(df_combined["OS"])
df_combined["Survival Status"] = (
    df_combined["住院号"].map(survival_lookup).combine_first(df_combined["Survival Status"])
)
# 保存修改后的文件
df_combined.to_excel("4总_75岁及以上.xlsx", index=False)
print(
    "已将 4总_75岁及以上_手动补全.xlsx 中的 In-hospital Death 和 ASA Score 列覆盖到 4总_75岁及以上.xlsx 中，并根据 住院号 进行匹配。"
)

# 删去“4总_75岁及以上”中“Delete”列中不为空的行
df_combined = pd.read_excel("4总_75岁及以上.xlsx")
df_combined = df_combined[df_combined["Delete"].isna() | (df_combined["Delete"] == "")]

# 删去“In-hospital Death”为“1”和“自动出院”的行
df_combined = df_combined[~df_combined["In-hospital Death"].isin([1, "自动出院"])]
# 添加一列”Chemotherapy situation“根据”Postoperative Chemotherapy Regimen“，若为空，则Chemotherapy situation填入”0“，不为空则填”1“
df_combined["Chemotherapy situation"] = df_combined["Postoperative Chemotherapy Regimen"].apply(
    lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
)

# 保存修改后的文件
df_combined.to_excel("5总_75岁及以上.xlsx", index=False)
print("已删除 5总_75岁及以上.xlsx 中 In-hospital Death 列为 1 或 '自动出院' 的行。")