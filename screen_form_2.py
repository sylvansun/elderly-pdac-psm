import os
import pandas as pd

BASE_DIR = os.getcwd()
TOTAL_PATH = os.path.join(BASE_DIR, "瑞金总表.xlsx")

# （输入文件, 输出文件）
TASKS = [
    ("1开腹_75岁及以上.xlsx", "2开腹_75岁及以上.xlsx"),
    ("1微创_75岁及以上.xlsx", "2微创_75岁及以上.xlsx"),
]

print("读取总表（交付总表）...")
df_total = pd.read_excel(TOTAL_PATH, sheet_name="交付总表", header=1)

# 构建 lookup 表
df_total["_key"] = df_total["住院号"].astype(str).str.strip()
df_lookup = df_total.drop_duplicates(subset="_key").set_index("_key")


def process(input_name, output_name):

    input_path = os.path.join(BASE_DIR, input_name)
    output_path = os.path.join(BASE_DIR, output_name)

    print(f"\n{'='*60}")
    print(f"正在处理：{input_name}")

    # 读取 1 文件
    df = pd.read_excel(input_path, sheet_name="Sheet1")

    # =========================
    # 一、数据清洗（基于 1 文件）
    # =========================

    # 1️⃣ AJCC分期：只保留类型部分
    if "AJCC分期" in df.columns:
        df["AJCC分期"] = df["AJCC分期"].astype(str).str.extract(r"^([A-Z]+[a-z]*)")

    # 2️⃣ ASA麻醉评分：转为数值
    if "ASA麻醉评分" in df.columns:
        df["ASA麻醉评分"] = df["ASA麻醉评分"].replace({"Ⅲ": 3, "III": 3, "Ⅱ": 2, "II": 2, "Ⅰ": 1, "I": 1})

    # 3️⃣ 肿瘤大小：如果是 2*1.5*0.5 这种，只保留第一个数字
    if "肿瘤大小" in df.columns:
        df["肿瘤大小"] = df["肿瘤大小"].astype(str).apply(lambda x: x.split("*")[0] if "*" in x else x)

    # 4️⃣ BMI 计算
    if "身高" in df.columns and "体重" in df.columns:
        df["BMI"] = (df["体重"] / (df["身高"] / 100) ** 2).round(2)

    # =========================
    # 二、匹配总表补充随访信息
    # =========================

    df["_key"] = df["住院号"].astype(str).str.strip()

    # 填充前统计
    before_death = df["死亡日期"].notna().sum() if "死亡日期" in df.columns else 0
    before_recur = df["复发时间"].notna().sum() if "复发时间" in df.columns else 0
    before_type = df["复发类型"].notna().sum() if "复发类型" in df.columns else 0
    before_surv = df["随访概要：生存"].notna().sum() if "随访概要：生存" in df.columns else 0
    before_stat = df["随访概要：状态"].notna().sum() if "随访概要：状态" in df.columns else 0

    # ---- 死亡日期 ----
    if "死亡日期" in df.columns:
        from_total_death = df["_key"].map(df_lookup["死亡时间"])
        mask = df["死亡日期"].isna()
        df.loc[mask, "死亡日期"] = from_total_death[mask].values

    # ---- 复发时间 ----
    if "复发时间" in df.columns:
        from_total_recur = df["_key"].map(df_lookup["复发时间"])
        mask = df["复发时间"].isna()
        df.loc[mask, "复发时间"] = from_total_recur[mask].values

    # ---- 复发类型 ----
    if "复发类型" not in df.columns:
        df["复发类型"] = None

    from_total_type = df["_key"].map(df_lookup["复发类型"])
    mask = df["复发类型"].isna()
    df.loc[mask, "复发类型"] = from_total_type[mask].values

    # ---- 随访概要：生存 / 状态 ----
    for new_col, src_col in [("随访概要：生存", "生存"), ("随访概要：状态", "状态")]:
        if new_col not in df.columns:
            df[new_col] = None
        from_total = df["_key"].map(df_lookup[src_col])
        mask = df[new_col].isna()
        df.loc[mask, new_col] = from_total[mask].values

    # 调整列顺序（插在复发时间后）
    NEW_COLS = ["复发类型", "随访概要：生存", "随访概要：状态"]
    if "复发时间" in df.columns:
        cols = df.columns.tolist()
        for c in NEW_COLS:
            if c in cols:
                cols.remove(c)
        idx = cols.index("复发时间") + 1
        for c in reversed(NEW_COLS):
            cols.insert(idx, c)
        df = df[cols]

    # 删除辅助列
    df.drop(columns=["_key"], inplace=True)

    # =========================
    # 三、保存为 2 文件
    # =========================

    df.to_excel(output_path, index=False, sheet_name="Sheet1")
    print(f"已生成：{output_name}")

    # 填充后统计
    after_death = df["死亡日期"].notna().sum() if "死亡日期" in df.columns else 0
    after_recur = df["复发时间"].notna().sum() if "复发时间" in df.columns else 0
    after_type = df["复发类型"].notna().sum()
    after_surv = df["随访概要：生存"].notna().sum()
    after_stat = df["随访概要：状态"].notna().sum()

    print("\n字段填充统计：")
    print(f"死亡日期：{before_death} → {after_death}")
    print(f"复发时间：{before_recur} → {after_recur}")
    print(f"复发类型：{before_type} → {after_type}")
    print(f"随访概要：生存：{before_surv} → {after_surv}")
    print(f"随访概要：状态：{before_stat} → {after_stat}")


# 执行所有任务
for input_name, output_name in TASKS:
    process(input_name, output_name)

print("\n全部处理完成！")