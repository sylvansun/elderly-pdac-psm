import pandas as pd
import numpy as np
import os


def process_file(file_path):
    print(f"Processing: {file_path}")
    df = pd.read_excel(file_path)

    initial_count = len(df)

    # 统一格式
    df["死亡日期"] = df["死亡日期"].fillna("").astype(str).str.strip()
    df["随访概要：生存"] = df["随访概要：生存"].fillna("").astype(str).str.strip()

    # 判断死亡日期是否为空
    null_death_mask = df["死亡日期"] == ""

    # 判断是否为“生存”
    is_alive_mask = df["随访概要：生存"] == "生存"

    # 如果死亡日期为空 且 生存 → 填入“生存”
    df.loc[null_death_mask & is_alive_mask, "死亡日期"] = "生存"

    # 仅保留处理后死亡日期不为空的
    df_processed = df[df["死亡日期"] != ""].copy()

    final_count = len(df_processed)
    print(f"Initial: {initial_count}, Final: {final_count}, Removed: {initial_count - final_count}")

    # ===== 自动把文件名前缀 2 → 3 =====
    base_name = os.path.basename(file_path)

    if base_name.startswith("2"):
        new_base_name = "3" + base_name[1:]
    else:
        new_base_name = "3_" + base_name  # 如果不是2开头，自动加3_防止错误

    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, new_base_name)

    df_processed.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print("-" * 50)


# 执行
process_file("2开腹_75岁及以上.xlsx")
process_file("2微创_75岁及以上.xlsx")