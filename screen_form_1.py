import pandas as pd
import utils

# =======================
# 基本参数
# =======================
file_path = "开腹2025-9-25.xlsx"  # 原始文件（只读）
output_path = "1开腹_75岁及以上.xlsx"  # 输出文件（新建）

# 目标列
target_columns = [
    "住院号",
    "病人年龄",
    "病人性别",
    "身高",
    "体重",
    "病人姓名",
    "手术名称",
    "手术日期",
    "ASA麻醉评分",
    "基础疾病",
    "WBC",
    "NE",
    "RBC",
    "Hb",
    "PLT",
    "总胆红素",
    "直接胆红素",
    "总蛋白",
    "ALB",
    "血淀粉酶",
    "空腹血糖",
    "CA125",
    "CA19-9",
    "CEA",
    "AFP",
    "CA724",
    "CA242",
    "病理诊断",
    "肿瘤大小",
    "神经侵犯",
    "血管侵犯(肠系膜上静脉)",
    "血管侵犯(门静脉)",
    "血管侵犯(肠系膜上动脉)",
    "血管侵犯(肝动脉)",
    "血管侵犯(脾静脉)",
    "血管侵犯(脾动脉)",
    "血管侵犯(腹主动脉)",
    "血管侵犯(胃十二指肠动脉)",
    "血管侵犯(胃左动脉)",
    "侵犯脏器(十二指肠)",
    "侵犯脏器(肝)",
    "侵犯脏器(胆囊)",
    "侵犯脏器(胆总管)",
    "侵犯脏器(胃)",
    "侵犯脏器(小肠)",
    "侵犯脏器(横结肠)",
    "侵犯脏器(膈)",
    "侵犯脏器(脾)",
    "侵犯脏器(大网膜)",
    "AJCC分期",
    "相关并发症(胰瘘)",
    "相关并发症(胆瘘)",
    "相关并发症(胃肠瘘)",
    "相关并发症(腹腔出血)",
    "相关并发症(感染)",
    "其他并发症",
    "胰瘘分级(ISGPF)",
    "再次手术",
    "再次手术原因",
    "术后输血",
    "院内死亡",
    "术后化疗方案",
    "持续时间",
    "随访状态",
    "生存状态",
    "死亡日期",
    "OS",
    "复发时间",
    "复发部位",
    "转移时间",
    "转移部位",
]

# 目标年份 sheet
target_sheets = [str(year) for year in range(2015, 2026)]

# =======================
# 开始处理
# =======================

print("开始读取 Excel 文件...")

# 只读模式加载
xl = pd.ExcelFile(file_path)
available_sheets = xl.sheet_names

# 筛选存在的 sheet
sheets_to_process = [s for s in target_sheets if s in available_sheets]
print(f"将处理以下 sheets: {sheets_to_process}")

all_filtered_data = []

for sheet in sheets_to_process:
    print(f"\n正在处理 sheet: {sheet}")

    # 读取数据（从已加载对象读取，更快）
    df = pd.read_excel(xl, sheet_name=sheet)

    # 如果存在“病案号”，改名为“住院号”（仅内存操作）
    if "病案号" in df.columns:
        df = df.rename(columns={"病案号": "住院号"})

    # 必须存在年龄列
    if "病人年龄" not in df.columns:
        print(f"⚠️  缺失 '病人年龄' 列，跳过该 sheet")
        continue

    # 补齐缺失列
    for col in target_columns:
        if col not in df.columns:
            df[col] = "/"

    # 年龄转为数值（防止字符串/空值）
    df["病人年龄"] = pd.to_numeric(df["病人年龄"], errors="coerce")

    # 筛选年龄 >= 75
    filtered_df = df[df["病人年龄"] >= 75][target_columns]

    if not filtered_df.empty:
        all_filtered_data.append(filtered_df)
        print(f"✅ 筛选出 {len(filtered_df)} 条记录")
    else:
        print("无符合条件患者")

# =======================
# 合并并导出
# =======================

if all_filtered_data:
    final_df = pd.concat(all_filtered_data, ignore_index=True)

    # 一次性写入新文件（不会影响原文件）
    final_df.to_excel(output_path, index=False)

    print("\n============================")
    print(f"处理完成！共筛选 {len(final_df)} 条记录")
    print(f"结果已保存至: {output_path}")
    print("============================")
else:
    print("\n未找到符合年龄 ≥ 75 岁的患者记录")


file_path_wc = "微创2025-9-25.xlsx"
output_path_wc = "1微创_75岁及以上.xlsx"
sheet_name_wc = "胰腺"

# 复用开腹表的列名定义
target_columns = [
    "住院号",
    "病人年龄",
    "病人性别",
    "身高",
    "体重",
    "病人姓名",
    "手术名称",
    "手术日期",
    "ASA麻醉评分",
    "基础疾病",
    "WBC",
    "NE",
    "RBC",
    "Hb",
    "PLT",
    "总胆红素",
    "直接胆红素",
    "总蛋白",
    "ALB",
    "血淀粉酶",
    "空腹血糖",
    "CA125",
    "CA19-9",
    "CEA",
    "AFP",
    "CA724",
    "CA242",
    "病理诊断",
    "肿瘤大小",
    "神经侵犯",
    "血管侵犯(肠系膜上静脉)",
    "血管侵犯(门静脉)",
    "血管侵犯(肠系膜上动脉)",
    "血管侵犯(肝动脉)",
    "血管侵犯(脾静脉)",
    "血管侵犯(脾动脉)",
    "血管侵犯(腹主动脉)",
    "血管侵犯(胃十二指肠动脉)",
    "血管侵犯(胃左动脉)",
    "侵犯脏器(十二指肠)",
    "侵犯脏器(肝)",
    "侵犯脏器(胆囊)",
    "侵犯脏器(胆总管)",
    "侵犯脏器(胃)",
    "侵犯脏器(小肠)",
    "侵犯脏器(横结肠)",
    "侵犯脏器(膈)",
    "侵犯脏器(脾)",
    "侵犯脏器(大网膜)",
    "病理分期（AJCC）",
    "相关并发症(胰瘘)",
    "相关并发症(胆瘘)",
    "相关并发症(胃肠瘘)",
    "相关并发症(腹腔出血)",
    "相关并发症(感染)",
    "其他并发症",
    "胰瘘分级ISGPF",
    "再次手术",
    "手术原因",
    "术后输血",
    "院内死亡",
    "化疗方案",
    "持续时间",
    "随访状态",
    "生存状态",
    "死亡日期",
    "OS",
    "复发时间",
    "复发部位",
    "转移时间",
    "转移部位",
]

print(f"正在读取文件: {file_path_wc}, Sheet: {sheet_name_wc}")


try:
    # 读取指定的 "胰腺" sheet
    df_wc = pd.read_excel(file_path_wc, sheet_name=sheet_name_wc)

    # 检查核心列 '病人年龄' 和 '手术日期'
    if "病人年龄" not in df_wc.columns or "手术日期" not in df_wc.columns:
        print(f"错误: 文件中缺失 '病人年龄' 或 '手术日期' 列。")
    else:
        # 处理缺失的列
        missing_cols = [col for col in target_columns if col not in df_wc.columns]
        if missing_cols:
            print(f"注意: 缺失列 {missing_cols}，将使用 '/' 填充。")
            for col in missing_cols:
                df_wc[col] = "/"

        # 转换年龄为数值
        df_wc["病人年龄"] = pd.to_numeric(df_wc["病人年龄"], errors="coerce")
        # 转换手术日期为日期类型
        df_wc["手术日期"] = pd.to_datetime(df_wc["手术日期"], errors="coerce")

        # 筛选年龄 >= 75 且 手术日期 >= 2015-01-01
        filtered_df_wc = df_wc[(df_wc["病人年龄"] >= 75) & (df_wc["手术日期"] >= "2015-01-01")][target_columns]

        if not filtered_df_wc.empty:
            # 将日期列重新转换为字符串以便保存，防止 excel 格式问题（可选，但通常 pd.to_excel 处理得很好）
            # filtered_df_wc["手术日期"] = filtered_df_wc["手术日期"].dt.strftime('%Y-%m-%d')

            # 保存结果
            filtered_df_wc.to_excel(output_path_wc, index=False)
            print(
                f"处理完成！筛选出 {len(filtered_df_wc)} 条满足 (年龄 >= 75 且 手术日期 >= 2015) 的记录，已保存至: {output_path_wc}"
            )
        else:
            print("未找到符合复合条件（年龄 >= 75 岁 且 手术日期在 2015 年及以后）的患者记录。")

except Exception as e:
    print(f"处理过程中出错: {e}")

# 将1微创_75岁及以上文件中“化疗方案”改为“术后化疗方案”,"病理分期（AJCC）"改为"AJCC分期"
try:
    df_wc = pd.read_excel(output_path_wc)
    df_wc.rename(
        columns={
            "化疗方案": "术后化疗方案",
            "病理分期（AJCC）": "AJCC分期",
            "胰瘘分级ISGPF": "胰瘘分级(ISGPF)",
            "手术原因": "再次手术原因",
        },
        inplace=True,
    )
    df_wc.to_excel(output_path_wc, index=False)
    print(f"已将文件 {output_path_wc} 中的 '化疗方案' 列名修改为 '术后化疗方案'")
except Exception as e:
    print(f"修改列名时出错: {e}")

# 将“Gender”中的Male改为1，Female改为2
try:
    df_wc = pd.read_excel(output_path_wc)
    df_wc["病人性别"] = df_wc["病人性别"].map({"M": 1, "F": 2})
    df_wc.to_excel(output_path_wc, index=False)
    print(f"已将文件 {output_path_wc} 中的 '病人性别' 列值修改为数字")
except Exception as e:
    print(f"修改性别列值时出错: {e}")