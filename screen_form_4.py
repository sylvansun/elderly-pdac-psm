import pandas as pd
import numpy as np
import os
import re


def calculate_os_feature(df):
    if "死亡日期" not in df.columns or "手术日期" not in df.columns:
        print("缺失 '死亡日期' 或 '手术日期' 列")
        return df

    df["死亡日期"] = df["死亡日期"].astype(str).str.strip()

    def format_death_date(x):

        if x in ["生存", "nan", "/", "None", ""]:
            return x

        # 已经是标准日期（包括带时间）
        try:
            dt = pd.to_datetime(x, errors="raise")
            return dt.strftime("%Y-%m-%d")
        except:
            pass

        # 处理 2023/3/不清楚 这种情况
        x = x.replace("不清楚", "15")

        parts = re.split(r"[/-]", x)

        try:
            year = parts[0]
            month = parts[1].zfill(2)

            if len(parts) < 3 or not parts[2].isdigit():
                day = "15"
            else:
                day = parts[2].zfill(2)

            return f"{year}-{month}-{day}"

        except:
            return np.nan

    df["死亡日期"] = df["死亡日期"].apply(format_death_date)

    return df


def calculate_os(df):

    df["死亡日期"] = df["死亡日期"].fillna("").astype(str).str.strip()

    censor_date = pd.to_datetime("2025-09-25")

    is_alive = df["死亡日期"] == "生存"

    death_date = pd.to_datetime(df["死亡日期"], errors="coerce")
    surgery_date = pd.to_datetime(df["手术日期"], errors="coerce")

    df["OS"] = np.nan

    death_date = pd.to_datetime(df["死亡日期"], errors="coerce").dt.normalize()
    surgery_date = pd.to_datetime(df["手术日期"], errors="coerce").dt.normalize()

    df.loc[~is_alive, "OS"] = (death_date - surgery_date).dt.days + 1
    df.loc[is_alive, "OS"] = (censor_date - surgery_date).dt.days + 1

    df.loc[df["OS"] < 0, "OS"] = 0

    df["生存状态"] = 1
    df.loc[is_alive, "生存状态"] = 0

    df.loc[df["OS"].isna(), "OS"] = "数据缺失"

    return df


files_to_update = ["3开腹_75岁及以上.xlsx", "3微创_75岁及以上.xlsx"]

for file_name in files_to_update:

    if os.path.exists(file_name):

        print(f"正在计算 {file_name} 的 OS...")

        df = pd.read_excel(file_name)

        df = calculate_os_feature(df)
        df = calculate_os(df)

        df.to_excel(file_name, index=False)

        print(f"✅ {file_name} 任务完成，OS 列已更新。")

    else:
        print(f"⚠️ 文件 {file_name} 不存在，请检查前序步骤。")

# 查看"3开腹_75岁及以上.xlsx", "3微创_75岁及以上.xlsx"的列名是否相同
file1 = "3开腹_75岁及以上.xlsx"
file2 = "3微创_75岁及以上.xlsx"
# 将file1和file2根据列名合并，并在合并后的数据框中添加一列“手术方式”，对于file1的记录填入“OPEN”，对于file2的记录填入“MIS”
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)
df1["Surgical approach"] = "OPEN"
df2["Surgical approach"] = "MIS"
# 合并数据框
combined_df = pd.concat([df1, df2], ignore_index=True)
# 保存合并后的数据框到新的Excel文件
combined_df.to_excel("4总_75岁及以上.xlsx", index=False)
print("已将两个文件合并为 4总_75岁及以上.xlsx，并添加了手术方式列。")

# 查看4总_75岁及以上的列名
combined_df = pd.read_excel("4总_75岁及以上.xlsx")
print("4总_75岁及以上.xlsx 的列名如下：")
print(combined_df.columns.tolist())


# 将combined_df中”病人年龄“改为”Age“，”病人性别“改为”Gender“，”身高“改为”Height“，”体重“改为”Weight“，”病人姓名“改为”Name“，”手术日期“改为”Surgery Date“，”ASA麻醉评分“改为”ASA Score“，”基础疾病“改为”Comorbidities“，”病理诊断“改为”Pathological Diagnosis“，”肿瘤大小“改为”Tumor Size“，”神经侵犯“改为”Nerve Invasion“，”血管侵犯(肠系膜上静脉)“改为”Vascular Invasion (SMV)“，
# ”血管侵犯(门静脉)“改为”Vascular Invasion (PV)“，”血管侵犯(肠系膜上动脉)“改为”Vascular Invasion (SMA)“，”血管侵犯(肝动脉)“改为”Vascular Invasion (HA)“，”血管侵犯(脾静脉)“改为”Vascular Invasion (SV)“，”血管侵犯(脾动脉)“改为”Vascular Invasion (SA)“，”血管侵犯(腹主动脉)“改为”Vascular Invasion (AA)“，”血管侵犯(胃十二指肠动脉)“改为”Vascular Invasion (GDA)“，”血管侵犯(胃左动脉)“改为”Vascular Invasion (LGA)“，
# ”侵犯脏器(十二指肠)“改为”Organ Invasion (Duodenum)“，”侵犯脏器(肝)“改为”Organ Invasion (Liver)“，”侵犯脏器(胆囊)“改为”Organ Invasion (Gallbladder)“，”侵犯脏器(胆总管)“改为”Organ Invasion (Bile Duct)“，”侵犯脏器(胃)“改为”Organ Invasion (Stomach)“，”侵犯脏器(小肠)“改为”Organ Invasion (Small Intestine)“，”侵犯脏器(横结肠)“改为”Organ Invasion (Transverse Colon)“，”侵犯脏器(膈)“改为”Organ Invasion (Diaphragm)“，”侵犯脏器(脾)“改为”Organ Invasion (Spleen)“，”侵犯脏器(大网膜)“改为”Organ Invasion (Omentum)“，
# ”AJCC分期“改为”AJCC Stage“，”相关并发症(胰瘘)“改为”Complication (Pancreatic Fistula)“，”相关并发症(胆瘘)“改为”Complication (Bile Fistula)“，”相关并发症(胃肠瘘)“改为”Complication (Gastrointestinal Fistula)“，”相关并发症(腹腔出血)“改为”Complication (Intra-abdominal Bleeding)“，”相关并发症(感染)“改为”Complication (Infection)“，”其他并发症“改为”Complication (Other)“，”胰瘘分级(ISGPF)“改为”Pancreatic Fistula Grade (ISGPF)“，”再次手术“改为”Reoperation"，“再次手术原因”改为”Reoperation Reason"，“术后输血”改为”Postoperative Transfusion"，“院内死亡”改为”In-hospital Death"，“术后化疗方案”改为”Postoperative Chemotherapy Regimen"，“持续时间”改为”Duration"，“随访状态”改为”Follow-up Status"，“生存状态”改为”Survival Status"，“死亡日期”改为”Death Date"，“OS”保持不变，“复发时间”改为”Recurrence Time"，“复发部位”改为”Recurrence Site"，“转移时间”改为”Metastasis Time"，“转移部位”改为”Metastasis Site"
combined_df.rename(
    columns={
        "病人年龄": "Age",
        "病人性别": "Gender",
        "身高": "Height",
        "体重": "Weight",
        "病人姓名": "Name",
        "手术名称": "Surgical procedure",
        "手术日期": "Surgery Date",
        "ASA麻醉评分": "ASA Score",
        "基础疾病": "Comorbidities",
        "病理诊断": "Pathological Diagnosis",
        "肿瘤大小": "Tumor Size",
        "神经侵犯": "Nerve Invasion",
        "血管侵犯(肠系膜上静脉)": "Vascular Invasion (SMV)",
        "血管侵犯(门静脉)": "Vascular Invasion (PV)",
        "血管侵犯(肠系膜上动脉)": "Vascular Invasion (SMA)",
        "血管侵犯(肝动脉)": "Vascular Invasion (HA)",
        "血管侵犯(脾静脉)": "Vascular Invasion (SV)",
        "血管侵犯(脾动脉)": "Vascular Invasion (SA)",
        "血管侵犯(腹主动脉)": "Vascular Invasion (AA)",
        "血管侵犯(胃十二指肠动脉)": "Vascular Invasion (GDA)",
        "血管侵犯(胃左动脉)": "Vascular Invasion (LGA)",
        "侵犯脏器(十二指肠)": "Organ Invasion (Duodenum)",
        "侵犯脏器(肝)": "Organ Invasion (Liver)",
        "侵犯脏器(胆囊)": "Organ Invasion (Gallbladder)",
        "侵犯脏器(胆总管)": "Organ Invasion (Bile Duct)",
        "侵犯脏器(胃)": "Organ Invasion (Stomach)",
        "侵犯脏器(小肠)": "Organ Invasion (Small Intestine)",
        "侵犯脏器(横结肠)": "Organ Invasion (Transverse Colon)",
        "侵犯脏器(膈)": "Organ Invasion (Diaphragm)",
        "侵犯脏器(脾)": "Organ Invasion (Spleen)",
        "侵犯脏器(大网膜)": "Organ Invasion (Omentum)",
        "AJCC分期": "AJCC Stage",
        "相关并发症(胰瘘)": "Complication (Pancreatic Fistula)",
        "相关并发症(胆瘘)": "Complication (Bile Fistula)",
        "相关并发症(胃肠瘘)": "Complication (Gastrointestinal Fistula)",
        "相关并发症(腹腔出血)": "Complication (Intra-abdominal Bleeding)",
        "相关并发症(感染)": "Complication (Infection)",
        "其他并发症": "Complication (Other)",
        "胰瘘分级(ISGPF)": "Pancreatic Fistula Grade (ISGPF)",
        "再次手术": "Reoperation",
        "再次手术原因": "Reoperation Reason",
        "术后输血": "Postoperative Transfusion",
        "院内死亡": "In-hospital Death",
        "术后化疗方案": "Postoperative Chemotherapy Regimen",
        "持续时间": "Duration",
        "随访状态": "Follow-up Status",
        "生存状态": "Survival Status",
        "死亡日期": "Death Date",
        "复发时间": "Recurrence Time",
        "复发部位": "Recurrence Site",
        "转移时间": "Metastasis Time",
        "转移部位": "Metastasis Site",
        "总胆红素": "Total Bilirubin",
        "直接胆红素": "Direct Bilirubin",
        "总蛋白": "Total Protein",
        "ALB": "Albumin",
        "血淀粉酶": "Amylase",
        "空腹血糖": "Fasting Blood Glucose",
    },
    inplace=True,
)
# 保存修改后的文件
combined_df.to_excel("4总_75岁及以上.xlsx", index=False)
print("已将 4总_75岁及以上.xlsx 中的列名修改为英文版本。")


# combined_df中新增一列“Hypertension”and "Diabetes"，根据“Comorbidities这一列
combined_df["Hypertension"] = combined_df["Comorbidities"].apply(lambda x: 1 if "高血压" in str(x) else 0)
combined_df["Diabetes"] = combined_df["Comorbidities"].apply(lambda x: 1 if "糖尿病" in str(x) else 0)

# combined_df中新增一列”Vascular Invasion“包含所有血管侵犯的综合信息，如果这一行中任意一个血管侵犯的列不为0，/，空白或NaN，则视为“有侵犯”，否则视为“无侵犯”
vessel_cols = [
    "Vascular Invasion (SMV)",
    "Vascular Invasion (PV)",
    "Vascular Invasion (SMA)",
    "Vascular Invasion (HA)",
    "Vascular Invasion (SV)",
    "Vascular Invasion (SA)",
    "Vascular Invasion (AA)",
    "Vascular Invasion (GDA)",
    "Vascular Invasion (LGA)",
]


# 2. 定义判定函数：如果是 0, /, 空白或 NaN，则视为“无侵犯”
def check_invasion(row):
    # 定义“无侵犯”的标志值
    null_values = ["0", "/", "", None, np.nan]

    # 检查这一行中指定的列，是否包含不在 null_values 里的值
    # str(x).strip() 可以处理带空格的情况
    for col in vessel_cols:
        val = str(row[col]).strip() if pd.notna(row[col]) else None
        if val not in null_values and pd.notna(row[col]):
            return 1
    return 0


combined_df["Vascular Invasion"] = combined_df.apply(check_invasion, axis=1)
print("已根据血管侵犯的列在 4总_75岁及以上.xlsx 中新增了 Vascular Invasion 列。")


# 修改'CA125', 'CA19-9', 'CEA', 'AFP', 'CA724', 'CA242'，若开头为”>“/""<"，则去掉符号并保留数字部分
def clean_ca199(value):
    if isinstance(value, str) and value.startswith((">", "<")):
        return value[1:].strip()
    return value


# Apply the cleaning function to each of the specified columns
ca_cols = ["CA125", "CA19-9", "CEA", "AFP", "CA724", "CA242"]
for col in ca_cols:
    combined_df[col] = combined_df[col].apply(clean_ca199)
print("已清洗 CA125, CA19-9, CEA, AFP, CA724, CA242 列中的值，去掉开头的 > 或 < 符号。")

# 保存修改后的文件
combined_df.to_excel("4总_75岁及以上.xlsx", index=False)
print("已根据 Comorbidities 列在 4总_75岁及以上.xlsx 中新增了 Hypertension 和 Diabetes 两列。")

# 只保留“Pathological Diagnosis”中“DA”这一项
combined_df = combined_df[combined_df["Pathological Diagnosis"].str.contains("DA", na=False)]
print("已筛选出 Pathological Diagnosis 列中包含 'DA' 的记录。")
# 查看combined_df的行数
print(f"筛选后剩余记录数: {len(combined_df)}")
# 查看Surgical approach中的分布
print("Surgical approach 分布:")
print(combined_df["Surgical approach"].value_counts())
# # 删去“OS”小于等于30的行
# combined_df = combined_df[
#     combined_df["OS"].apply(lambda x: pd.to_numeric(x, errors="coerce") > 30 if pd.notna(x) else False)
# ]
# print("已删除 OS 小于等于 30 的记录。")
# # 查看“In-hospital Death”这列的分布
# print("In-hospital Death 分布:")
# print(combined_df["In-hospital Death"].value_counts())
# # 删去“In-hospital Death”这列中值为1的行及“自动出院”
# combined_df = combined_df[combined_df["In-hospital Death"] != 1]
# combined_df = combined_df[combined_df["In-hospital Death"] != "自动出院"]
# print("已删除 In-hospital Death 列中值为 1 和 '自动出院' 的记录。")
# 画出“OS”这列的直方图（先转为数值，避免字符串导致报错）
import matplotlib.pyplot as plt

os_numeric = pd.to_numeric(combined_df["OS"], errors="coerce")
os_valid = os_numeric.dropna()

if os_valid.empty:
    print("OS 列无可用于绘图的数值。")
else:
    plt.figure()  # 确保新建图像
    os_valid.hist(bins=30)
    plt.xlabel("OS")
    plt.ylabel("Frequency")
    plt.title("Distribution of OS")
    
    # 保存到当前路径
    save_path = os.path.join(os.getcwd(), "os_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  # 关闭图像，避免占用内存
    print(f"图片已保存到: {save_path}")


# 保存修改后的文件
combined_df.to_excel("4总_75岁及以上.xlsx", index=False)
print("已将筛选后的数据保存到 4总_75岁及以上.xlsx 文件中。")

# 将4总_75岁及以上.xlsx中"Surgical approach"这列添加到4总_75岁及以上_手动补全.xlsx中，匹配的依据是“住院号”这一列
df_combined = pd.read_excel("4总_75岁及以上.xlsx")
df_manual = pd.read_excel("4总_75岁及以上_手动补全.xlsx")
# 创建一个以“住院号”为索引的 lookup 表
lookup = df_combined.set_index("住院号")["Surgical procedure"].to_dict()
# 将“Surgical procedure”列添加到 df_manual 中
df_manual["Surgical procedure"] = df_manual["住院号"].map(lookup)
# 保存修改后的文件
df_manual.to_excel("4总_75岁及以上_手动补全.xlsx", index=False)
print("已将 Surgical procedure 列添加到 4总_75岁及以上_手动补全.xlsx 文件中，并根据 住院号 进行匹配。")