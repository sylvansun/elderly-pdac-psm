import pandas as pd
import numpy as np


def convert_to_numeric_with_fill(series, col_name):
    """
    将Series转换为数值型，无法转换的填充为 "/"
    """
    try:
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_series = numeric_series.fillna("/")
        return numeric_series
    except Exception as e:
        print(f"转换 {col_name} 时出错: {e}")
        return series


import pandas as pd
import logging
import os


def process_file(file_name):
    # ========= 创建日志文件 =========
    log_file = os.path.splitext(file_name)[0] + ".log"

    # 创建 logger
    logger = logging.getLogger(file_name)

    # 防止重复添加 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # mode="w" 表示每次覆盖旧日志
    file_handler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # ========= 读取文件 =========
    df = pd.read_excel(file_name)

    # 去除列名空格
    df.columns = df.columns.str.strip()

    # 需要检查的列
    required_cols = ["Age", "BMI", "Tumor Size", "AJCC Stage", "Albumin"]

    # 去除关键列缺失值
    df = df.dropna(subset=required_cols)

    # 去除包含 "/" 的行
    for col in required_cols:
        df = df[~df[col].astype(str).str.contains("/")]

    # 特殊值替换
    df["Tumor Size"] = df["Tumor Size"].replace("7、4", "7")

    # 清理 Surgical procedure 列
    df["Surgical procedure"] = (
        df["Surgical procedure"]
        .astype(str)
        .str.strip()
        .str.replace("\xa0", "", regex=False)
    )

    # Surgical procedure 归类
    df["Surgical procedure"] = df["Surgical procedure"].replace(
        {
            "Child": "PD",
            "全胰切除": "TP",
            "全胰切除术": "TP",
            "Appleby": "DP",
            "RAMPS": "DP",
            "En": "DP",
            "MP": "DP",
        }
    )

    # 清理 ASA Score
    df["ASA Score"] = (
        df["ASA Score"]
        .astype(str)
        .str.strip()
        .str.replace("\xa0", "", regex=False)
    )

    # ========= 写入日志 =========

    logger.info(f"\n===== {file_name} =====")

    logger.info("\n总样本数：")
    logger.info(len(df))

    logger.info("\n列名：")
    logger.info(df.columns.tolist())

    logger.info("\nSurgical approach 分布：")
    logger.info("\n%s", df["Surgical approach"].value_counts(dropna=False))

    logger.info("\nSurgical procedure 分布：")
    logger.info("\n%s", df["Surgical procedure"].value_counts(dropna=False))

    logger.info("\nASA Score 分布：")
    logger.info("\n%s", df["ASA Score"].value_counts(dropna=False))

    logger.info("\n转换前数据类型：")
    logger.info(
        "\n%s",
        df[
            [
                "Age",
                "BMI",
                "Tumor Size",
                "AJCC Stage",
                "Albumin",
                "ASA Score",
            ]
        ].dtypes,
    )

    # 转换为数值型
    numeric_cols = [
        "Age",
        "BMI",
        "Tumor Size",
        "AJCC Stage",
        "Albumin",
        "OS",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = convert_to_numeric_with_fill(df[col], col)

    logger.info("\n转换后数据类型：")
    logger.info(
        "\n%s",
        df[
            [
                "Age",
                "BMI",
                "Tumor Size",
                "AJCC Stage",
                "Albumin",
                "Survival Status",
                "OS",
            ]
        ].dtypes,
    )

    return df


if __name__ == "__main__":

    # 需要处理的文件列表
    files_to_process = [
        "5总_75岁及以上_processed.xlsx",
        "5总_并发症_processed.xlsx",
    ]

    # 用于存储处理后的DataFrame
    processed_data = {}

    # 批量处理
    for file_name in files_to_process:
        try:
            df = process_file(file_name)

            # 保存到字典
            processed_data[file_name] = df

            print(f"\n{file_name} 处理完成")

        except Exception as e:
            print(f"\n处理 {file_name} 时发生错误: {e}")

    print("\n全部文件处理结束")