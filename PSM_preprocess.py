import pandas as pd
import numpy as np
import logging
import os
from utils import determine_pT

# ==========================================================
# 路径配置
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "subtask")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# 主日志
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

main_logger = logging.getLogger("main")


def convert_to_numeric_with_fill(series, col_name):
    """
    将Series转换为数值型，无法转换的填充为 "/"
    """
    try:
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_series = numeric_series.fillna("/")
        return numeric_series
    except Exception as e:
        main_logger.exception(f"转换 {col_name} 时出错")
        return series


def process_file(file_path):

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # ======================================================
    # 文件专属日志
    # ======================================================

    logger = logging.getLogger(base_name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    log_file = os.path.join(LOG_DIR, f"{base_name}.log")

    file_handler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8",
    )

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # ======================================================
    # 读取数据
    # ======================================================

    logger.info(f"开始处理文件: {file_path}")

    df = pd.read_excel(file_path)

    # 去除列名空格
    df.columns = df.columns.str.strip()

    # 需要检查的列
    required_cols = [
        "Age",
        "BMI",
        "Tumor Size",
        "AJCC Stage",
        "Albumin",
    ]

    # 去除关键列缺失值
    df = df.dropna(subset=required_cols)

    # 去除包含 "/" 的行
    for col in required_cols:
        df = df[~df[col].astype(str).str.contains("/")]

    # 替换异常值
    df["Tumor Size"] = df["Tumor Size"].replace("7、4", "7")

    # 转换数值
    df["Tumor Size"] = pd.to_numeric(
        df["Tumor Size"],
        errors="coerce",
    )

    df["pTNM_T"] = df.apply(determine_pT, axis=1)

    # ======================================================
    # Surgical procedure
    # ======================================================

    df["Surgical procedure"] = (
        df["Surgical procedure"]
        .astype(str)
        .str.strip()
        .str.replace("\xa0", "", regex=False)
    )

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

    # ======================================================
    # ASA Score
    # ======================================================

    df["ASA Score"] = (
        df["ASA Score"].astype(str).str.strip().str.replace("\xa0", "", regex=False)
    )

    # ======================================================
    # 日志统计
    # ======================================================

    logger.info("\n===== 数据概览 =====")

    logger.info(f"总样本数: {len(df)}")

    logger.info("\n列名:")
    logger.info(df.columns.tolist())

    logger.info("\nSurgical approach 分布:")
    logger.info("\n%s", df["Surgical approach"].value_counts(dropna=False))

    logger.info("\nSurgical procedure 分布:")
    logger.info("\n%s", df["Surgical procedure"].value_counts(dropna=False))

    logger.info("\nASA Score 分布:")
    logger.info("\n%s", df["ASA Score"].value_counts(dropna=False))

    logger.info("\n转换前数据类型:")
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

    # ======================================================
    # 数值转换
    # ======================================================

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
            df[col] = convert_to_numeric_with_fill(
                df[col],
                col,
            )

    logger.info("\n转换后数据类型:")
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

    logger.info("处理完成")

    return df


if __name__ == "__main__":

    files_to_process = [
        "5总_75岁及以上_processed.xlsx",
        "5总_并发症_processed.xlsx",
    ]

    output_names = {
        "5总_75岁及以上_processed.xlsx": "elderly.xlsx",
        "5总_并发症_processed.xlsx": "complications.xlsx",
    }

    processed_data = {}

    for file_name in files_to_process:

        input_path = os.path.join(
            INPUT_DIR,
            file_name,
        )

        try:

            main_logger.info(f"开始处理: {file_name}")

            df = process_file(input_path)

            processed_data[file_name] = df

            output_path = os.path.join(
                OUTPUT_DIR,
                output_names[file_name],
            )

            df.to_excel(
                output_path,
                index=False,
            )

            main_logger.info(f"{file_name} 处理完成，保存至: {output_path}")

        except Exception:

            main_logger.exception(f"处理 {file_name} 时发生错误")

    main_logger.info("全部文件处理结束")
