import pandas as pd
import numpy as np
import utils

import pandas as pd

files_to_process = ["5总_75岁及以上.xlsx", "5总_并发症.xlsx"]

for file in files_to_process:
    # 读取文件
    df = pd.read_excel(file)

    # Hb 转数值 + 贫血判断
    df["Hb"] = pd.to_numeric(df["Hb"], errors="coerce")
    df["Anaemia"] = df.apply(utils.is_anaemia, axis=1)

    # 术后化疗方案 one-hot（二分类）
    df["Postoperative Chemotherapy Regimen"] = (
        df["Postoperative Chemotherapy Regimen"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    # 保存（建议避免覆盖原文件）
    output_file = file.replace(".xlsx", "_processed.xlsx")
    df.to_excel(output_file, index=False)

    print(f"{file} 处理完成 -> {output_file}")