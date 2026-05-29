import os
import numpy as np
import pandas as pd

from config import CONTINUOUS_VARS

from lifelines import CoxPHFitter


# ==========================================
# Multivariable Cox Regression
# ==========================================
def run_cox_analysis(
    matched_df,
    dataset_name,
    output_dir,
    time_col="OS",
    event_col="Survival Status",
):

    # ==========================================
    # 1. define variables
    # ==========================================

    chemo_var = "Postoperative Chemotherapy Regimen"

    cox_vars = ["treat"] + CONTINUOUS_VARS + [chemo_var]

    # ==========================================
    # 2. build dataframe
    # ==========================================

    cox_df = matched_df[[time_col, event_col] + cox_vars].copy()

    # ==========================================
    # 3. remove missing values
    # ==========================================

    cox_df = cox_df.dropna()

    # ==========================================
    # 4. force numeric for survival columns
    # ==========================================

    cox_df[time_col] = pd.to_numeric(cox_df[time_col], errors="coerce")

    cox_df[event_col] = pd.to_numeric(cox_df[event_col], errors="coerce")

    cox_df = cox_df.dropna()

    # ==========================================
    # 5. categorical variables
    # ==========================================

    categorical_vars = [chemo_var]

    cox_df = pd.get_dummies(cox_df, columns=categorical_vars, drop_first=True)

    # ==========================================
    # 6. remove constant columns
    # ==========================================

    constant_cols = []

    for col in cox_df.columns:

        if col in [time_col, event_col]:
            continue

        if cox_df[col].nunique() <= 1:
            constant_cols.append(col)

    if len(constant_cols) > 0:

        cox_df = cox_df.drop(columns=constant_cols)

    # ==========================================
    # 7. fit Cox model
    # ==========================================

    cph = CoxPHFitter()

    cph.fit(
        cox_df,
        duration_col=time_col,
        event_col=event_col,
    )

    # ==========================================
    # 8. summary table
    # ==========================================

    summary = cph.summary.copy()

    summary["HR"] = np.exp(summary["coef"])

    summary["HR_lower"] = np.exp(summary["coef lower 95%"])

    summary["HR_upper"] = np.exp(summary["coef upper 95%"])

    result_table = summary[
        [
            "HR",
            "HR_lower",
            "HR_upper",
            "p",
        ]
    ].copy()

    result_table = result_table.round(4)

    # ==========================================
    # 9. save outputs
    # ==========================================

    os.makedirs(output_dir, exist_ok=True)

    result_table.to_excel(os.path.join(output_dir, f"{dataset_name}_Cox_results.xlsx"))

    summary.to_excel(os.path.join(output_dir, f"{dataset_name}_Cox_full_summary.xlsx"))

    # ==========================================
    # 10. PH assumption check
    # ==========================================

    ph_test_result = cph.check_assumptions(
        cox_df, p_value_threshold=0.05, show_plots=False
    )

    # ==========================================
    # 11. save model dataframe
    # ==========================================

    cox_df.to_excel(
        os.path.join(output_dir, f"{dataset_name}_Cox_input_dataframe.xlsx"),
        index=False,
    )

    return cph, result_table
