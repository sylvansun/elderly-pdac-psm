import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest


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

    cox_vars = ["treat"] + cfg.continuous_vars() + [chemo_var]

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


def run_km_analysis(
    matched_df, dataset_name, output_dir, time_col="OS", event_col="Survival Status"
):

    g0 = matched_df[matched_df["treat"] == 0]
    g1 = matched_df[matched_df["treat"] == 1]

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(6, 6))

    for g, label in zip([g0, g1], ["OPEN", "MIS"]):

        kmf.fit(durations=g[time_col], event_observed=g[event_col], label=label)

        kmf.plot_survival_function(ci_show=True)

    # log-rank
    res = logrank_test(
        g0[time_col],
        g1[time_col],
        event_observed_A=g0[event_col],
        event_observed_B=g1[event_col],
    )

    p_value = res.p_value

    plt.title(f"{dataset_name} Kaplan-Meier Curve")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")

    plt.text(0.6, 0.1, f"log-rank p = {p_value:.4f}", transform=plt.gca().transAxes)

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_KM_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"KM analysis finished: p={p_value:.4f}")

    return p_value


def run_rsf_analysis(
    matched_df, dataset_name, output_dir, time_col="OS", event_col="Survival Status"
):

    ml_X = matched_df[cfg.continuous_vars()]

    y_ml = Surv.from_dataframe(event_col, time_col, matched_df)

    rsf = RandomSurvivalForest(
        **cfg.RSF_PARAMS,
        random_state=cfg.RANDOM_SEED,
        n_jobs=-1,
    )

    rsf.fit(ml_X, y_ml)

    print("RSF fitted.")

    return rsf, ml_X, y_ml


def run_two_year_os_analysis(
    matched_df,
    dataset_name,
    output_dir,
    time_col="OS",
    event_col="Survival Status",
    landmark_days=730,  # 2 years
):

    df = matched_df.copy()
    df = df[[time_col, event_col, "treat"]].copy()
    df = df.dropna(subset=[time_col, event_col, "treat"])
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna(subset=[time_col, event_col])

    df["OS_2yr"] = np.where(
        (
            (df[time_col] >= landmark_days)
            | ((df[time_col] < landmark_days) & (df[event_col] == 0))
        ),
        1,  # alive
        0,  # dead
    )

    table = pd.crosstab(df["treat"], df["OS_2yr"])

    # ensure both columns exist
    for col in [0, 1]:
        if col not in table.columns:
            table[col] = 0

    # reorder columns
    table = table[[0, 1]]

    # rename
    table.index = ["OPEN", "MIS"]
    table.columns = ["Death<2yr", "Alive≥2yr"]

    print("2-year OS contingency table:")
    print(table)

    open_survival = table.loc["OPEN", "Alive≥2yr"] / table.loc["OPEN"].sum()

    mis_survival = table.loc["MIS", "Alive≥2yr"] / table.loc["MIS"].sum()

    chi2, p_chi, dof, expected = chi2_contingency(table)

    if (expected < 5).any():
        use_fisher = True
    else:
        use_fisher = False

    if use_fisher:

        _, p = fisher_exact(table.values)

        test_used = "Fisher exact test"

    else:

        p = p_chi

        test_used = "Chi-square test"

    table_for_or = table.values.astype(float)

    if (table_for_or == 0).any():
        table_for_or += 0.5

    ct = Table2x2(table_for_or)

    or_val = ct.oddsratio

    ci_low, ci_high = ct.oddsratio_confint()

    risk_diff = mis_survival - open_survival

    summary_df = pd.DataFrame(
        {
            "Group": ["OPEN", "MIS"],
            "2yr_survival_rate": [
                open_survival,
                mis_survival,
            ],
            "2yr_survival_percent": [
                round(open_survival * 100, 2),
                round(mis_survival * 100, 2),
            ],
        }
    )

    stats_df = pd.DataFrame(
        {
            "Metric": [
                "Statistical test",
                "P value",
                "Odds Ratio",
                "OR 95% CI low",
                "OR 95% CI high",
                "Absolute Risk Difference (MIS-OPEN)",
            ],
            "Value": [
                test_used,
                p,
                or_val,
                ci_low,
                ci_high,
                risk_diff,
            ],
        }
    )

    os.makedirs(output_dir, exist_ok=True)

    summary_df.to_excel(
        os.path.join(output_dir, f"{dataset_name}_2yr_OS_rates.xlsx"),
        index=False,
    )

    stats_df.to_excel(
        os.path.join(output_dir, f"{dataset_name}_2yr_OS_stats.xlsx"),
        index=False,
    )

    table.to_excel(os.path.join(output_dir, f"{dataset_name}_2yr_OS_table.xlsx"))

    return summary_df, stats_df, table
