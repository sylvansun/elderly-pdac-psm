import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import plots

from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest

from config import CATEGORICAL_VARS, CONTINUOUS_VARS
from analysis import run_cox_analysis


# =========================
# 0. seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# =========================
# 1. 数据预处理
# =========================
def preprocess_data(df):
    df_model = df.copy().sort_index()

    for col in CONTINUOUS_VARS:
        df_model[col] = df_model[col].fillna(df_model[col].median())

    for col in CATEGORICAL_VARS:
        df_model[col] = df_model[col].fillna("Unknown")

    df_model["treat"] = df_model["Surgical approach"].map({"OPEN": 0, "MIS": 1})

    return df_model


# =========================
# 2. PS模型（升级版）
# =========================
def fit_ps_model(df_model):

    # =========================
    # 1. 构造设计矩阵
    # =========================
    X_raw = df_model[CONTINUOUS_VARS + CATEGORICAL_VARS].copy()

    X_encoded = pd.get_dummies(X_raw, columns=CATEGORICAL_VARS, drop_first=True)

    y = df_model["treat"]

    # =========================
    # 2. 标准化（用于logistic PS model）
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # =========================
    # 3. PS model
    # =========================
    logit = LogisticRegression(max_iter=2000, random_state=42)

    logit.fit(X_scaled, y)

    # =========================
    # 4. propensity score
    # =========================
    ps = logit.predict_proba(X_scaled)[:, 1]

    # 防止log(0)
    ps = np.clip(ps, 1e-5, 1 - 1e-5)

    df_model["ps"] = ps
    df_model["logit_ps"] = np.log(ps / (1 - ps))

    # =========================
    # 5. 返回内容（关键改动）
    # =========================
    return df_model, X_encoded, scaler, logit


# =========================
# 4. PSM matching（核心）
# =========================
def ps_matching(df_model, caliper_ratio=0.2):

    treated = df_model[df_model["treat"] == 1]
    control = df_model[df_model["treat"] == 0]

    caliper = caliper_ratio * df_model["logit_ps"].std()

    matched_pairs = []
    used_control = set()

    for t_idx, t_row in treated.iterrows():

        dist = abs(control["logit_ps"] - t_row["logit_ps"])
        dist = dist[~dist.index.isin(used_control)]

        if dist.empty:
            continue

        nearest = dist.idxmin()

        if dist.min() <= caliper:
            matched_pairs.append((t_idx, nearest))
            used_control.add(nearest)

    treated_idx = [x[0] for x in matched_pairs]
    control_idx = [x[1] for x in matched_pairs]

    matched_df = pd.concat([df_model.loc[treated_idx], df_model.loc[control_idx]])

    return matched_df


# =========================
# 5. Table 1（基线比较）
# =========================
def compute_table1(matched_df):

    rows = []

    for var in CONTINUOUS_VARS:

        t = matched_df[matched_df["treat"] == 1][var]
        c = matched_df[matched_df["treat"] == 0][var]

        p = stats.ttest_ind(t, c).pvalue

        rows.append(
            [var, f"{t.mean():.2f}±{t.std():.2f}", f"{c.mean():.2f}±{c.std():.2f}", p]
        )

    return pd.DataFrame(rows, columns=["Variable", "MIS", "OPEN", "p"])


# =========================
# 7. 单个dataset pipeline
# =========================
def run_single_dataset(file_path, output_dir):

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nProcessing: {dataset_name}")

    df = pd.read_excel(file_path)

    # preprocess
    df_model = preprocess_data(df)

    # ps model
    df_model, X_encoded, scaler, logit = fit_ps_model(df_model)

    # plot ps
    plots.plot_ps_density(df_model, dataset_name, output_dir)

    # matching
    matched_df = ps_matching(df_model)

    # love plot
    df_before = df_model.copy()
    df_after = matched_df.copy()

    plots.plot_love_continuous(df_before, df_after, dataset_name, output_dir)
    plots.plot_love_categorical(df_before, df_after, dataset_name, output_dir)

    # table1
    table1 = compute_table1(matched_df)

    table1.to_excel(
        os.path.join(output_dir, f"{dataset_name}_Table1.xlsx"), index=False
    )

    # KM
    run_km_analysis(matched_df, dataset_name, output_dir)

    # Cox
    cph, result_table = run_cox_analysis(matched_df, dataset_name, output_dir)

    # forest plot
    plots.plot_cox_forest(result_table, dataset_name, output_dir)

    # RSF
    rsf, ml_X, y_ml = run_rsf_analysis(matched_df, dataset_name, output_dir)

    # Time ROC
    plots.plot_time_auc(rsf, ml_X, y_ml, dataset_name, output_dir)

    # Importance
    plots.plot_feature_importance(rsf, ml_X, y_ml, dataset_name, output_dir)

    run_two_year_os_analysis(matched_df, dataset_name, output_dir)

    matched_df.to_excel(
        os.path.join(output_dir, f"{dataset_name}_PSM_dataset.xlsx"), index=False
    )

    return df_model, matched_df


# =========================
# 8. Kaplan-Meier analysis
# =========================
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


# =========================
# 11. Random Survival Forest
# =========================
def run_rsf_analysis(
    matched_df, dataset_name, output_dir, time_col="OS", event_col="Survival Status"
):

    ml_X = matched_df[CONTINUOUS_VARS]

    y_ml = Surv.from_dataframe(event_col, time_col, matched_df)

    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1,
    )

    rsf.fit(ml_X, y_ml)

    print("RSF fitted.")

    return rsf, ml_X, y_ml


# ==========================================
# 2-Year Overall Survival Analysis
# OS time unit = DAYS
# ==========================================
def run_two_year_os_analysis(
    matched_df,
    dataset_name,
    output_dir,
    time_col="OS",
    event_col="Survival Status",
    landmark_days=730,  # 2 years
):

    # ==========================================
    # 0. copy dataframe
    # ==========================================
    df = matched_df.copy()

    # ==========================================
    # 1. basic cleaning
    # ==========================================
    df = df[[time_col, event_col, "treat"]].copy()

    # remove missing
    df = df.dropna(subset=[time_col, event_col, "treat"])

    # force numeric
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    # drop invalid
    df = df.dropna(subset=[time_col, event_col])

    # ==========================================
    # event definition
    #
    # ASSUMED:
    # 1 = death
    # 0 = censored/alive
    # ==========================================

    # ==========================================
    # 3. define 2-year OS status
    # ==========================================
    #
    # Alive at 2yr:
    #   - survived >=730 days
    #   - censored before 730 days
    #
    # Dead before 2yr:
    #   - died before 730 days
    #
    # ==========================================

    df["OS_2yr"] = np.where(
        (
            (df[time_col] >= landmark_days)
            | ((df[time_col] < landmark_days) & (df[event_col] == 0))
        ),
        1,  # alive
        0,  # dead
    )

    # ==========================================
    # 4. contingency table
    # ==========================================
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

    # ==========================================
    # 5. survival rates
    # ==========================================
    open_survival = table.loc["OPEN", "Alive≥2yr"] / table.loc["OPEN"].sum()

    mis_survival = table.loc["MIS", "Alive≥2yr"] / table.loc["MIS"].sum()

    # ==========================================
    # 6. choose statistical test
    # ==========================================
    #
    # Fisher:
    #   if any expected cell <5
    #
    # Chi-square:
    #   otherwise
    #
    # ==========================================

    chi2, p_chi, dof, expected = chi2_contingency(table)

    if (expected < 5).any():
        use_fisher = True
    else:
        use_fisher = False

    # ==========================================
    # 7. perform test
    # ==========================================
    if use_fisher:

        _, p = fisher_exact(table.values)

        test_used = "Fisher exact test"

    else:

        p = p_chi

        test_used = "Chi-square test"

    # ==========================================
    # 8. odds ratio
    # ==========================================
    #
    # add 0.5 continuity correction
    # if zero cell exists
    #
    # ==========================================

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


# =========================
# batch入口
# =========================
def run_psm_batch(file_list, output_dir="PSM_results"):

    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_list:
        run_single_dataset(file_path, output_dir)

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":

    set_seed(42)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    files = [
        os.path.join(DATA_DIR, "complications.xlsx"),
        os.path.join(DATA_DIR, "elderly.xlsx"),
    ]

    run_psm_batch(files)
