import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shap

from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc

from config import CATEGORICAL_VARS, CONTINUOUS_VARS
from analysis import run_cox_analysis
from utils import smd_categorical, smd_continuous, smd


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
# 3. PS分布图
# =========================
def plot_ps_density(df_model, dataset_name, output_dir):

    plt.figure()
    sns.kdeplot(df_model[df_model["treat"] == 1]["ps"], label="MIS")
    sns.kdeplot(df_model[df_model["treat"] == 0]["ps"], label="OPEN")

    plt.title(f"{dataset_name} - PS density")
    plt.legend()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_ps_density.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


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
# 6. SMD Love plot
# =========================
def plot_smd(X_before, matched_df, dataset_name, output_dir):

    X_full = pd.concat([X_before, matched_df["treat"]], axis=1)

    # before matching
    continuous_smd_before = [smd(X_full, c) for c in CONTINUOUS_VARS]

    # after matching
    continuous_smd_after = [
        smd(X_full.loc[matched_df.index], c) for c in CONTINUOUS_VARS
    ]

    plt.figure(figsize=(6, 4))

    plt.scatter(continuous_smd_before, CONTINUOUS_VARS, label="Before Matching")

    plt.scatter(continuous_smd_after, CONTINUOUS_VARS, label="After Matching")

    plt.axvline(0.1, linestyle="--")

    plt.xlabel("Standardized Mean Difference")

    plt.title(f"{dataset_name} Love Plot")

    plt.gca().invert_yaxis()

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_love_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_love_continuous(df_before, df_after, dataset_name, output_dir):

    before_smd = []
    after_smd = []

    for var in CONTINUOUS_VARS:

        before_smd.append(smd_continuous(df_before, var))

        after_smd.append(smd_continuous(df_after, var))

    plt.figure(figsize=(6, 4))

    plt.scatter(before_smd, CONTINUOUS_VARS, label="Before", marker="o")
    plt.scatter(after_smd, CONTINUOUS_VARS, label="After", marker="o")

    plt.axvline(0.1, linestyle="--")

    plt.xlabel("Standardized Mean Difference")
    plt.title(f"{dataset_name} Love Plot - Continuous")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_love_continuous.png"), dpi=300
    )

    plt.close()


def plot_love_categorical(df_before, df_after, dataset_name, output_dir):

    before_smd = []
    after_smd = []

    for var in CATEGORICAL_VARS:

        before_smd.append(smd_categorical(df_before, var))

        after_smd.append(smd_categorical(df_after, var))

    plt.figure(figsize=(6, 4))

    plt.scatter(before_smd, CATEGORICAL_VARS, label="Before", marker="s")
    plt.scatter(after_smd, CATEGORICAL_VARS, label="After", marker="s")

    plt.axvline(0.1, linestyle="--")

    plt.xlabel("Standardized Mean Difference")
    plt.title(f"{dataset_name} Love Plot - Categorical")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_love_categorical.png"), dpi=300
    )

    plt.close()


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
    plot_ps_density(df_model, dataset_name, output_dir)

    # matching
    matched_df = ps_matching(df_model)

    # love plot
    df_before = df_model.copy()
    df_after = matched_df.copy()

    plot_love_continuous(df_before, df_after, dataset_name, output_dir)
    plot_love_categorical(df_before, df_after, dataset_name, output_dir)

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
    plot_cox_forest(result_table, dataset_name, output_dir)

    # RSF
    rsf, ml_X, y_ml = run_rsf_analysis(matched_df, dataset_name, output_dir)

    # Time ROC
    plot_time_auc(rsf, ml_X, y_ml, dataset_name, output_dir)

    # Importance
    plot_feature_importance(rsf, ml_X, y_ml, dataset_name, output_dir)

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
# 10. Forest plot
# =========================
def plot_cox_forest(result_table, dataset_name, output_dir):

    plt.figure(figsize=(6, 4))

    y_pos = np.arange(len(result_table))

    hr = result_table["HR"]
    lower = result_table["HR_lower"]
    upper = result_table["HR_upper"]

    plt.errorbar(hr, y_pos, xerr=[hr - lower, upper - hr], fmt="o")

    plt.yticks(y_pos, result_table.index)

    plt.axvline(1, linestyle="--")

    plt.xlabel("Hazard Ratio")
    plt.title(f"{dataset_name} Cox Forest Plot")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_Cox_forest.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


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


def plot_time_auc(rsf, ml_X, y_ml, dataset_name, output_dir):

    # =========================
    # 1. 原始候选时间点
    # =========================
    raw_times = np.arange(12, 60, 12, dtype=float)

    min_followup = np.min(y_ml["OS"])
    max_followup = np.max(y_ml["OS"])

    # =========================
    # 2. 过滤
    # =========================
    times = raw_times[(raw_times >= min_followup) & (raw_times < max_followup)]

    # =========================
    # 3. 🔥 fallback（关键）
    # =========================
    if len(times) == 0:
        print("⚠️ Warning: default times invalid, using quantiles instead")

        times = np.quantile(y_ml["OS"], [0.25, 0.5, 0.75])

        times = np.unique(times)

        # 如果还是不行，再兜底
        if len(times) == 0:
            times = np.array([max_followup * 0.3, max_followup * 0.6])

    # =========================
    # 4. prediction
    # =========================
    risk_scores = rsf.predict(ml_X)

    auc, mean_auc = cumulative_dynamic_auc(y_ml, y_ml, risk_scores, times)

    # =========================
    # 5. plot
    # =========================
    plt.figure()

    plt.plot(times, auc, marker="o")

    plt.xlabel("Time")
    plt.ylabel("AUC")

    plt.title(f"{dataset_name} Time-dependent ROC\nMean AUC={mean_auc:.3f}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_TimeROC.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Mean AUC = {mean_auc:.3f}")

    return auc, mean_auc


# =========================
# 13. Feature importance
# =========================
def plot_feature_importance(rsf, ml_X, y_ml, dataset_name, output_dir):

    perm = permutation_importance(
        rsf, ml_X, y_ml, n_repeats=20, random_state=42, n_jobs=-1
    )

    importance_df = pd.DataFrame(
        {"Feature": ml_X.columns, "Importance": perm.importances_mean}
    )

    importance_df = importance_df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 5))

    plt.barh(importance_df["Feature"], importance_df["Importance"])

    plt.gca().invert_yaxis()

    plt.xlabel("Permutation Importance")

    plt.title(f"{dataset_name} RSF Importance")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_RSF_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    importance_df.to_excel(
        os.path.join(output_dir, f"{dataset_name}_RSF_importance.xlsx"), index=False
    )

    return importance_df


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

    # ==========================================
    # 9. risk difference
    # ==========================================
    risk_diff = mis_survival - open_survival

    # ==========================================
    # 10. results dataframe
    # ==========================================
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

    # ==========================================
    # 11. save outputs
    # ==========================================
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
