import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import smd
import config
from utils import smd_categorical, smd_continuous, smd

from sklearn.inspection import permutation_importance
from sksurv.metrics import cumulative_dynamic_auc

CATEGORICAL_VARS = config.categorical_vars()
CONTINUOUS_VARS = config.continuous_vars()


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


# smd staled version
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
