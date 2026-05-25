import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shap

from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc

from utils import smd

CONTINUOUS_VARS = ["Age", "BMI", "Albumin"]
CATEGORICAL_VARS = ["AJCC Stage", "ASA Score", "pTNM_T", "Anaemia", "Surgical procedure"]


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
# 2. PS模型
# =========================
def fit_ps_model(df_model):

    X = df_model[CONTINUOUS_VARS + CATEGORICAL_VARS]
    X = pd.get_dummies(X, columns=CATEGORICAL_VARS, drop_first=True)

    y = df_model["treat"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logit = LogisticRegression(max_iter=2000, random_state=42)
    logit.fit(X_scaled, y)

    ps = logit.predict_proba(X_scaled)[:, 1].clip(0.0001, 0.9999)

    df_model["ps"] = ps
    df_model["logit_ps"] = np.log(ps / (1 - ps))

    return df_model, X


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
        bbox_inches="tight"
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

    matched_df = pd.concat([
        df_model.loc[treated_idx],
        df_model.loc[control_idx]
    ])

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

        rows.append([
            var,
            f"{t.mean():.2f}±{t.std():.2f}",
            f"{c.mean():.2f}±{c.std():.2f}",
            p
        ])

    return pd.DataFrame(rows, columns=["Variable", "MIS", "OPEN", "p"])


# =========================
# 6. SMD Love plot（拆开写更清晰）
# =========================
def plot_smd(X_before, X_after, matched_df, dataset_name, output_dir):

    X_full = pd.concat([X_before, matched_df["treat"]], axis=1)

    continuous_smd_before = [smd(X_full, c) for c in CONTINUOUS_VARS]
    continuous_smd_after = [smd(X_full.loc[matched_df.index], c) for c in CONTINUOUS_VARS]

    plt.figure()
    plt.scatter(continuous_smd_before, CONTINUOUS_VARS, label="Before")
    plt.scatter(continuous_smd_after, CONTINUOUS_VARS, label="After")
    plt.axvline(0.1, linestyle="--")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f"{dataset_name} - Love plot (Continuous)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_love_plot_continuous.png"),
        dpi=300,
        bbox_inches="tight"
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
    df_model, X = fit_ps_model(df_model)

    # plot ps
    plot_ps_density(df_model, dataset_name, output_dir)

    # matching
    matched_df = ps_matching(df_model)

    matched_df.to_excel(
        os.path.join(output_dir, f"{dataset_name}_PSM_dataset.xlsx"),
        index=False
    )

    # table1
    table1 = compute_table1(matched_df)

    table1.to_excel(
        os.path.join(output_dir, f"{dataset_name}_Table1.xlsx"),
        index=False
    )

    return df_model, matched_df, X


# =========================
# 8. batch入口
# =========================
def run_psm_batch(file_list, output_dir="PSM_results"):

    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_list:
        run_single_dataset(file_path, output_dir)

    print("\nAll datasets processed successfully.")

if __name__ == "__main__":

    set_seed(42)

    files = [
        "complications.xlsx",
        "elderly.xlsx"
    ]

    run_psm_batch(files)