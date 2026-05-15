import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

import shap

def run_psm_batch(file_list, output_dir="PSM_results"):

    # =========================
    # 创建输出文件夹
    # =========================
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # SMD函数
    # =========================
    def smd(df, col):
        t = df[df["treat"] == 1][col]
        c = df[df["treat"] == 0][col]
        pooled = np.sqrt((t.var() + c.var()) / 2)
        return abs((t.mean() - c.mean()) / pooled)

    # =========================
    # 批处理每个Excel
    # =========================
    for file_path in file_list:

        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nProcessing: {dataset_name}")

        df = pd.read_excel(file_path)

        # =========================
        # 1 数据准备
        # =========================
        df_model = df.copy().sort_index()

        continuous_vars = ["Age", "BMI", "Tumor Size", "Albumin"]
        categorical_vars = ["AJCC Stage", "ASA Score"]

        for col in continuous_vars:
            df_model[col] = df_model[col].fillna(df_model[col].median())

        df_model["AJCC Stage"] = df_model["AJCC Stage"].fillna("Unknown")
        df_model["ASA Score"] = df_model["ASA Score"].fillna("Unknown")

        df_model["treat"] = df_model["Surgical approach"].map({"OPEN": 0, "MIS": 1})

        # =========================
        # 2 PS model
        # =========================
        X = df_model[continuous_vars + categorical_vars]
        X = pd.get_dummies(X, columns=categorical_vars, drop_first=True)

        y = df_model["treat"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logit = LogisticRegression(max_iter=2000, random_state=42)
        logit.fit(X_scaled, y)

        df_model["ps"] = logit.predict_proba(X_scaled)[:, 1].clip(0.0001, 0.9999)
        df_model["logit_ps"] = np.log(df_model["ps"] / (1 - df_model["ps"]))

        # =========================
        # 3 PS density（保存图）
        # =========================
        plt.figure()
        sns.kdeplot(df_model[df_model["treat"] == 1]["ps"], label="MIS")
        sns.kdeplot(df_model[df_model["treat"] == 0]["ps"], label="OPEN")
        plt.title(f"{dataset_name} - PS density")
        plt.legend()

        plt.savefig(os.path.join(output_dir, f"{dataset_name}_ps_density.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        # =========================
        # 4 PSM matching
        # =========================
        treated = df_model[df_model["treat"] == 1]
        control = df_model[df_model["treat"] == 0]

        caliper = 0.2 * df_model["logit_ps"].std()

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

        matched_df = pd.concat(
            [df_model.loc[treated_idx], df_model.loc[control_idx]]
        ).copy()

        matched_df.to_excel(
            os.path.join(output_dir, f"{dataset_name}_PSM_dataset.xlsx"),
            index=False
        )

        # =========================
        # 5 SMD Love plot
        # =========================
        X_full = pd.concat([X, df_model["treat"]], axis=1)
        X_matched = pd.concat([X.loc[matched_df.index], matched_df["treat"]], axis=1)

        smd_before = [smd(X_full, c) for c in X.columns]
        smd_after = [smd(X_matched, c) for c in X.columns]

        plt.figure()
        plt.scatter(smd_before, X.columns, label="Before")
        plt.scatter(smd_after, X.columns, label="After")
        plt.axvline(0.1, linestyle="--")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title(f"{dataset_name} - Love plot")

        plt.savefig(os.path.join(output_dir, f"{dataset_name}_love_plot.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        # =========================
        # 6 Table1
        # =========================
        rows = []

        for var in continuous_vars:
            t = matched_df[matched_df["treat"] == 1][var]
            c = matched_df[matched_df["treat"] == 0][var]

            p = stats.ttest_ind(t, c).pvalue

            rows.append([
                var,
                f"{t.mean():.2f}±{t.std():.2f}",
                f"{c.mean():.2f}±{c.std():.2f}",
                p
            ])

        table1 = pd.DataFrame(rows, columns=["Variable", "MIS", "OPEN", "p"])

        table1.to_excel(
            os.path.join(output_dir, f"{dataset_name}_Table1.xlsx"),
            index=False
        )

    print("\nAll datasets processed successfully.")

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":

    set_seed(42)

    files = [
        "complications.xlsx",
        "elderly.xlsx"
    ]

    run_psm_batch(files)

