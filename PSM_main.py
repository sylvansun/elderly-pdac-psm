import os
import random
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import analysis
import plots
import config as cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def preprocess_data(df):
    df_model = df.copy().sort_index()

    for col in cfg.continuous_vars():
        df_model[col] = df_model[col].fillna(df_model[col].median())

    for col in cfg.categorical_vars():
        df_model[col] = df_model[col].fillna("Unknown")

    df_model["treat"] = df_model[cfg.TREATMENT_COL].map(cfg.TREATMENT_MAPPING)

    return df_model


def fit_ps_model(df_model):

    # =========================
    # 1. 构造设计矩阵
    # =========================
    X_raw = df_model[cfg.all_covariates()].copy()

    X_encoded = pd.get_dummies(
        X_raw,
        columns=cfg.categorical_vars(),
        drop_first=True,
    )

    y = df_model["treat"]

    # =========================
    # 2. 标准化（用于logistic PS model）
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # =========================
    # 3. PS model
    # =========================
    logit = LogisticRegression(
        max_iter=cfg.LOGISTIC_MAX_ITER,
        random_state=cfg.RANDOM_SEED,
    )

    logit.fit(X_scaled, y)

    # =========================
    # 4. propensity score
    # =========================
    ps = logit.predict_proba(X_scaled)[:, 1]

    # 防止log(0)
    ps = np.clip(ps, 1e-5, 1 - 1e-5)

    df_model["ps"] = ps
    df_model["logit_ps"] = np.log(ps / (1 - ps))

    return df_model, X_encoded, scaler, logit


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


def compute_table1(matched_df):

    rows = []

    for var in cfg.continuous_vars():

        t = matched_df[matched_df["treat"] == 1][var]
        c = matched_df[matched_df["treat"] == 0][var]

        p = stats.ttest_ind(t, c).pvalue

        rows.append(
            [var, f"{t.mean():.2f}±{t.std():.2f}", f"{c.mean():.2f}±{c.std():.2f}", p]
        )

    return pd.DataFrame(rows, columns=["Variable", "MIS", "OPEN", "p"])


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
    matched_df = ps_matching(
        df_model,
        caliper_ratio=cfg.PSM_CALIPER_RATIO,
    )

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

    analysis.run_km_analysis(
        matched_df,
        dataset_name,
        output_dir,
        time_col=cfg.TIME_COL,
        event_col=cfg.EVENT_COL,
    )

    cph, result_table = analysis.run_cox_analysis(matched_df, dataset_name, output_dir)

    plots.plot_cox_forest(result_table, dataset_name, output_dir)

    rsf, ml_X, y_ml = analysis.run_rsf_analysis(matched_df, dataset_name, output_dir)

    plots.plot_time_auc(rsf, ml_X, y_ml, dataset_name, output_dir)

    plots.plot_feature_importance(rsf, ml_X, y_ml, dataset_name, output_dir)

    analysis.run_two_year_os_analysis(matched_df, dataset_name, output_dir)

    matched_df.to_excel(
        os.path.join(output_dir, f"{dataset_name}_PSM_dataset.xlsx"), index=False
    )

    return df_model, matched_df


def run_psm_batch(file_list, output_dir="PSM_results"):

    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_list:
        run_single_dataset(file_path, output_dir)

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":

    set_seed(cfg.RANDOM_SEED)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(BASE_DIR, "data")

    files = [
        os.path.join(DATA_DIR, f)
        for f in [
            "complications.xlsx",
            "elderly.xlsx",
        ]
    ]

    run_psm_batch(files)
