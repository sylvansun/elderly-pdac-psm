# =====================================
# Treatment
# =====================================

TREATMENT_COL = "Surgical approach"

TREATMENT_MAPPING = {
    "OPEN": 0,
    "MIS": 1,
}


# =====================================
# Survival
# =====================================

TIME_COL = "OS"

EVENT_COL = "Survival Status"


# =====================================
# Variables
# =====================================

VARIABLES = {
    "continuous": [
        "Age",
        "BMI",
        "Albumin",
    ],
    "categorical": [
        "AJCC Stage",
        "ASA Score",
        "pTNM_T",
        "Anaemia",
        "Surgical procedure",
    ],
}


# =====================================
# Analysis Parameters
# =====================================

RANDOM_SEED = 42

PSM_CALIPER_RATIO = 0.2

LOGISTIC_MAX_ITER = 2000

RSF_PARAMS = {
    "n_estimators": 500,
    "min_samples_split": 10,
    "min_samples_leaf": 15,
}


def continuous_vars():
    return VARIABLES["continuous"]


def categorical_vars():
    return VARIABLES["categorical"]


def all_covariates():
    return VARIABLES["continuous"] + VARIABLES["categorical"]
