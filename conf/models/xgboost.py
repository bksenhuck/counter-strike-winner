"""XGBoost classifier configuration.

All hyperparameters can be overridden via environment variables.
"""

import os

from xgboost import XGBClassifier

PARAMS: dict = {
    "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "300")),
    "max_depth": int(os.getenv("XGB_MAX_DEPTH", "5")),
    "learning_rate": float(os.getenv("XGB_LEARNING_RATE", "0.05")),
    "subsample": float(os.getenv("XGB_SUBSAMPLE", "0.8")),
    "colsample_bytree": float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8")),
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def build_classifier(random_state: int) -> XGBClassifier:
    return XGBClassifier(**PARAMS, random_state=random_state)
