"""LightGBM classifier configuration.

All hyperparameters can be overridden via environment variables.
"""

import os

from lightgbm import LGBMClassifier

PARAMS: dict = {
    "n_estimators": int(os.getenv("LGBM_N_ESTIMATORS", "300")),
    "max_depth": int(os.getenv("LGBM_MAX_DEPTH", "5")),
    "learning_rate": float(os.getenv("LGBM_LEARNING_RATE", "0.05")),
    "subsample": float(os.getenv("LGBM_SUBSAMPLE", "0.8")),
    "colsample_bytree": float(os.getenv("LGBM_COLSAMPLE_BYTREE", "0.8")),
    "n_jobs": -1,
    "verbose": -1,
}


def build_classifier(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(**PARAMS, random_state=random_state)
