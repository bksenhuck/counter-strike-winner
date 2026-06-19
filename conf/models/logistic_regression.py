"""Logistic Regression classifier configuration.

All hyperparameters can be overridden via environment variables.
"""

import os

from sklearn.linear_model import LogisticRegression

PARAMS: dict = {
    "C": float(os.getenv("LR_C", "1.0")),
    "max_iter": int(os.getenv("LR_MAX_ITER", "1000")),
    "solver": os.getenv("LR_SOLVER", "lbfgs"),
}


def build_classifier(random_state: int) -> LogisticRegression:
    return LogisticRegression(**PARAMS, random_state=random_state)
