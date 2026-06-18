"""Model-local configuration — hyperparameters, feature schema, SHAP display.

These are domain/model-specific values that do not depend on the deployment
environment. Change here to tune the model without touching any logic.
"""

import os

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COL: str = "winner"

# Feature schema lives in src/module/model/features/ (one list per theme).
# FEATURE_COLS and BASE_NUMERIC_FEATURES are assembled in features/__init__.py.

# ---------------------------------------------------------------------------
# sklearn Pipeline step names
# ---------------------------------------------------------------------------
IMPUTER_STEP: str = "imputer"
SCALER_STEP: str = "scaler"
CLASSIFIER_STEP: str = "classifier"
IMPUTER_STRATEGY: str = "median"

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------------
XGB_N_ESTIMATORS: int = int(os.getenv("XGB_N_ESTIMATORS", "300"))
XGB_MAX_DEPTH: int = int(os.getenv("XGB_MAX_DEPTH", "5"))
XGB_LEARNING_RATE: float = float(os.getenv("XGB_LEARNING_RATE", "0.05"))
XGB_SUBSAMPLE: float = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
XGB_COLSAMPLE_BYTREE: float = float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8"))
XGB_EVAL_METRIC: str = "logloss"
XGB_N_JOBS: int = -1

# ---------------------------------------------------------------------------
# Persisted model
# ---------------------------------------------------------------------------
PIPELINE_NAME: str = os.getenv("PIPELINE_NAME", "xgboost_pipeline")

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_N_SPLITS: int = int(os.getenv("CV_N_SPLITS", "5"))
CV_SCORING: str = "roc_auc"

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
PREDICTION_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# SHAP display
# ---------------------------------------------------------------------------
SHAP_MAX_DISPLAY_SUMMARY: int = 15
SHAP_MAX_DISPLAY_WATERFALL: int = 10
SHAP_TOP_REASONS_N: int = 5
SHAP_FIGSIZE_SUMMARY: tuple[int, int] = (10, 6)
SHAP_FIGSIZE_WATERFALL: tuple[int, int] = (10, 5)
SHAP_DPI: int = 150
SHAP_TITLE_SUMMARY: str = "SHAP Feature Importance — CS Match Winner"
