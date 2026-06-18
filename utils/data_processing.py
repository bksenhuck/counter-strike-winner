"""Data processing utilities — cleaning, splitting, scaling and transformation.

Single file for all generic data operations used across the pipeline.
Feature-specific encoding (e.g. encode_format) lives in src/module/model/features/.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from conf.settings import IQR_OUTLIER_FACTOR, MISSING_COL_THRESHOLD, RANDOM_STATE, ROLLING_WINDOW, TEST_SIZE, VAL_SIZE
from utils.decorators import log_call, timer
from utils.logger import get_logger

_logger = get_logger(__name__)

ScalerType = Literal["standard", "minmax", "robust"]

_SCALERS: dict[str, type] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def fill_feature_means(df: pd.DataFrame, cols: list[str]) -> None:
    """Fill NaN in feature columns with the column mean (in-place).

    Used at the end of every feature module to ensure no NaN is passed to the
    model from rows with insufficient history.
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())


@log_call
def drop_missing(df: pd.DataFrame, threshold: float = MISSING_COL_THRESHOLD) -> pd.DataFrame:
    """Drop columns where the fraction of NaN exceeds `threshold`."""
    missing_frac = df.isna().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    if cols_to_drop:
        _logger.warning("Dropping %d high-NaN columns: %s", len(cols_to_drop), cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df


@log_call
def fill_numeric_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in numeric columns with the column median."""
    num_cols = df.select_dtypes(include="number").columns
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    return df


@log_call
def remove_outliers_iqr(
    df: pd.DataFrame,
    cols: list[str],
    factor: float = IQR_OUTLIER_FACTOR,
) -> pd.DataFrame:
    """Remove rows where any col falls outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= df[col].between(q1 - factor * iqr, q3 + factor * iqr)
    removed = (~mask).sum()
    if removed:
        _logger.info("Removed %d outlier rows (IQR factor=%.1f).", removed, factor)
    return df[mask].reset_index(drop=True)


@log_call
def clip_outliers(
    df: pd.DataFrame,
    cols: list[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Clip values in `cols` to percentile bounds (keeps all rows)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            _logger.warning("Column '%s' not found, skipping clip.", col)
            continue
        lo = df[col].quantile(lower_quantile)
        hi = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lo, hi)
        _logger.debug("Clipped '%s' to [%.4f, %.4f]", col, lo, hi)
    return df


@log_call
def log_transform(df: pd.DataFrame, cols: list[str], offset: float = 1.0) -> pd.DataFrame:
    """Apply log1p to right-skewed columns (e.g. kill counts)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            _logger.warning("Column '%s' not found, skipping log transform.", col)
            continue
        df[col] = np.log1p(df[col] + offset - 1)
        _logger.debug("Log-transformed '%s'", col)
    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@log_call
def encode_categorical(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode `cols` in-place; return (df, dict of fitted encoders)."""
    encoders: dict[str, LabelEncoder] = {}
    for col in cols:
        if col not in df.columns:
            _logger.warning("Column '%s' not found, skipping encoding.", col)
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        _logger.debug("Encoded '%s': %d classes", col, len(le.classes_))
    return df, encoders


# ---------------------------------------------------------------------------
# Rolling stats
# ---------------------------------------------------------------------------

@log_call
def add_rolling_stats(
    df: pd.DataFrame,
    group_col: str,
    stat_cols: list[str],
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Add rolling mean of `stat_cols` per `group_col` (sorted by date)."""
    df = df.sort_values("date").copy()
    for col in stat_cols:
        if col not in df.columns:
            continue
        new_col = f"{col}_roll{window}"
        df[new_col] = df.groupby(group_col)[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        _logger.debug("Added rolling feature: %s", new_col)
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

@timer
@log_call
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train / val / test split.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp, random_state=random_state
    )
    _logger.info(
        "Split sizes -> train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

def get_scaler(kind: ScalerType = "standard") -> StandardScaler | MinMaxScaler | RobustScaler:
    """Return an unfitted sklearn scaler by name ('standard', 'minmax', 'robust')."""
    if kind not in _SCALERS:
        raise ValueError(f"Unknown scaler '{kind}'. Choose from {list(_SCALERS)}")
    return _SCALERS[kind]()


@log_call
def fit_scaler(
    df: pd.DataFrame,
    cols: list[str],
    kind: ScalerType = "standard",
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler | RobustScaler]:
    """Fit a scaler on `cols` and return (transformed_df, fitted_scaler).

    The fitted scaler can be persisted with utils.file_utils.save_pkl
    and reloaded at inference time.
    """
    scaler = get_scaler(kind)
    df = df.copy()
    df[cols] = scaler.fit_transform(df[cols])
    _logger.info("Fitted %s scaler on %d columns.", kind, len(cols))
    return df, scaler


@log_call
def apply_scaler(
    df: pd.DataFrame,
    cols: list[str],
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
) -> pd.DataFrame:
    """Apply a fitted scaler (transform-only, no fit — use at inference time)."""
    df = df.copy()
    df[cols] = scaler.transform(df[cols])
    _logger.info("Applied fitted scaler to %d columns.", len(cols))
    return df


# ---------------------------------------------------------------------------
# ColumnTransformer / Pipeline factories
# ---------------------------------------------------------------------------

@log_call
def build_column_transformer(
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    numeric_scaler: ScalerType = "standard",
    remainder: str = "passthrough",
) -> ColumnTransformer:
    """sklearn ColumnTransformer: scales numerics and optionally encodes categoricals."""
    from sklearn.preprocessing import OrdinalEncoder

    transformers = [("num", get_scaler(numeric_scaler), numeric_cols)]
    if categorical_cols:
        transformers.append(
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols)
        )
    ct = ColumnTransformer(transformers=transformers, remainder=remainder)
    _logger.debug(
        "ColumnTransformer built: %d numeric, %d categorical cols.",
        len(numeric_cols),
        len(categorical_cols) if categorical_cols else 0,
    )
    return ct


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    scaler: ScalerType = "standard",
) -> Pipeline:
    """Standalone sklearn preprocessing Pipeline (no classifier).

    Use to inspect transformed distributions or export the scaler separately.
    """
    ct = build_column_transformer(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_scaler=scaler,
    )
    pipeline = Pipeline(steps=[("preprocessor", ct)])
    _logger.info(
        "Preprocessing pipeline built (scaler=%s, %d numeric cols).", scaler, len(numeric_cols)
    )
    return pipeline
