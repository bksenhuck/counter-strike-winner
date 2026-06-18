"""CSWinnerModel — all model logic in one place.

Responsibilities:
  - Build the sklearn Pipeline (imputer → scaler → XGBoost)
  - Train, evaluate, and cross-validate
  - Save / load the fitted pipeline
  - Predict (batch labels, batch probabilities, single-match raw dict)

train.py and predict.py are thin entry points that instantiate this class.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from conf.settings import MODELS_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE
from src.module.model.features import (
    FEATURE_COLS,
    build_features,
    build_team_stats_lookup,
    get_feature_matrix,
)
from src.module.model.settings import (
    CLASSIFIER_STEP,
    CV_N_SPLITS,
    CV_SCORING,
    IMPUTER_STEP,
    IMPUTER_STRATEGY,
    PIPELINE_NAME,
    PREDICTION_THRESHOLD,
    SCALER_STEP,
    XGB_COLSAMPLE_BYTREE,
    XGB_EVAL_METRIC,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_N_JOBS,
    XGB_SUBSAMPLE,
)
from utils.decorators import log_call, timer, validate_dataframe
from utils.file_utils import load_pkl, save_pkl
from utils.logger import get_logger

_logger = get_logger(__name__)


class CSWinnerModel:
    """XGBoost-based classifier for CS match winner prediction.

    Usage — training
    ----------------
    >>> model = CSWinnerModel()
    >>> model.fit(X_train, y_train)
    >>> model.evaluate(X_test, y_test)
    >>> model.save()

    Usage — inference
    -----------------
    >>> model = CSWinnerModel.load()
    >>> labels = model.predict(df)
    >>> probas  = model.predict(df, return_proba=True)
    >>> result  = model.predict_from_raw({"rating_team1": 1.2, ...})
    """

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        n_splits: int = CV_N_SPLITS,
        models_dir: Path = MODELS_DIR,
        pipeline_name: str = PIPELINE_NAME,
        threshold: float = PREDICTION_THRESHOLD,
    ) -> None:
        self.random_state = random_state
        self.n_splits = n_splits
        self.models_dir = models_dir
        self.pipeline_name = pipeline_name
        self.threshold = threshold
        self.pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @timer
    @log_call
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> CSWinnerModel:
        """Fit the Pipeline on training data and optionally evaluate on val set.

        Returns:
            self, for method chaining.
        """
        _logger.info(
            "=== Training Pipeline: SimpleImputer → StandardScaler → XGBClassifier ===",
        )
        _logger.info(
            "  Train samples: %d | Features: %d",
            len(X_train), X_train.shape[1],
        )
        _logger.info(
            "  XGBoost params: n_estimators=%d  max_depth=%d  lr=%.3f  subsample=%.2f  colsample=%.2f",
            XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
        )

        self.pipeline = self._build_pipeline()

        _logger.info("  [1/3] SimpleImputer (strategy=%s)...", IMPUTER_STRATEGY)
        _logger.info("  [2/3] StandardScaler...")
        _logger.info("  [3/3] XGBClassifier (fitting %d estimators)...", XGB_N_ESTIMATORS)
        t0 = time.perf_counter()
        self.pipeline.fit(X_train, y_train)
        _logger.info("  Pipeline fit complete in %.1fs", time.perf_counter() - t0)

        if X_val is not None and y_val is not None:
            _logger.info("--- Validation set metrics ---")
            self.evaluate(X_val, y_val)

        return self

    @timer
    @log_call
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Run stratified k-fold CV and return mean / std AUC.

        Folds run sequentially so each XGBClassifier can use all CPU cores
        without nested-parallelism conflicts (important on Windows).

        Returns:
            dict with keys: mean_auc, std_auc.
        """
        _logger.info(
            "=== %d-Fold Stratified Cross-Validation ===", self.n_splits,
        )
        _logger.info(
            "  Samples: %d | Features: %d | Scoring: %s",
            len(X), X.shape[1], CV_SCORING,
        )

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            _logger.info(
                "  Fold %d/%d — train=%d  val=%d",
                fold, self.n_splits, len(train_idx), len(val_idx),
            )
            t0 = time.perf_counter()
            fold_pipeline = self._build_pipeline()
            fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            auc = roc_auc_score(
                y.iloc[val_idx],
                fold_pipeline.predict_proba(X.iloc[val_idx])[:, 1],
            )
            scores.append(auc)
            _logger.info("         AUC: %.4f  (%.1fs)", auc, time.perf_counter() - t0)

        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))
        _logger.info(
            "  CV result: AUC = %.4f ± %.4f  (folds: %s)",
            mean_auc, std_auc,
            "  ".join(f"{s:.4f}" for s in scores),
        )
        return {"mean_auc": mean_auc, "std_auc": std_auc}

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Log AUC + classification report and return AUC score."""
        self._ensure_fitted()
        y_pred = self.pipeline.predict(X)
        y_proba = self.pipeline.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        _logger.info("AUC: %.4f", auc)
        _logger.info(
            "\n%s",
            classification_report(y, y_pred, target_names=["team2 wins", "team1 wins"]),
        )
        return {"auc": auc}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """Persist the fitted pipeline and return the saved path."""
        self._ensure_fitted()
        path = self.models_dir / f"{self.pipeline_name}.pkl"
        return save_pkl(self.pipeline, path)

    @classmethod
    def load(
        cls,
        pipeline_name: str = PIPELINE_NAME,
        models_dir: Path = MODELS_DIR,
    ) -> CSWinnerModel:
        """Load a persisted pipeline from disk and return a ready model instance."""
        path = models_dir / f"{pipeline_name}.pkl"
        instance = cls(pipeline_name=pipeline_name, models_dir=models_dir)
        instance.pipeline = load_pkl(path)
        return instance

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @timer
    @log_call
    @validate_dataframe(*FEATURE_COLS)
    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Return predicted winner labels or class probabilities.

        Args:
            df: DataFrame with FEATURE_COLS.
            return_proba: When True return DataFrame[prob_team2_wins, prob_team1_wins].

        Returns:
            Series of int labels (0 = team2, 1 = team1), or probability DataFrame.
        """
        self._ensure_fitted()
        X = df[FEATURE_COLS].copy()
        _logger.info("Inference on %d rows", len(X))

        if return_proba:
            proba = self.pipeline.predict_proba(X)
            return pd.DataFrame(
                proba,
                columns=["prob_team2_wins", "prob_team1_wins"],
                index=X.index,
            )

        preds = pd.Series(self.pipeline.predict(X), index=X.index, name="predicted_winner")
        _logger.info(
            "Results: team1_wins=%d  team2_wins=%d",
            (preds == 1).sum(),
            (preds == 0).sum(),
        )
        return preds

    @timer
    @log_call
    def predict_match(
        self,
        team1: str,
        team2: str,
        match_format: str = "bo3",
        stage: str | None = None,
        team_stats: dict | None = None,
    ) -> dict:
        """Pre-match prediction using only team names and historical data.

        This is the real-world prediction API — no current-match stats needed.

        Args:
            team1: Name of team 1 (must match historical data spelling).
            team2: Name of team 2.
            match_format: Series format — "bo1", "bo3" or "bo5".
            stage: Optional stage label (e.g. "Grand Final", "Group Stage").
            team_stats: Lookup dict from build_team_stats_lookup(). Build it
                        once per session and reuse to avoid reloading data.

        Returns:
            dict with predicted_winner, team1, team2, prob_team1_wins,
            prob_team2_wins, elo_diff.

        Example::

            from src.module.model.data import load_raw_data, preprocess
            from src.module.model.features import build_features, build_team_stats_lookup

            raw = load_raw_data()
            df  = preprocess(raw)
            df  = build_features(df)
            lookup = build_team_stats_lookup(df)

            model  = CSWinnerModel.load()
            result = model.predict_match("NaVi", "Astralis", "bo3",
                                         stage="Grand Final", team_stats=lookup)
        """
        self._ensure_fitted()

        if team_stats is None:
            raise ValueError(
                "Pass team_stats=build_team_stats_lookup(df) to predict_match(). "
                "Build it once per session from the full feature DataFrame."
            )

        t1_stats = team_stats.get(team1, {})
        t2_stats = team_stats.get(team2, {})

        if not t1_stats:
            _logger.warning("No historical data found for '%s'. Using defaults.", team1)
        if not t2_stats:
            _logger.warning("No historical data found for '%s'. Using defaults.", team2)

        row: dict[str, float] = {}
        for stat, val in t1_stats.items():
            row[f"team1_{stat}"] = val
        for stat, val in t2_stats.items():
            row[f"team2_{stat}"] = val

        # Format context (known before match)
        row["format_maps"] = {"bo1": 1, "bo3": 3, "bo5": 5}.get(match_format.lower(), 3)

        # Derived pairwise features not captured per-team
        row.setdefault("elo_diff", row.get("team1_elo", 1000.0) - row.get("team2_elo", 1000.0))
        row.setdefault("elo_ratio", row.get("team1_elo", 1000.0) / max(row.get("team2_elo", 1.0), 1e-6))
        row.setdefault("overall_wr_diff", row.get("team1_overall_wr", 0.5) - row.get("team2_overall_wr", 0.5))
        row.setdefault("momentum_diff", row.get("team1_rating_trend", 0.0) - row.get("team2_rating_trend", 0.0))
        row.setdefault("rating_ma_diff", row.get("team1_rating_ma5", 1.0) - row.get("team2_rating_ma5", 1.0))
        row.setdefault("form_advantage", row.get("team1_win_rate_last10", 0.5) - row.get("team2_win_rate_last10", 0.5))
        row.setdefault("firepower_diff", row.get("team1_firepower", 0.5) - row.get("team2_firepower", 0.5))
        row.setdefault("star_dependency_diff", row.get("team1_star_dependency", 1.0) - row.get("team2_star_dependency", 1.0))
        row.setdefault("consistency_diff", row.get("team1_consistency", 10.0) - row.get("team2_consistency", 10.0))
        row.setdefault("form3_diff", row.get("team1_win_rate_last3", 0.5) - row.get("team2_win_rate_last3", 0.5))
        row.setdefault("form_30d_diff", row.get("team1_win_rate_30d", 0.5) - row.get("team2_win_rate_30d", 0.5))
        row.setdefault("rest_advantage", row.get("team1_days_rest", 7.0) - row.get("team2_days_rest", 7.0))
        row.setdefault("fatigue_advantage", row.get("team2_matches_30d", 5.0) - row.get("team1_matches_30d", 5.0))

        # Stage context (optional)
        if stage:
            from src.module.model.features.event_stage_features import _encode_stage
            encoded = float(_encode_stage(stage))
            row["stage_encoded"] = encoded
            row["is_playoff"] = float(encoded >= 1)
            row["is_final"] = float(encoded >= 3)
            row["pressure_score"] = encoded / 4.0

        df = pd.DataFrame([row])
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        X = df[[c for c in FEATURE_COLS if c in df.columns]]
        proba = self.pipeline.predict_proba(X)[0]
        team1_prob = float(proba[1])

        result = {
            "predicted_winner": team1 if team1_prob >= self.threshold else team2,
            "team1": team1,
            "team2": team2,
            "prob_team1_wins": round(team1_prob, 4),
            "prob_team2_wins": round(1 - team1_prob, 4),
            "elo_diff": round(row.get("elo_diff", 0.0), 1),
        }
        _logger.info(
            "predict_match: %s vs %s → %s (p=%.4f)",
            team1, team2, result["predicted_winner"], team1_prob,
        )
        return result

    @timer
    @log_call
    def predict_from_raw(self, match_data: dict) -> dict:
        """Single-match inference from a raw dict (applies feature engineering internally).

        Returns:
            dict with keys: predicted_winner, prob_team1_wins, prob_team2_wins.
        """
        self._ensure_fitted()
        df = build_features(pd.DataFrame([match_data]))
        proba_df = self.predict(df, return_proba=True)
        team1_prob = float(proba_df["prob_team1_wins"].iloc[0])
        return {
            "predicted_winner": "team1" if team1_prob >= self.threshold else "team2",
            "prob_team1_wins": round(team1_prob, 4),
            "prob_team2_wins": round(1 - team1_prob, 4),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (IMPUTER_STEP, SimpleImputer(strategy=IMPUTER_STRATEGY)),
                (SCALER_STEP, StandardScaler()),
                (
                    CLASSIFIER_STEP,
                    XGBClassifier(
                        n_estimators=XGB_N_ESTIMATORS,
                        max_depth=XGB_MAX_DEPTH,
                        learning_rate=XGB_LEARNING_RATE,
                        subsample=XGB_SUBSAMPLE,
                        colsample_bytree=XGB_COLSAMPLE_BYTREE,
                        eval_metric=XGB_EVAL_METRIC,
                        random_state=self.random_state,
                        n_jobs=XGB_N_JOBS,
                    ),
                ),
            ]
        )

    def _ensure_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
