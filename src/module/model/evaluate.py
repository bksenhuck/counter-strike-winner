"""ModelEvaluator — centralised model evaluation, metrics and visualisation.

Responsibilities:
  - Compute and log classification metrics (AUC, F1, precision, recall, accuracy)
  - Plot and save confusion matrix and ROC curve
  - Run SHAP global summary and optional per-match waterfall via ModelExplainer
  - Aggregate everything in a single run() call for use in train.py or notebooks

explain.py owns the SHAP machinery; evaluate.py calls it as a dependency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from conf.settings import MODELS_DIR, PLOTS_DIR
from utils.decorators import log_call, timer
from utils.logger import get_logger

_logger = get_logger(__name__)


class ModelEvaluator:
    """Run and persist all evaluation artefacts for a fitted CSWinnerModel.

    Usage::

        from src.module.model.evaluate import ModelEvaluator
        from src.module.model.model import CSWinnerModel

        model    = CSWinnerModel.load()
        evaluator = ModelEvaluator(model)

        metrics = evaluator.compute_metrics(X_test, y_test)
        evaluator.plot_confusion_matrix(X_test, y_test)
        evaluator.plot_roc_curve(X_test, y_test)
        evaluator.run_shap(df_features)

        # Or everything at once:
        evaluator.run(X_test, y_test, df_features)
    """

    def __init__(
        self,
        model,
        plots_dir: Path = PLOTS_DIR,
    ) -> None:
        self.model = model
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    @timer
    @log_call
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df_features: pd.DataFrame | None = None,
        shap_filename: str = "shap_summary.png",
    ) -> dict[str, float]:
        """Run all evaluation steps and return the metrics dict.

        Args:
            X: Feature matrix (test set).
            y: True labels (test set).
            df_features: Full feature DataFrame for SHAP; skipped when None.
            shap_filename: Filename for the SHAP beeswarm plot.

        Returns:
            dict with auc, accuracy, f1, precision, recall.
        """
        _logger.info("=== Model Evaluation ===")
        metrics = self.compute_metrics(X, y)
        self.plot_confusion_matrix(X, y)
        self.plot_roc_curve(X, y)
        if df_features is not None:
            self.run_shap(df_features, filename=shap_filename)
        _logger.info("=== Evaluation Complete ===")
        return metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @log_call
    def compute_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float | None = None,
    ) -> dict[str, float]:
        """Compute and log AUC, accuracy, F1, precision and recall.

        Args:
            threshold: Override the model's default prediction threshold.

        Returns:
            dict with keys: auc, accuracy, f1, precision, recall.
        """
        threshold = threshold or self.model.threshold
        pipeline = self.model.pipeline
        y_proba = pipeline.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y, y_proba)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
        }

        _logger.info(
            "AUC=%.4f  Accuracy=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f",
            metrics["auc"], metrics["accuracy"], metrics["f1"],
            metrics["precision"], metrics["recall"],
        )
        _logger.info(
            "\n%s",
            classification_report(
                y, y_pred, target_names=["team2 wins", "team1 wins"], zero_division=0
            ),
        )
        return metrics

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    @log_call
    def plot_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        filename: str = "confusion_matrix.png",
        normalize: str = "true",
    ) -> Path:
        """Save a normalised confusion matrix plot."""
        pipeline = self.model.pipeline
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(
            pipeline, X, y,
            display_labels=["team2 wins", "team1 wins"],
            normalize=normalize,
            ax=ax,
            colorbar=False,
        )
        ax.set_title("Confusion Matrix (normalised)")
        path = self._save(fig, filename)
        _logger.info("Confusion matrix saved → %s", path)
        return path

    @log_call
    def plot_roc_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        filename: str = "roc_curve.png",
    ) -> Path:
        """Save an ROC curve plot."""
        pipeline = self.model.pipeline
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(pipeline, X, y, ax=ax, name="XGBoost")
        ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.50)")
        ax.set_title("ROC Curve — CS Match Winner")
        ax.legend(loc="lower right")
        path = self._save(fig, filename)
        _logger.info("ROC curve saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # SHAP integration
    # ------------------------------------------------------------------

    @log_call
    def run_shap(
        self,
        df_features: pd.DataFrame,
        filename: str = "shap_summary.png",
        explain_idx: int | None = None,
        waterfall_filename: str = "shap_waterfall.png",
    ) -> None:
        """Run SHAP global summary and optionally a per-row waterfall.

        Args:
            df_features: Full feature DataFrame (after build_features()).
            filename: Output file for the beeswarm summary plot.
            explain_idx: If given, also produce a waterfall plot for that row index.
            waterfall_filename: Output file for the waterfall plot.
        """
        from src.module.model.explain import ModelExplainer

        explainer = ModelExplainer(model=self.model, plots_dir=self.plots_dir)
        explainer.shap_summary(df_features, save=True, filename=filename)

        if explain_idx is not None and explain_idx < len(df_features):
            row = df_features.iloc[[explain_idx]]
            match_data = row.to_dict("records")[0]
            explainer.explain_match(match_data, save=True, filename=waterfall_filename)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        path = self.plots_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
