"""SHAP-based explainability encapsulated in ModelExplainer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from conf.settings import MODELS_DIR, RESULTS_DIR
from src.module.model.features import FEATURE_COLS, build_features
from src.module.model.model import CSWinnerModel
from src.module.model.settings import (
    CLASSIFIER_STEP,
    PIPELINE_NAME,
    SHAP_DPI,
    SHAP_FIGSIZE_SUMMARY,
    SHAP_FIGSIZE_WATERFALL,
    SHAP_MAX_DISPLAY_SUMMARY,
    SHAP_MAX_DISPLAY_WATERFALL,
    SHAP_TITLE_SUMMARY,
    SHAP_TOP_REASONS_N,
)
from utils.decorators import log_call, timer
from utils.logger import get_logger

_logger = get_logger(__name__)


class ModelExplainer:
    """SHAP-based explainer for the XGBoost Pipeline.

    Answers two questions:
      - Global: which features matter most across ALL matches? → shap_summary()
      - Local:  why does the model pick team1/team2 for THIS match? → explain_match()

    Usage
    -----
    >>> explainer = ModelExplainer().load()
    >>> explainer.shap_summary(df)
    >>> result = explainer.explain_match({"rating_team1": 1.2, ...})
    """

    def __init__(
        self,
        pipeline_name: str = PIPELINE_NAME,
        models_dir: Path = MODELS_DIR,
        plots_dir: Path = RESULTS_DIR,
        model: CSWinnerModel | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.models_dir = models_dir
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self._model: CSWinnerModel | None = model  # accept pre-loaded model

    def load(self) -> ModelExplainer:
        """Load the trained CSWinnerModel from disk. Returns self for chaining."""
        self._model = CSWinnerModel.load(
            pipeline_name=self.pipeline_name,
            models_dir=self.models_dir,
        )
        return self

    @timer
    @log_call
    def shap_summary(
        self,
        df: pd.DataFrame,
        save: bool = False,
        filename: str = "shap_summary.png",
    ) -> shap.Explanation:
        """Global SHAP beeswarm — which features drive predictions across all matches.

        Args:
            df: Feature-engineered DataFrame with FEATURE_COLS.
            save: Save the figure to plots_dir when True.
            filename: Output filename (used when save=True).

        Returns:
            shap.Explanation for programmatic analysis.
        """
        self._ensure_loaded()
        X = df[FEATURE_COLS].dropna()
        X_transformed = self._transform(X)

        explainer = shap.TreeExplainer(self._model.pipeline.named_steps[CLASSIFIER_STEP])
        shap_values = explainer(X_transformed)
        shap_values.feature_names = FEATURE_COLS

        _logger.info("Computing SHAP summary for %d samples…", len(X))

        plt.figure(figsize=SHAP_FIGSIZE_SUMMARY)
        shap.plots.beeswarm(shap_values, max_display=SHAP_MAX_DISPLAY_SUMMARY, show=False)
        plt.title(SHAP_TITLE_SUMMARY)
        plt.tight_layout()

        if save:
            self._save_figure(filename)

        plt.show()
        return shap_values

    @timer
    @log_call
    def explain_match(
        self,
        match_data: dict | pd.Series,
        save: bool = False,
        filename: str = "shap_match.png",
    ) -> dict:
        """Local SHAP waterfall — why the model predicts team1/team2 for one match.

        Args:
            match_data: Raw match stats dict or Series (pre-feature-engineering).
            save: Save the waterfall figure to plots_dir when True.
            filename: Output filename (used when save=True).

        Returns:
            dict with keys:
                predicted_winner, prob_team1_wins, prob_team2_wins,
                top_reasons (list[{feature, shap_value, feature_value}]).
        """
        self._ensure_loaded()

        df = (
            pd.DataFrame([match_data])
            if isinstance(match_data, dict)
            else match_data.to_frame().T
        )
        df = build_features(df)
        X = df[FEATURE_COLS].fillna(0)
        X_transformed = self._transform(X)

        explainer = shap.TreeExplainer(self._model.pipeline.named_steps[CLASSIFIER_STEP])
        shap_values = explainer(X_transformed)
        shap_values.feature_names = FEATURE_COLS

        sv = shap_values[0]
        base_prob = float(shap.utils.sigmoid(sv.base_values))
        pred_prob = float(shap.utils.sigmoid(sv.base_values + sv.values.sum()))

        plt.figure(figsize=SHAP_FIGSIZE_WATERFALL)
        shap.plots.waterfall(sv, max_display=SHAP_MAX_DISPLAY_WATERFALL, show=False)
        plt.title(f"Match explanation — team1 win prob: {pred_prob:.2%}")
        plt.tight_layout()

        if save:
            self._save_figure(filename)

        plt.show()

        top_reasons = (
            pd.DataFrame(
                {
                    "feature": FEATURE_COLS,
                    "shap_value": sv.values,
                    "feature_value": X.iloc[0].values,
                }
            )
            .reindex(pd.Series(sv.values).abs().sort_values(ascending=False).index)
            .head(SHAP_TOP_REASONS_N)
            .to_dict(orient="records")
        )

        predicted_winner = "team1" if pred_prob >= 0.5 else "team2"
        _logger.info(
            "Match: predicted=%s  prob_team1=%.4f  base_prob=%.4f",
            predicted_winner,
            pred_prob,
            base_prob,
        )
        return {
            "predicted_winner": predicted_winner,
            "prob_team1_wins": round(pred_prob, 4),
            "prob_team2_wins": round(1 - pred_prob, 4),
            "top_reasons": top_reasons,
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all Pipeline steps except the final classifier."""
        steps = list(self._model.pipeline.named_steps.items())
        X_out = X.copy()
        for _, step in steps[:-1]:
            X_out = step.transform(X_out)
        return pd.DataFrame(X_out, columns=FEATURE_COLS)

    def _save_figure(self, filename: str) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        path = self.plots_dir / filename
        plt.savefig(path, dpi=SHAP_DPI)
        _logger.info("Figure saved -> %s", path)

    def _ensure_loaded(self) -> None:
        if self._model is None or self._model.pipeline is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
