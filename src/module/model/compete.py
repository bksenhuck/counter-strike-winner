"""Model competition: train all candidates on the same split and pick the best.

Trains XGBoost, LightGBM, Logistic Regression and MLP on identical
train/val/test splits so results are directly comparable. Saves the winning
model as the production pipeline and exports comparison artefacts to
data/results/.

By default runs everything: cross-validation, SHAP analysis and saves the
best model. Use --no-* flags to disable individual steps.

Usage::

    python -m src.module.model.compete
    python -m src.module.model.compete --no-cross-val
    python -m src.module.model.compete --no-shap --no-save-best
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from conf.settings import RANDOM_STATE, RESULTS_DIR, TEST_SIZE, VAL_SIZE
from src.module.model.data import load_raw_data, preprocess
from src.module.model.evaluate import ModelEvaluator
from src.module.model.features import build_features, get_feature_matrix
from src.module.model.model import CSWinnerModel
from utils.data_processing import split_data
from utils.decorators import timer
from utils.logger import get_logger

_logger = get_logger(__name__)

_CANDIDATES = [
    "xgboost",
    "lightgbm",
    "logistic_regression",
    "neural_network",
]

_SEP = "=" * 60


@timer
def run_competition(
    cross_val: bool = True,
    save_best: bool = True,
    run_shap: bool = True,
) -> pd.DataFrame:
    """Train all candidate models on the same split and compare metrics.

    Args:
        cross_val: Run stratified k-fold CV for each model (default True).
        save_best: Persist the winning model's pipeline to disk (default True).
        run_shap: Generate SHAP summary for the winning model (default True).

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    _logger.info(_SEP)
    _logger.info("  CS WINNER — Model Competition")
    _logger.info("  Candidates: %s", ", ".join(_CANDIDATES))
    _logger.info("  cross_val=%s  save_best=%s  shap=%s", cross_val, save_best, run_shap)
    _logger.info(_SEP)

    _logger.info("[1/4] Load & preprocess data")
    raw = load_raw_data()
    df = preprocess(raw)

    _logger.info("[2/4] Feature engineering")
    df = build_features(df)
    X, y = get_feature_matrix(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    _logger.info(
        "Split — train: %d | val: %d | test: %d | features: %d",
        len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )

    _logger.info("[3/4] Train & evaluate each candidate")
    rows = []
    models: dict[str, CSWinnerModel] = {}

    for model_type in _CANDIDATES:
        _logger.info(_SEP)
        _logger.info("  >>> %s", model_type.upper())
        _logger.info(_SEP)

        model = CSWinnerModel(model_type=model_type)

        if cross_val:
            cv = model.cross_validate(X_train, y_train)
            _logger.info(
                "  CV AUC: %.4f ± %.4f", cv["mean_auc"], cv["std_auc"]
            )

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        evaluator = ModelEvaluator(model)
        metrics = evaluator.compute_metrics(X_test, y_test)

        row = {"model": model_type, **metrics}
        if cross_val:
            row["cv_auc_mean"] = cv["mean_auc"]
            row["cv_auc_std"] = cv["std_auc"]
        rows.append(row)
        models[model_type] = model

        _logger.info(
            "  %s — AUC=%.4f  Acc=%.4f  F1=%.4f",
            model_type, metrics["auc"], metrics["accuracy"], metrics["f1"],
        )

    results = pd.DataFrame(rows).set_index("model")

    _logger.info(_SEP)
    _logger.info("[4/4] Competition results")
    _logger.info(_SEP)
    _logger.info("\n%s", results.to_string())

    best_model_type = results["auc"].idxmax()
    best_auc = results.loc[best_model_type, "auc"]
    _logger.info("  Winner: %s  (AUC=%.4f)", best_model_type.upper(), best_auc)

    _save_results(results)

    winner = models[best_model_type]

    if run_shap:
        _logger.info(_SEP)
        _logger.info("  SHAP analysis for winner: %s", best_model_type.upper())
        _logger.info(_SEP)
        evaluator = ModelEvaluator(winner)
        evaluator.run_shap(
            df,
            filename=f"shap_summary_{best_model_type}.png",
        )

    if save_best:
        path = winner.save()
        _logger.info("  Best model saved → %s", path)

    _logger.info(_SEP)
    _logger.info("  Competition complete.")
    _logger.info(_SEP)
    return results


def _save_results(results: pd.DataFrame) -> None:
    """Save comparison table (xlsx) and bar chart (png) to data/results/."""
    xlsx_path = RESULTS_DIR / "model_comparison.xlsx"
    results.to_excel(xlsx_path)
    _logger.info("Comparison table saved → %s", xlsx_path)

    _save_chart(results)


def _save_chart(results: pd.DataFrame) -> None:
    metrics_to_plot = [c for c in ("auc", "accuracy", "f1") if c in results.columns]
    n_metrics = len(metrics_to_plot)
    n_models = len(results)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for ax, metric in zip(axes, metrics_to_plot):
        vals = results[metric]
        bars = ax.bar(range(n_models), vals, color=colors[:n_models], alpha=0.85)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(
            [m.replace("_", "\n") for m in results.index],
            fontsize=9,
        )
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.05))
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center",
                fontsize=9,
            )

    fig.suptitle("Model Competition — CS Winner Predictor", fontweight="bold", fontsize=13)
    fig.tight_layout()

    png_path = RESULTS_DIR / "model_comparison.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    _logger.info("Comparison chart saved → %s", png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS Winner — Model Competition")
    parser.add_argument("--no-cross-val", action="store_true", help="Skip k-fold CV")
    parser.add_argument("--no-save-best", action="store_true", help="Skip saving the winning model")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    args = parser.parse_args()
    run_competition(
        cross_val=not args.no_cross_val,
        save_best=not args.no_save_best,
        run_shap=not args.no_shap,
    )
