"""Training entry point.

Responsibilities (single):
  Load data → build features → split → fit CSWinnerModel → evaluate → save.

The sklearn Pipeline used internally by CSWinnerModel is:
    SimpleImputer(median) → StandardScaler → XGBClassifier

All model logic lives in model.py; evaluation logic in evaluate.py.
This script is a thin orchestrator.
"""

import argparse

from conf.settings import RANDOM_STATE, TEST_SIZE, VAL_SIZE
from src.module.model.data import load_raw_data, preprocess
from src.module.model.evaluate import ModelEvaluator
from src.module.model.features import build_features, build_team_stats_lookup, get_feature_matrix
from src.module.model.model import CSWinnerModel
from utils.data_processing import split_data
from utils.decorators import timer
from utils.logger import get_logger

_logger = get_logger(__name__)


@timer
def run(cross_val: bool = False, run_shap: bool = False) -> tuple[CSWinnerModel, object]:
    """Execute the full training pipeline end-to-end.

    Args:
        cross_val: When True, run stratified k-fold CV before the final fit.
        run_shap: When True, generate SHAP summary plot after evaluation.

    Returns:
        (fitted CSWinnerModel, its internal sklearn Pipeline)
    """
    _sep = "=" * 60
    n_steps = 5 + int(cross_val) + int(run_shap)
    step = 0

    def _banner(title: str) -> None:
        nonlocal step
        step += 1
        _logger.info(_sep)
        _logger.info("[%d/%d] %s", step, n_steps, title)
        _logger.info(_sep)

    _logger.info(_sep)
    _logger.info("  CS WINNER PREDICTOR — Training Run")
    _logger.info("  cross_val=%s  shap=%s", cross_val, run_shap)
    _logger.info(_sep)

    _banner("Load raw data")
    raw = load_raw_data()

    _banner("Preprocess")
    df = preprocess(raw)

    _banner("Feature engineering")
    df = build_features(df)
    X, y = get_feature_matrix(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    _logger.info(
        "Data split — train: %d | val: %d | test: %d | features: %d",
        len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )

    model = CSWinnerModel()

    if cross_val:
        _banner("Cross-validation")
        cv_results = model.cross_validate(X_train, y_train)
        _logger.info(
            "CV summary — AUC: %.4f ± %.4f",
            cv_results["mean_auc"], cv_results["std_auc"],
        )

    _banner("Train final model")
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    _banner("Evaluate on test set")
    evaluator = ModelEvaluator(model)
    evaluator.run(
        X_test,
        y_test,
        df_features=df if run_shap else None,
    )

    if run_shap:
        _banner("SHAP analysis")

    path = model.save()
    _logger.info("Pipeline saved → %s", path)

    _logger.info("--- Pre-match prediction demo ---")
    lookup = build_team_stats_lookup(df)
    teams = list(lookup.keys())
    if len(teams) >= 2:
        result = model.predict_match(teams[0], teams[1], match_format="bo3", team_stats=lookup)
        _logger.info(
            "  %s  vs  %s  →  winner: %s  (p1=%.3f | p2=%.3f)",
            result["team1"], result["team2"],
            result["predicted_winner"],
            result["prob_team1_wins"],
            result["prob_team2_wins"],
        )

    _logger.info(_sep)
    _logger.info("  Training pipeline complete.")
    _logger.info(_sep)
    return model, model.pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CS Winner Predictor")
    parser.add_argument("--cross-val", action="store_true", help="Run k-fold CV before training")
    parser.add_argument("--shap", action="store_true", help="Generate SHAP summary plot")
    args = parser.parse_args()
    run(cross_val=args.cross_val, run_shap=args.shap)
