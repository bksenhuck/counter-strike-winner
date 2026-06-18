"""Prediction entry point.

Responsibilities (single):
  Load trained CSWinnerModel → load input data → run Pipeline.predict → return results.

The sklearn Pipeline (Imputer → Scaler → XGBoost) is loaded from disk as part
of CSWinnerModel.load(). Inference passes data through all Pipeline steps
automatically; no manual preprocessing is needed here.

All model logic lives in model.py. This script is a thin orchestrator.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.module.model.features import build_features
from src.module.model.model import CSWinnerModel
from utils.decorators import timer
from utils.file_utils import load_parquet
from utils.logger import get_logger

_logger = get_logger(__name__)


@timer
def run(
    input_path: str | Path,
    return_proba: bool = False,
    output_path: str | Path | None = None,
) -> tuple[pd.Series | pd.DataFrame, Pipeline]:
    """Load the model and predict winners for a parquet file of match features.

    Args:
        input_path: Path to a parquet file with raw match data.
        return_proba: Return probability DataFrame instead of label Series.
        output_path: Optional parquet path to save predictions.

    Returns:
        (predictions, sklearn Pipeline used for inference)
    """
    _logger.info("=== Prediction pipeline started ===")

    df = load_parquet(input_path)
    df = build_features(df)

    _logger.info("--- Loading Pipeline: Imputer → Scaler → XGBoost ---")
    model = CSWinnerModel.load()

    result = model.predict(df, return_proba=return_proba)

    if output_path is not None:
        out = result if isinstance(result, pd.DataFrame) else result.to_frame()
        out.to_parquet(output_path, index=False)
        _logger.info("Predictions saved -> %s", output_path)

    _logger.info("=== Prediction pipeline finished ===")
    return result, model.pipeline


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.parquet"
    predictions, _ = run(path)
    print(predictions)
