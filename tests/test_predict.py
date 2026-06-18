"""Tests for CSWinnerModel.predict() (mocks the pipeline to avoid disk I/O)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.module.model.features import FEATURE_COLS
from src.module.model.model import CSWinnerModel


@pytest.fixture()
def feature_df() -> pd.DataFrame:
    n = 5
    return pd.DataFrame(
        {
            "maps_won_team1": [2] * n,
            "maps_won_team2": [1] * n,
            "total_maps": [3] * n,
            "kills_team1": [100.0] * n,
            "kills_team2": [80.0] * n,
            "deaths_team1": [80.0] * n,
            "deaths_team2": [100.0] * n,
            "adr_team1": [85.0] * n,
            "adr_team2": [70.0] * n,
            "kast_team1": [72.0] * n,
            "kast_team2": [65.0] * n,
            "rating_team1": [1.2] * n,
            "rating_team2": [0.9] * n,
            "rating_diff": [0.3] * n,
            "adr_diff": [15.0] * n,
            "kast_diff": [7.0] * n,
            "kill_death_ratio_team1": [1.25] * n,
            "kill_death_ratio_team2": [0.80] * n,
            "map_win_rate": [2 / 3] * n,
            "maps_advantage": [1] * n,
            "format_maps": [3] * n,
        }
    )


@pytest.fixture()
def fitted_model() -> CSWinnerModel:
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([1, 1, 0, 1, 0])
    mock_pipeline.predict_proba.return_value = np.array(
        [[0.3, 0.7], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9], [0.55, 0.45]]
    )
    model = CSWinnerModel()
    model.pipeline = mock_pipeline
    return model


def test_predict_returns_series(feature_df, fitted_model):
    result = fitted_model.predict(feature_df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(feature_df)
    assert result.name == "predicted_winner"


def test_predict_returns_proba_dataframe(feature_df, fitted_model):
    result = fitted_model.predict(feature_df, return_proba=True)
    assert isinstance(result, pd.DataFrame)
    assert "prob_team1_wins" in result.columns
    assert "prob_team2_wins" in result.columns


def test_predict_raises_on_missing_columns(feature_df, fitted_model):
    bad_df = feature_df.drop(columns=["rating_diff"])
    with pytest.raises(ValueError, match="missing columns"):
        fitted_model.predict(bad_df)


def test_predict_raises_when_not_fitted():
    model = CSWinnerModel()
    df = pd.DataFrame(columns=FEATURE_COLS)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(df)


def test_predict_from_raw_returns_winner_key(fitted_model):
    raw = {col: 1.0 for col in FEATURE_COLS}
    with patch("src.module.model.model.build_features", return_value=pd.DataFrame([raw])):
        result = fitted_model.predict_from_raw(raw)
    assert "predicted_winner" in result
    assert result["predicted_winner"] in ("team1", "team2")
    assert abs(result["prob_team1_wins"] + result["prob_team2_wins"] - 1.0) < 1e-6
