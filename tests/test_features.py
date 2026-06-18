"""Tests for src/module/features.py."""

import numpy as np
import pandas as pd
import pytest

from src.module.model.features import FEATURE_COLS, TARGET_COL, build_features, get_feature_matrix


@pytest.fixture()
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "maps_won_team1": [2, 1],
            "maps_won_team2": [1, 2],
            "total_maps": [3, 3],
            "kills_team1": [100.0, 80.0],
            "kills_team2": [80.0, 100.0],
            "deaths_team1": [80.0, 100.0],
            "deaths_team2": [100.0, 80.0],
            "adr_team1": [85.0, 70.0],
            "adr_team2": [70.0, 85.0],
            "kast_team1": [72.0, 65.0],
            "kast_team2": [65.0, 72.0],
            "rating_team1": [1.2, 0.9],
            "rating_team2": [0.9, 1.2],
            "format_maps": [3, 3],
            "winner": [1, 0],
        }
    )


def test_build_features_adds_diff_columns(base_df):
    result = build_features(base_df)
    assert "rating_diff" in result.columns
    assert "adr_diff" in result.columns
    assert "kast_diff" in result.columns


def test_build_features_rating_diff_sign(base_df):
    result = build_features(base_df)
    assert result["rating_diff"].iloc[0] > 0
    assert result["rating_diff"].iloc[1] < 0


def test_build_features_map_win_rate(base_df):
    result = build_features(base_df)
    assert pytest.approx(result["map_win_rate"].iloc[0], rel=1e-4) == 2 / 3


def test_build_features_kdr(base_df):
    result = build_features(base_df)
    assert pytest.approx(result["kill_death_ratio_team1"].iloc[0], rel=1e-4) == 100 / 80


def test_missing_base_col_raises(base_df):
    df_bad = base_df.drop(columns=["rating_team1"])
    with pytest.raises(ValueError, match="missing columns"):
        build_features(df_bad)


def test_get_feature_matrix_shape(base_df):
    enriched = build_features(base_df)
    X, y = get_feature_matrix(enriched)
    assert X.shape[1] == len(FEATURE_COLS)
    assert len(y) == len(X)
    assert y.name == TARGET_COL


def test_get_feature_matrix_drops_nan_rows(base_df):
    enriched = build_features(base_df)
    enriched.loc[0, "rating_team1"] = np.nan
    X, y = get_feature_matrix(enriched)
    assert len(X) < len(base_df)
