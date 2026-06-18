"""Tests for src/module/data.py."""

import pandas as pd
import pytest

from src.module.model.data import _aggregate_maps, _aggregate_player_stats, _clean_matches


@pytest.fixture()
def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "date": ["2020-01-01", "2020-01-02", "bad-date"],
            "team1": ["NaVi", "Astralis", None],
            "team2": ["Faze", "G2", "Liquid"],
            "score_team1": [2, 1, 1],
            "score_team2": [0, 2, 1],
            "event": ["A", "B", "C"],
            "stage": ["GS", "GS", "GS"],
            "format": ["bo3", "bo3", "bo1"],
            "match_url": ["u1", "u2", "u3"],
            "team1_lineup": ["p1,p2", "p3,p4", "p5,p6"],
            "team2_lineup": ["p7,p8", "p9,p10", "p11,p12"],
            "year": [2020, 2020, 2020],
            "month": [1, 1, 1],
            "match_time": ["12:00", "13:00", "14:00"],
        }
    )


@pytest.fixture()
def sample_maps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1, 1, 2, 2],
            "map_order": [1, 2, 1, 2],
            "map_name": ["Dust2", "Inferno", "Mirage", "Overpass"],
            "score_team1": [16, 14, 8, 16],
            "score_team2": [10, 16, 16, 12],
        }
    )


@pytest.fixture()
def sample_player_stats() -> pd.DataFrame:
    rows = []
    for match_id in [1, 2]:
        for team in [1, 2]:
            for i in range(5):
                rows.append(
                    {
                        "match_id": match_id,
                        "map_order": 1,
                        "team": team,
                        "side": "CT",
                        "player_name": f"player{i}",
                        "kills": 20 + i,
                        "deaths": 15,
                        "adr": 80.0 + i,
                        "kast": 70.0,
                        "rating": 1.0 + i * 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_clean_matches_removes_draws_and_nulls(sample_matches):
    result = _clean_matches(sample_matches)
    assert result["team1"].notna().all()
    assert (result["score_team1"] != result["score_team2"]).all()


def test_clean_matches_parses_dates(sample_matches):
    result = _clean_matches(sample_matches)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_aggregate_maps_returns_correct_columns(sample_maps):
    result = _aggregate_maps(sample_maps)
    assert {"match_id", "maps_won_team1", "maps_won_team2", "total_maps"} <= set(result.columns)


def test_aggregate_maps_counts(sample_maps):
    result = _aggregate_maps(sample_maps).set_index("match_id")
    assert result.loc[1, "maps_won_team1"] == 1
    assert result.loc[1, "maps_won_team2"] == 1


def test_aggregate_player_stats_wide_format(sample_player_stats):
    result = _aggregate_player_stats(sample_player_stats)
    assert "rating_team1" in result.columns
    assert "rating_team2" in result.columns
    assert len(result) == 2
