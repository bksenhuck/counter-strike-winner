"""Momentum features: player and team performance trends over recent matches.

Captures whether a team is improving, declining, on a hot streak, or cooling
down â€” signals that static per-match stats cannot express on their own.

All features use shift(1) before rolling so the current match is excluded,
preventing any data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.data_processing import fill_feature_means

REQUIRED_COLS: list[str] = [
    "date", "team1", "team2",
    "rating_team1", "rating_team2",
    "adr_team1", "adr_team2",
    "winner",
]

FEATURE_COLS: list[str] = [
    "team1_rating_ma5",
    "team2_rating_ma5",
    "team1_adr_ma5",
    "team2_adr_ma5",
    "team1_rating_trend",
    "team2_rating_trend",
    "team1_on_win_streak",
    "team2_on_win_streak",
    "momentum_diff",
    "rating_ma_diff",
]

_WINDOW: int = 5
_STREAK_THRESHOLD: int = 3


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add player/team momentum columns.

    Features added
    --------------
    team1_rating_ma5    â€” team1 rolling mean rating over last 5 matches
    team2_rating_ma5    â€” team2 rolling mean rating over last 5 matches
    team1_adr_ma5       â€” team1 rolling mean ADR over last 5 matches
    team2_adr_ma5       â€” team2 rolling mean ADR over last 5 matches
    team1_rating_trend  â€” linear slope of team1 rating over last 5 matches
                          (positive = improving, negative = declining)
    team2_rating_trend  â€” same for team2
    team1_on_win_streak â€” 1 if team1 won their last 3+ matches, else 0
    team2_on_win_streak â€” 1 if team2 won their last 3+ matches, else 0
    momentum_diff       â€” team1_rating_trend âˆ’ team2_rating_trend
    rating_ma_diff      â€” team1_rating_ma5 âˆ’ team2_rating_ma5
    """
    df = df.sort_values("date").copy()

    df["team1_rating_ma5"] = _rolling_stat(df, "team1", "rating_team1")
    df["team2_rating_ma5"] = _rolling_stat(df, "team2", "rating_team2")
    df["team1_adr_ma5"] = _rolling_stat(df, "team1", "adr_team1")
    df["team2_adr_ma5"] = _rolling_stat(df, "team2", "adr_team2")

    df["team1_rating_trend"] = _rolling_trend(df, "team1", "rating_team1")
    df["team2_rating_trend"] = _rolling_trend(df, "team2", "rating_team2")

    df["team1_on_win_streak"] = _win_streak(df, team_col="team1", winner_val=1)
    df["team2_on_win_streak"] = _win_streak(df, team_col="team2", winner_val=0)

    df["momentum_diff"] = df["team1_rating_trend"] - df["team2_rating_trend"]
    df["rating_ma_diff"] = df["team1_rating_ma5"] - df["team2_rating_ma5"]

    fill_feature_means(df, FEATURE_COLS)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_stat(df: pd.DataFrame, team_col: str, stat_col: str) -> pd.Series:
    """Rolling mean of `stat_col` per team, excluding the current match."""
    return (
        df.groupby(df[team_col])[stat_col]
        .transform(lambda s: s.shift(1).rolling(_WINDOW, min_periods=1).mean())
    )


def _rolling_trend(df: pd.DataFrame, team_col: str, stat_col: str) -> pd.Series:
    """Linear regression slope of `stat_col` over the last _WINDOW matches per team.

    A positive slope means the stat is rising (team is improving).
    Returns NaN when fewer than 2 prior data points are available.
    """
    def _slope(series: pd.Series) -> pd.Series:
        result = pd.Series(index=series.index, dtype=float)
        shifted = series.shift(1)
        for i in range(len(series)):
            window = shifted.iloc[max(0, i - _WINDOW + 1): i + 1].dropna()
            if len(window) < 2:
                result.iloc[i] = float("nan")
            else:
                x = np.arange(len(window))
                result.iloc[i] = float(np.polyfit(x, window.values, 1)[0])
        return result

    return df.groupby(df[team_col])[stat_col].transform(_slope)


def _win_streak(df: pd.DataFrame, team_col: str, winner_val: int) -> pd.Series:
    """1 if team won their last _STREAK_THRESHOLD consecutive matches, else 0."""
    won = (df["winner"] == winner_val).astype(int)

    def _streak(s: pd.Series) -> pd.Series:
        result = pd.Series(0, index=s.index, dtype=int)
        s_shifted = s.shift(1)
        for i in range(len(s)):
            window = s_shifted.iloc[max(0, i - _STREAK_THRESHOLD + 1): i + 1].dropna()
            if len(window) >= _STREAK_THRESHOLD and window.sum() == _STREAK_THRESHOLD:
                result.iloc[i] = 1
        return result

    return won.groupby(df[team_col]).transform(_streak)


