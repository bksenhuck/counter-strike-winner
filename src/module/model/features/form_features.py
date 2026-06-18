"""Recent form features using multiple time windows.

Captures short-term momentum with both game-count windows (last 3, last 20)
and calendar-day windows (last 30 / 60 / 90 days). All windows exclude the
current match via shift(1) or strict '<' date comparisons.

Hot/Cold team indicator measures whether a team is trending up or down by
comparing their short-term rating MA to their long-term rating MA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.data_processing import fill_feature_means

from conf.settings import FORM_DAYS_LONG, FORM_DAYS_MID, FORM_DAYS_SHORT, FORM_WINDOW_SHORT

REQUIRED_COLS: list[str] = ["date", "team1", "team2", "winner"]

FEATURE_COLS: list[str] = [
    "team1_win_rate_last3",
    "team2_win_rate_last3",
    "team1_win_rate_last20",
    "team2_win_rate_last20",
    "team1_win_rate_30d",
    "team2_win_rate_30d",
    "team1_win_rate_60d",
    "team2_win_rate_60d",
    "team1_win_rate_90d",
    "team2_win_rate_90d",
    "team1_rating_ma20",
    "team2_rating_ma20",
    "team1_hot_cold",
    "team2_hot_cold",
    "form3_diff",
    "form_30d_diff",
]


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-window form and hot/cold team indicators."""
    df = df.sort_values("date").copy()

    # Game-count windows
    df["team1_win_rate_last3"] = _rolling_win_rate(df, "team1", 1, 3)
    df["team2_win_rate_last3"] = _rolling_win_rate(df, "team2", 0, 3)
    df["team1_win_rate_last20"] = _rolling_win_rate(df, "team1", 1, 20)
    df["team2_win_rate_last20"] = _rolling_win_rate(df, "team2", 0, 20)

    # Calendar-day windows
    df["team1_win_rate_30d"] = _rolling_win_rate_days(df, "team1", 1, FORM_DAYS_SHORT)
    df["team2_win_rate_30d"] = _rolling_win_rate_days(df, "team2", 0, FORM_DAYS_SHORT)
    df["team1_win_rate_60d"] = _rolling_win_rate_days(df, "team1", 1, FORM_DAYS_MID)
    df["team2_win_rate_60d"] = _rolling_win_rate_days(df, "team2", 0, FORM_DAYS_MID)
    df["team1_win_rate_90d"] = _rolling_win_rate_days(df, "team1", 1, FORM_DAYS_LONG)
    df["team2_win_rate_90d"] = _rolling_win_rate_days(df, "team2", 0, FORM_DAYS_LONG)

    # Long-term rating MA (20 games) â€” needed for hot/cold
    if "rating_team1" in df.columns and "rating_team2" in df.columns:
        df["team1_rating_ma20"] = _rolling_rating(df, "team1", "rating_team1", 20)
        df["team2_rating_ma20"] = _rolling_rating(df, "team2", "rating_team2", 20)
        # Hot/cold = short MA âˆ’ long MA; positive = improving, negative = declining
        team1_ma5 = df["team1_rating_ma5"] if "team1_rating_ma5" in df.columns else _rolling_rating(df, "team1", "rating_team1", 5)
        team2_ma5 = df["team2_rating_ma5"] if "team2_rating_ma5" in df.columns else _rolling_rating(df, "team2", "rating_team2", 5)
        df["team1_hot_cold"] = team1_ma5 - df["team1_rating_ma20"]
        df["team2_hot_cold"] = team2_ma5 - df["team2_rating_ma20"]
    else:
        for col in ["team1_rating_ma20", "team2_rating_ma20", "team1_hot_cold", "team2_hot_cold"]:
            df[col] = float("nan")

    # Differential features
    df["form3_diff"] = df["team1_win_rate_last3"] - df["team2_win_rate_last3"]
    df["form_30d_diff"] = df["team1_win_rate_30d"] - df["team2_win_rate_30d"]

    fill_feature_means(df, FEATURE_COLS)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_win_rate(
    df: pd.DataFrame, team_col: str, winner_val: int, window: int
) -> pd.Series:
    """Rolling win rate over last `window` matches per team, excluding current."""
    won = (df["winner"] == winner_val).astype(float)
    return (
        won.groupby(df[team_col])
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )


def _rolling_win_rate_days(
    df: pd.DataFrame, team_col: str, winner_val: int, days: int
) -> pd.Series:
    """Win rate in the last `days` calendar days per team, excluding current match.

    Uses searchsorted + prefix sums for O(N log N) total instead of O(N²) iterrows.
    Same semantics as the original: window is [current-days, current).
    """
    won = (df["winner"] == winner_val).astype(float)
    result = pd.Series(float("nan"), index=df.index)
    td = np.timedelta64(days, "D")

    for team, group_idx in df.groupby(team_col).groups.items():
        sub = df.loc[group_idx].sort_values("date")
        dates = sub["date"].values.astype("datetime64[D]")
        w = won.loc[sub.index].values
        n = len(dates)
        if n == 0:
            continue

        cutoffs = dates - td
        lefts = np.searchsorted(dates, cutoffs, side="left")
        rights = np.searchsorted(dates, dates, side="left")  # count strictly before

        # Prefix-sum array for O(1) range win sums
        cumw = np.empty(n + 1)
        cumw[0] = 0.0
        np.cumsum(w, out=cumw[1:])

        counts = rights - lefts
        win_sums = cumw[rights] - cumw[lefts]
        result.loc[sub.index] = np.where(counts > 0, win_sums / counts, float("nan"))

    return result


def _rolling_rating(
    df: pd.DataFrame, team_col: str, rating_col: str, window: int
) -> pd.Series:
    return (
        df[rating_col]
        .groupby(df[team_col])
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )


