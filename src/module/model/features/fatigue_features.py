"""Fatigue and rest features: game load and recovery time per team.

A team that played 5 matches in 7 days is likely more fatigued than one that
had 2 weeks of rest. These features provide the model with that context.
All are computed from prior matches only â€” safe for pre-match prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.data_processing import fill_feature_means

REQUIRED_COLS: list[str] = ["date", "team1", "team2"]

FEATURE_COLS: list[str] = [
    "team1_days_rest",
    "team2_days_rest",
    "team1_matches_7d",
    "team2_matches_7d",
    "team1_matches_14d",
    "team2_matches_14d",
    "team1_matches_30d",
    "team2_matches_30d",
    "rest_advantage",
    "fatigue_advantage",
]


def add_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest days and recent game-load features for both teams."""
    df = df.sort_values("date").copy()

    df["team1_days_rest"] = _days_since_last_match(df, "team1")
    df["team2_days_rest"] = _days_since_last_match(df, "team2")
    df["team1_matches_7d"] = _matches_in_window(df, "team1", 7)
    df["team2_matches_7d"] = _matches_in_window(df, "team2", 7)
    df["team1_matches_14d"] = _matches_in_window(df, "team1", 14)
    df["team2_matches_14d"] = _matches_in_window(df, "team2", 14)
    df["team1_matches_30d"] = _matches_in_window(df, "team1", 30)
    df["team2_matches_30d"] = _matches_in_window(df, "team2", 30)

    # Positive = team1 more rested; negative = team2 more rested
    df["rest_advantage"] = df["team1_days_rest"] - df["team2_days_rest"]
    # Positive = team2 more fatigued (played more recently)
    df["fatigue_advantage"] = df["team2_matches_30d"] - df["team1_matches_30d"]

    fill_feature_means(df, FEATURE_COLS)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_since_last_match(df: pd.DataFrame, team_col: str) -> pd.Series:
    """Days between the current match date and the team's most recent prior match."""
    # df is already sorted by date; diff() within each group gives days since last match.
    return df.groupby(team_col)["date"].diff().dt.days


def _matches_in_window(df: pd.DataFrame, team_col: str, days: int) -> pd.Series:
    """Count of prior matches played by the team in the last `days` days.

    Uses searchsorted for O(N log N) total instead of O(N²) iterrows per team.
    Same semantics as the original: count matches where date in [current-days, current).
    """
    result = pd.Series(0.0, index=df.index)
    td = np.timedelta64(days, "D")

    for team, group_idx in df.groupby(team_col).groups.items():
        sub = df.loc[group_idx].sort_values("date")
        dates = sub["date"].values.astype("datetime64[D]")

        cutoffs = dates - td
        # lefts[i] = first position with date >= cutoff  (window left bound)
        # rights[i] = first position with date >= current (= count of dates strictly before current)
        lefts = np.searchsorted(dates, cutoffs, side="left")
        rights = np.searchsorted(dates, dates, side="left")
        result.loc[sub.index] = (rights - lefts).astype(float)

    return result


