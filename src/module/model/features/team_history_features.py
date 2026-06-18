"""Team history features: rolling win rates and head-to-head records.

All features are computed retrospectively â€” each match only looks at results
*before* its own date to prevent data leakage. Rows with insufficient history
are filled with the dataset mean rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.data_processing import fill_feature_means

REQUIRED_COLS: list[str] = ["match_id", "date", "team1", "team2", "winner"]

FEATURE_COLS: list[str] = [
    "team1_win_rate_last10",
    "team2_win_rate_last10",
    "team1_win_rate_last5",
    "team2_win_rate_last5",
    "h2h_team1_wins",
    "h2h_team2_wins",
    "h2h_total",
    "h2h_team1_win_rate",
    "form_advantage",
]


def add_team_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team win-rate and head-to-head columns.

    Features added
    --------------
    team1_win_rate_last10  â€” team1 overall win rate across last 10 matches
    team2_win_rate_last10  â€” team2 overall win rate across last 10 matches
    team1_win_rate_last5   â€” team1 overall win rate across last 5 matches
    team2_win_rate_last5   â€” team2 overall win rate across last 5 matches
    h2h_team1_wins         â€” historical wins of team1 vs team2 (before match date)
    h2h_team2_wins         â€” historical wins of team2 vs team1
    h2h_total              â€” total prior meetings between these teams
    h2h_team1_win_rate     â€” h2h_team1_wins / h2h_total  (NaN if no prior meetings)
    form_advantage         â€” team1_win_rate_last10 âˆ’ team2_win_rate_last10
    """
    df = df.sort_values("date").copy()

    team1_last10, team1_last5 = _rolling_win_rate(df, team_col="team1", winner_val=1)
    team2_last10, team2_last5 = _rolling_win_rate(df, team_col="team2", winner_val=0)

    df["team1_win_rate_last10"] = team1_last10
    df["team1_win_rate_last5"] = team1_last5
    df["team2_win_rate_last10"] = team2_last10
    df["team2_win_rate_last5"] = team2_last5

    h2h = _head_to_head(df)
    df = df.merge(h2h, on="match_id", how="left")

    df["form_advantage"] = df["team1_win_rate_last10"] - df["team2_win_rate_last10"]

    fill_feature_means(
        df,
        [
            "team1_win_rate_last10", "team2_win_rate_last10",
            "team1_win_rate_last5", "team2_win_rate_last5",
            "h2h_team1_wins", "h2h_team2_wins", "h2h_total",
            "h2h_team1_win_rate", "form_advantage",
        ],
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_win_rate(
    df: pd.DataFrame, team_col: str, winner_val: int
) -> tuple[pd.Series, pd.Series]:
    """Compute per-team rolling win rate (windows 10 and 5) using only prior matches."""
    won = (df["winner"] == winner_val).astype(float)

    last10 = (
        won.groupby(df[team_col])
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    )
    last5 = (
        won.groupby(df[team_col])
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    return last10, last5


def _head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-match head-to-head stats using vectorized cumsum (O(N log N))."""
    base = df[["date", "match_id", "team1", "team2", "winner"]].sort_values("date")

    # Expand to two rows per match: one from each team's perspective
    t1 = base.assign(
        team=base["team1"], opponent=base["team2"],
        won=(base["winner"] == 1).astype(int), _src=1,
    )
    t2 = base.assign(
        team=base["team2"], opponent=base["team1"],
        won=(base["winner"] == 0).astype(int), _src=2,
    )

    edges = (
        pd.concat(
            [t1[["date", "match_id", "team", "opponent", "won", "_src"]],
             t2[["date", "match_id", "team", "opponent", "won", "_src"]]],
            ignore_index=True,
        )
        .sort_values("date")
    )

    # cumsum() - current_won = wins BEFORE this match (excludes current row).
    # cumcount() = 0-indexed rank within group = number of prior meetings.
    grp = edges.groupby(["team", "opponent"])
    edges["h2h_wins_pre"] = (grp["won"].cumsum() - edges["won"]).values
    edges["h2h_total_pre"] = grp.cumcount().values

    t1_h2h = (
        edges.loc[edges["_src"] == 1, ["match_id", "h2h_wins_pre", "h2h_total_pre"]]
        .rename(columns={"h2h_wins_pre": "h2h_team1_wins", "h2h_total_pre": "h2h_total"})
    )
    t2_h2h = (
        edges.loc[edges["_src"] == 2, ["match_id", "h2h_wins_pre"]]
        .rename(columns={"h2h_wins_pre": "h2h_team2_wins"})
    )

    result = t1_h2h.merge(t2_h2h, on="match_id")
    result["h2h_team1_win_rate"] = (
        result["h2h_team1_wins"] / result["h2h_total"].replace(0, float("nan"))
    )
    return result[["match_id", "h2h_team1_wins", "h2h_team2_wins", "h2h_total", "h2h_team1_win_rate"]]


