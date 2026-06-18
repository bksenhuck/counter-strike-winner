"""Team history features: rolling win rates and head-to-head records.

All features are computed retrospectively â€” each match only looks at results
*before* its own date to prevent data leakage. Rows with insufficient history
are filled with the dataset mean rather than dropped.
"""

from __future__ import annotations

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
    """Return per-match head-to-head stats (only matches before the current date)."""
    records = []
    df_sorted = df.sort_values("date").reset_index(drop=True)

    for idx, row in df_sorted.iterrows():
        prior = df_sorted.iloc[:idx]

        mask = (
            ((prior["team1"] == row["team1"]) & (prior["team2"] == row["team2"])) |
            ((prior["team1"] == row["team2"]) & (prior["team2"] == row["team1"]))
        )
        h2h = prior[mask]

        t1_wins = int(
            ((h2h["team1"] == row["team1"]) & (h2h["winner"] == 1)).sum() +
            ((h2h["team2"] == row["team1"]) & (h2h["winner"] == 0)).sum()
        )
        t2_wins = int(
            ((h2h["team1"] == row["team2"]) & (h2h["winner"] == 1)).sum() +
            ((h2h["team2"] == row["team2"]) & (h2h["winner"] == 0)).sum()
        )
        total = len(h2h)

        records.append(
            {
                "match_id": row["match_id"],
                "h2h_team1_wins": t1_wins,
                "h2h_team2_wins": t2_wins,
                "h2h_total": total,
                "h2h_team1_win_rate": t1_wins / total if total > 0 else float("nan"),
            }
        )

    return pd.DataFrame(records)


