"""Differential features: team1 minus team2 for key performance stats.

Differentials capture relative strength, which is more predictive than raw
values because it directly encodes which team outperforms the other.
"""

import pandas as pd

REQUIRED_COLS: list[str] = [
    "rating_team1", "rating_team2",
    "adr_team1", "adr_team2",
    "kast_team1", "kast_team2",
]

FEATURE_COLS: list[str] = [
    "rating_diff",
    "adr_diff",
    "kast_diff",
]


def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team-differential columns to `df`.

    Features added:
        rating_diff — mean player rating team1 − team2
                      (strongest single predictor of match outcome)
        adr_diff    — average damage per round differential
                      (measures which team causes more damage per round)
        kast_diff   — Kill/Assist/Survive/Trade % differential
                      (measures consistency: who more often contributes positively)
    """
    df = df.copy()
    df["rating_diff"] = df["rating_team1"] - df["rating_team2"]
    df["adr_diff"] = df["adr_team1"] - df["adr_team2"]
    df["kast_diff"] = df["kast_team1"] - df["kast_team2"]
    return df
