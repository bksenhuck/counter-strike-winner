"""Player-level features: per-team aggregated individual stats."""

import numpy as np
import pandas as pd

REQUIRED_COLS: list[str] = [
    "kills_team1", "kills_team2",
    "deaths_team1", "deaths_team2",
    "adr_team1", "adr_team2",
    "kast_team1", "kast_team2",
    "rating_team1", "rating_team2",
]

FEATURE_COLS: list[str] = REQUIRED_COLS + [
    "kill_death_ratio_team1",
    "kill_death_ratio_team2",
]


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add player-level derived columns to `df`.

    Features added:
        kill_death_ratio_team1 — kills_team1 / deaths_team1  (team1 efficiency)
        kill_death_ratio_team2 — kills_team2 / deaths_team2  (team2 efficiency)

    The raw aggregates (kills, deaths, adr, kast, rating per team) are passthrough
    base columns already present from the data pipeline; they are listed in
    FEATURE_COLS so downstream code knows they belong to this theme.
    """
    df = df.copy()
    df["kill_death_ratio_team1"] = df["kills_team1"] / df["deaths_team1"].replace(0, np.nan)
    df["kill_death_ratio_team2"] = df["kills_team2"] / df["deaths_team2"].replace(0, np.nan)
    return df
