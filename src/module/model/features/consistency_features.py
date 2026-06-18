"""Consistency, star power and firepower features.

These features characterise HOW a team wins, not just whether they win:

* Star Dependency  â€” how much the team relies on one player
* Fragility Index  â€” whether two players carry the rest of the squad
* Consistency      â€” how stable individual performance is (low std = consistent)
* Firepower Score  â€” composite offensive power index

Requires max/std player stats produced by data.py:_aggregate_player_stats.
If those columns are absent the features fall back gracefully to NaN.
"""

from __future__ import annotations

import pandas as pd

from utils.data_processing import fill_feature_means

REQUIRED_COLS: list[str] = []  # all columns handled gracefully

# FEATURE_COLS is intentionally empty: all computed values use current-match
# player stats (rating_max, rating_std, etc.) which are unavailable before the
# match starts → data leakage if included in the model.
# The columns are still computed so future modules can roll them with shift(1).
FEATURE_COLS: list[str] = []

_FIREPOWER_W_RATING: float = 0.5
_FIREPOWER_W_ADR: float = 0.3
_FIREPOWER_W_KILLS: float = 0.2
_CONSISTENCY_EPS: float = 0.01  # avoid division by zero


def add_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add star power, fragility, consistency and firepower features."""
    df = df.copy()

    for pos in [1, 2]:
        t = f"team{pos}"
        rating_mean = df.get(f"rating_{t}", pd.Series(float("nan"), index=df.index))
        rating_max = df.get(f"rating_max_{t}", pd.Series(float("nan"), index=df.index))
        rating_std = df.get(f"rating_std_{t}", pd.Series(float("nan"), index=df.index))
        adr_mean = df.get(f"adr_{t}", pd.Series(float("nan"), index=df.index))
        kills_mean = df.get(f"kills_{t}", pd.Series(float("nan"), index=df.index))

        # Star dependency: best_player_rating / team_avg_rating
        df[f"{t}_star_dependency"] = rating_max / rating_mean.replace(0, float("nan"))

        # Fragility: top-2-player dominance proxy using max and mean
        # top2_avg â‰ˆ (max + mean*5 - min*1) / 2 â€” approximate since individual
        # player rows aren't available; use max and mean as a proxy.
        df[f"{t}_fragility"] = rating_max / (rating_mean + _CONSISTENCY_EPS)

        # Consistency: higher = more consistent (lower within-team spread)
        df[f"{t}_consistency"] = 1.0 / (rating_std + _CONSISTENCY_EPS)

        # Firepower: composite offensive score
        df[f"{t}_firepower"] = (
            _FIREPOWER_W_RATING * rating_mean
            + _FIREPOWER_W_ADR * adr_mean / 100.0  # normalise ADR ~70-100
            + _FIREPOWER_W_KILLS * kills_mean / 20.0  # normalise kills ~15-25
        )

    df["firepower_diff"] = df["team1_firepower"] - df["team2_firepower"]
    df["star_dependency_diff"] = df["team1_star_dependency"] - df["team2_star_dependency"]
    df["consistency_diff"] = df["team1_consistency"] - df["team2_consistency"]

    fill_feature_means(df, FEATURE_COLS)
    return df


