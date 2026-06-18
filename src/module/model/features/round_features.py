"""Round-level dominance features.

Captures not just *whether* a team wins but *how convincingly* they win.
A team that wins 16-3 is more dominant than one that wins 16-14.
These features require the round-score aggregations from data.py.
"""

from __future__ import annotations

import pandas as pd

from utils.data_processing import fill_feature_means

REQUIRED_COLS: list[str] = []  # avg_round_diff is optional

FEATURE_COLS: list[str] = [
    # dominance_score excluded: it IS the current match's avg_round_diff
    # (direct leakage — the model would see the score of the match it's predicting).
    "avg_round_diff_ma5",  # safe: rolling mean of PAST 5 matches with shift(1)
]


def add_round_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add dominance score and rolling round-differential features."""
    df = df.copy()

    if "avg_round_diff" not in df.columns:
        for col in FEATURE_COLS:
            df[col] = float("nan")
        return df

    # dominance_score = current match avg_round_diff (team1 perspective)
    df["dominance_score"] = df["avg_round_diff"].fillna(0.0)

    # Rolling mean of avg_round_diff per team1 (prior 5 matches)
    df["avg_round_diff_ma5"] = (
        df["avg_round_diff"]
        .groupby(df["team1"])
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    fill_feature_means(df, FEATURE_COLS)
    return df


