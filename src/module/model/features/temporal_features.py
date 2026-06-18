"""Temporal context features derived from the match date and time.

Teams may perform differently depending on the day of week, season or time
of day (online vs LAN, jet-lag effects, meta shifts by season). All values
are derivable from the date column — fully available before the match.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLS: list[str] = ["date"]

FEATURE_COLS: list[str] = [
    "day_of_week",
    "month",
    "is_weekend",
    "season",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived temporal context columns."""
    df = df.copy()
    dt = pd.to_datetime(df["date"], errors="coerce")

    df["day_of_week"] = dt.dt.dayofweek.astype(float)       # 0=Mon … 6=Sun
    df["month"] = dt.dt.month.astype(float)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)  # Sat or Sun

    # Meteorological seasons (Northern Hemisphere)
    # Winter=1, Spring=2, Summer=3, Autumn=4
    df["season"] = dt.dt.month.map(
        {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    ).astype(float)

    if "match_time" in df.columns:
        # Try to extract hour (format may vary)
        hour = pd.to_datetime(df["match_time"], errors="coerce", format="%H:%M").dt.hour
        if hour.notna().any():
            df["match_hour"] = hour.astype(float)
            FEATURE_COLS.append("match_hour")  # only add when data exists

    return df
