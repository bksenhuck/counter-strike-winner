"""Format features: match format context (bo1/bo3/bo5 → numeric).

Encodes the series format string into a numeric count of maximum maps,
giving the model context on how much variance is expected in the outcome.
A bo1 series is far more volatile than a bo5.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLS: list[str] = []  # handles missing format column gracefully
FEATURE_COLS: list[str] = ["format_maps"]

_FORMAT_MAP: dict[str, int] = {"bo1": 1, "bo3": 3, "bo5": 5}


def add_format_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the match format string to a numeric map count.

    Mapping: bo1 → 1, bo3 → 3, bo5 → 5.
    Falls back to 1 if the format column is absent or contains an unknown value.
    If format_maps already exists the function is a no-op.
    """
    if "format_maps" in df.columns:
        return df
    df = df.copy()
    if "format" not in df.columns:
        df["format_maps"] = 1
        return df
    df["format_maps"] = (
        df["format"].str.lower().map(_FORMAT_MAP).fillna(1).astype(int)
    )
    return df
