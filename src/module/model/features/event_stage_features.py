"""Event and stage context features.

High-pressure matches (finals, semifinals) behave differently from group stage
matches. These features encode the stakes of the match, which helps the model
calibrate confidence appropriately. Available before the match starts.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLS: list[str] = []  # stage / event are optional; handled gracefully

FEATURE_COLS: list[str] = [
    "stage_encoded",
    "is_playoff",
    "is_final",
    "pressure_score",
]

# Maps stage name fragments to encoded level (higher = more important)
_STAGE_MAP: dict[str, int] = {
    "grand final": 4,
    "final": 3,
    "semifinal": 2,
    "semi-final": 2,
    "quarterfinal": 1,
    "quarter-final": 1,
    "playoff": 1,
    "elimination": 1,
    "group": 0,
    "qualifier": 0,
}


def add_event_stage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode match stage and derive pressure/playoff flags."""
    df = df.copy()

    if "stage" in df.columns:
        df["stage_encoded"] = df["stage"].apply(_encode_stage).astype(float)
    else:
        df["stage_encoded"] = 0.0

    df["is_playoff"] = (df["stage_encoded"] >= 1).astype(float)
    df["is_final"] = (df["stage_encoded"] >= 3).astype(float)
    df["pressure_score"] = df["stage_encoded"] / 4.0  # normalised [0, 1]

    return df


def _encode_stage(stage: object) -> int:
    if not isinstance(stage, str):
        return 0
    stage_lower = stage.lower()
    for fragment, level in _STAGE_MAP.items():
        if fragment in stage_lower:
            return level
    return 0
