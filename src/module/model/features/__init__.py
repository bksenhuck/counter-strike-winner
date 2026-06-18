"""Feature pipeline orchestrator — pre-match safe features only.

All features in FEATURE_COLS are computable BEFORE a match starts using only
historical data. No current-match statistics (kills, map scores, live ratings)
are included, so the model can genuinely predict future matches.

Themed modules
--------------
format_features       — match format context (bo1→1, bo3→3, bo5→5)
historical_features   — ELO, all-time win rates, streaks, resilience
team_history_features — rolling win rates and head-to-head records
momentum_features     — rating trends and win-streak indicators
form_features         — multi-window form (last-3, last-20, 30/60/90 days)
fatigue_features      — rest days and recent game load
event_stage_features  — stage encoding, pressure score
temporal_features     — day of week, month, season
consistency_features  — star dependency, fragility, firepower
round_features        — dominance score from round margins

Modules below are still called during build_features() to add columns that
the rolling/trend modules need as raw inputs, but their outputs are NOT
included in FEATURE_COLS because they reflect current-match stats:
  map_features, player_features, differential_features
"""

import time
from pathlib import Path
from typing import Any

import pandas as pd

from conf.settings import FEATURES_FILE, PROCESSED_DIR
from src.module.model.features.consistency_features import (
    FEATURE_COLS as _CONSISTENCY_COLS,
    REQUIRED_COLS as _CONSISTENCY_REQUIRED,
    add_consistency_features,
)
from src.module.model.features.differential_features import add_differential_features
from src.module.model.features.event_stage_features import (
    FEATURE_COLS as _EVENT_COLS,
    REQUIRED_COLS as _EVENT_REQUIRED,
    add_event_stage_features,
)
from src.module.model.features.fatigue_features import (
    FEATURE_COLS as _FATIGUE_COLS,
    REQUIRED_COLS as _FATIGUE_REQUIRED,
    add_fatigue_features,
)
from src.module.model.features.form_features import (
    FEATURE_COLS as _FORM_COLS,
    REQUIRED_COLS as _FORM_REQUIRED,
    add_form_features,
)
from src.module.model.features.format_features import (
    FEATURE_COLS as _FORMAT_COLS,
    REQUIRED_COLS as _FORMAT_REQUIRED,
    add_format_features,
)
from src.module.model.features.historical_features import (
    FEATURE_COLS as _HIST_COLS,
    REQUIRED_COLS as _HIST_REQUIRED,
    add_historical_features,
)
from src.module.model.features.map_features import add_map_features
from src.module.model.features.momentum_features import (
    FEATURE_COLS as _MOMENTUM_COLS,
    REQUIRED_COLS as _MOMENTUM_REQUIRED,
    add_momentum_features,
)
from src.module.model.features.player_features import add_player_features
from src.module.model.features.round_features import (
    FEATURE_COLS as _ROUND_COLS,
    REQUIRED_COLS as _ROUND_REQUIRED,
    add_round_features,
)
from src.module.model.features.team_history_features import (
    FEATURE_COLS as _HISTORY_COLS,
    REQUIRED_COLS as _HISTORY_REQUIRED,
    add_team_history_features,
)
from src.module.model.features.temporal_features import (
    FEATURE_COLS as _TEMPORAL_COLS,
    REQUIRED_COLS as _TEMPORAL_REQUIRED,
    add_temporal_features,
)
from src.module.model.settings import TARGET_COL
from utils.decorators import log_call, timer, validate_dataframe
from utils.file_utils import load_parquet, save_parquet
from utils.logger import get_logger

_logger = get_logger(__name__)

# Minimum columns required before feature engineering starts.
# Includes only what the pre-match modules strictly need.
BASE_NUMERIC_FEATURES: list[str] = list(
    dict.fromkeys(
        _HIST_REQUIRED
        + _HISTORY_REQUIRED
        + _MOMENTUM_REQUIRED
        + _FORM_REQUIRED
        + _FATIGUE_REQUIRED
        + _EVENT_REQUIRED
        + _TEMPORAL_REQUIRED
        + _FORMAT_REQUIRED
        + _CONSISTENCY_REQUIRED
        + _ROUND_REQUIRED
    )
)

# Canonical ordered feature list — PRE-MATCH SAFE ONLY.
# In-match features (kills, map scores, current ratings) are intentionally
# excluded; they are unavailable before the match starts.
FEATURE_COLS: list[str] = list(
    dict.fromkeys(
        _FORMAT_COLS
        + _HIST_COLS
        + _HISTORY_COLS
        + _MOMENTUM_COLS
        + _FORM_COLS
        + _FATIGUE_COLS
        + _EVENT_COLS
        + _TEMPORAL_COLS
        # _CONSISTENCY_COLS excluded: uses current-match player stats (data leakage)
        + _ROUND_COLS  # only avg_round_diff_ma5 (shift(1) historical)
    )
)

# Ordered list of (function, module_name, output_description).
# Entries marked "[helper]" produce raw columns for rolling modules but are
# NOT included in FEATURE_COLS (in-match stats, unavailable pre-match).
_PIPELINE_STEPS: list[tuple] = [
    (add_map_features,          "map_features",          "[helper] map_win_rate, maps_advantage"),
    (add_player_features,       "player_features",       "[helper] kill_death_ratio_team1/2"),
    (add_differential_features, "differential_features", "[helper] rating_diff, adr_diff, kast_diff"),
    (add_format_features,       "format_features",       "format_maps"),
    (add_historical_features,   "historical_features",   "ELO, win rates, streaks, resilience"),
    (add_team_history_features, "team_history_features", "win_rate_last5/10, h2h, form_advantage"),
    (add_momentum_features,     "momentum_features",     "rating_ma5, rating_trend, win_streak"),
    (add_form_features,         "form_features",         "last3/20, 30d/60d/90d win rates, hot/cold"),
    (add_fatigue_features,      "fatigue_features",      "days_rest, matches_7d/14d/30d"),
    (add_event_stage_features,  "event_stage_features",  "stage_encoded, pressure_score"),
    (add_temporal_features,     "temporal_features",     "day_of_week, month, season"),
    (add_consistency_features,  "consistency_features",  "[helper] star_dependency, fragility, firepower (current-match raw)"),
    (add_round_features,        "round_features",        "dominance_score, avg_round_diff_ma5"),
]


@timer
@log_call
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature modules in sequence and return the enriched DataFrame."""
    total = len(_PIPELINE_STEPS)
    _logger.info("=== Feature Engineering — %d modules | %d rows ===", total, len(df))

    for idx, (fn, name, desc) in enumerate(_PIPELINE_STEPS, 1):
        _logger.info("  [%2d/%d] %-25s  %s", idx, total, name, desc)
        t0 = time.perf_counter()
        df = fn(df)
        elapsed = time.perf_counter() - t0
        _logger.info("         done in %.1fs  →  %d cols total", elapsed, df.shape[1])

    _logger.info(
        "=== Feature Engineering complete: %d rows × %d cols | %d feature cols ===",
        len(df), df.shape[1], len(FEATURE_COLS),
    )
    return df


@log_call
def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) dropping rows with NaN in any feature or target column."""
    required = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    df = df.dropna(subset=required)
    X = df[[c for c in FEATURE_COLS if c in df.columns]]
    y = df[TARGET_COL]
    _logger.info(
        "Feature matrix: X=%s  y=%s  (pos_rate=%.2f%%)",
        X.shape,
        y.shape,
        y.mean() * 100,
    )
    return X, y


def build_team_stats_lookup(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Extract each team's most recent pre-match feature vector.

    Processes rows chronologically. For each match, both teams' feature values
    are stored under their team name, overwriting older values. The final dict
    holds the state of every team as of their last known match.

    Returns:
        dict: {team_name: {stat_name_without_position_prefix: value}}

    Usage in predict_match()::

        team_stats = build_team_stats_lookup(df)
        model.predict_match("NaVi", "Astralis", "bo3", team_stats=team_stats)
    """
    df = df.sort_values("date")
    lookup: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        for pos in (1, 2):
            team = row.get(f"team{pos}")
            if not team or not isinstance(team, str):
                continue
            prefix = f"team{pos}_"
            stats = {
                col[len(prefix):]: val
                for col, val in row.items()
                if col.startswith(prefix) and not pd.isna(val)
            }
            lookup[team] = stats

    return lookup


def load_features(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return load_parquet(processed_dir / FEATURES_FILE)


def save_features(df: pd.DataFrame, processed_dir: Path = PROCESSED_DIR) -> None:
    save_parquet(df, processed_dir / FEATURES_FILE)


__all__ = [
    "build_features",
    "build_team_stats_lookup",
    "get_feature_matrix",
    "load_features",
    "save_features",
    "FEATURE_COLS",
    "BASE_NUMERIC_FEATURES",
    "TARGET_COL",
]
