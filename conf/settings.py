"""General project configuration — infrastructure, paths, data splits.

All values here are environment/deployment concerns that apply across the
entire project (not model-specific). Override via environment variables or
a local conf/settings_local.py that is git-ignored.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

RAW_DIR: Path = BASE_DIR / os.getenv("RAW_DIR", "data/datasets/results")
PROCESSED_DIR: Path = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
MODELS_DIR: Path = BASE_DIR / os.getenv("MODELS_DIR", "models")
PLOTS_DIR: Path = MODELS_DIR / "plots"
RESULTS_DIR: Path = BASE_DIR / os.getenv("RESULTS_DIR", "data/results")
LOGS_DIR: Path = BASE_DIR / "logs"

# External ranking datasets (HLTV annual stats)
TEAM_RANKING_DIR: Path = BASE_DIR / os.getenv("TEAM_RANKING_DIR", "data/datasets/team_stats")
PLAYER_RANKING_DIR: Path = BASE_DIR / os.getenv("PLAYER_RANKING_DIR", "data/datasets/player_stats")

for _dir in (PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw filenames (inside each year sub-folder)
# ---------------------------------------------------------------------------
MATCHES_FILE: str = "matches.parquet"
MAPS_FILE: str = "maps.parquet"
PLAYER_STATS_FILE: str = "player_stats.parquet"

# Processed output filenames
DATASET_FILE: str = "dataset.parquet"
FEATURES_FILE: str = "features.parquet"

# ---------------------------------------------------------------------------
# Data quality thresholds
# ---------------------------------------------------------------------------
MISSING_COL_THRESHOLD: float = float(os.getenv("MISSING_COL_THRESHOLD", "0.5"))
IQR_OUTLIER_FACTOR: float = float(os.getenv("IQR_OUTLIER_FACTOR", "3.0"))

# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE: float = float(os.getenv("VAL_SIZE", "0.1"))
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))

# ---------------------------------------------------------------------------
# Rolling stats default window
# ---------------------------------------------------------------------------
ROLLING_WINDOW: int = int(os.getenv("ROLLING_WINDOW", "5"))

# ---------------------------------------------------------------------------
# ELO rating system
# ---------------------------------------------------------------------------
ELO_INITIAL: float = float(os.getenv("ELO_INITIAL", "1000.0"))
ELO_K: float = float(os.getenv("ELO_K", "32.0"))

# ---------------------------------------------------------------------------
# Form windows (games and days)
# ---------------------------------------------------------------------------
FORM_WINDOW_SHORT: int = int(os.getenv("FORM_WINDOW_SHORT", "3"))
FORM_WINDOW_LONG: int = int(os.getenv("FORM_WINDOW_LONG", "20"))
FORM_DAYS_SHORT: int = int(os.getenv("FORM_DAYS_SHORT", "30"))
FORM_DAYS_MID: int = int(os.getenv("FORM_DAYS_MID", "60"))
FORM_DAYS_LONG: int = int(os.getenv("FORM_DAYS_LONG", "90"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path | None = LOGS_DIR / "cs_winner.log" if os.getenv("ENABLE_LOG_FILE") else None
