# Counter-Strike Winner Predictor

An XGBoost-based machine learning pipeline that predicts the winner of Counter-Strike matches **before the match starts**, using only historical team and player data. Built with a clean, modular architecture and SHAP explainability.

> **Pre-match by design.** Every feature in `FEATURE_COLS` is computable from prior match history — no current-match statistics are used, so the model can genuinely predict future matches given only team names and format.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Data](#data)
- [Quick Start](#quick-start)
- [Pre-Match Prediction API](#pre-match-prediction-api)
- [Configuration](#configuration)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Overview](#feature-overview)
- [Explainability (SHAP)](#explainability-shap)
- [Utilities](#utilities)
- [Testing](#testing)
- [Development](#development)

---

## Project Structure

```
counter-strike-winner/
│
├── conf/
│   └── settings.py              # General config: paths, splits, ELO params (env-overridable)
│
├── data/processed/              # Generated: dataset.parquet, features.parquet
│
├── datasets/                    # Source parquet files organised by year
│   └── 20XX/
│       ├── matches.parquet
│       ├── maps.parquet
│       └── player_stats.parquet
│
├── models/                      # Saved pipelines (.pkl) and SHAP plots
│   └── plots/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/module/model/
│   ├── data.py                  # Data loading and preprocessing
│   ├── features/                # Feature engineering (themed package)
│   │   ├── __init__.py              # Orchestrator: build_features(), FEATURE_COLS, build_team_stats_lookup()
│   │   ├── format_features.py       # bo1/bo3/bo5 → 1/3/5
│   │   ├── historical_features.py   # ELO, all-time win rates, streaks, resilience
│   │   ├── team_history_features.py # Rolling win rates, head-to-head
│   │   ├── momentum_features.py     # Rating trends, win streaks
│   │   ├── form_features.py         # Multi-window form (last-3/20, 30/60/90 days)
│   │   ├── fatigue_features.py      # Rest days and game load
│   │   ├── event_stage_features.py  # Stage encoding, pressure score
│   │   ├── temporal_features.py     # Day of week, month, season
│   │   ├── consistency_features.py  # Star dependency, fragility, firepower
│   │   ├── round_features.py        # Dominance score, round margins
│   │   ├── map_features.py          # (helper — raw inputs only, not in FEATURE_COLS)
│   │   ├── player_features.py       # (helper — raw inputs only, not in FEATURE_COLS)
│   │   ├── differential_features.py # (helper — raw inputs only, not in FEATURE_COLS)
│   │   └── README.md                # Full feature documentation
│   ├── model.py                 # CSWinnerModel class
│   ├── train.py                 # Training entry point
│   ├── predict.py               # Batch prediction entry point
│   ├── explain.py               # ModelExplainer class (SHAP)
│   └── settings.py              # Model-local config: hyperparams, SHAP display
│
├── utils/
│   ├── logger.py                # get_logger()
│   ├── decorators.py            # @log_call, @timer, @validate_dataframe
│   ├── file_utils.py            # All file I/O: parquet, pkl, csv, json
│   └── data_processing.py       # Cleaning, splitting, scaling, ColumnTransformer helpers
│
├── tests/
├── pyproject.toml
└── .gitignore
```

---

## Data

Three parquet files per year (2012–2020):

| File | Key Columns |
|---|---|
| `matches.parquet` | `match_id`, `date`, `team1`, `team2`, `score_team1`, `score_team2`, `format`, `event`, `stage`, `team1_lineup`, `team2_lineup` |
| `maps.parquet` | `match_id`, `map_order`, `map_name`, `score_team1`, `score_team2` |
| `player_stats.parquet` | `match_id`, `map_order`, `team`, `player_name`, `kills`, `deaths`, `adr`, `kast`, `rating` |

**Target variable:** `winner` — `1` if team1 won, `0` if team2 won. Draws are excluded.

---

## Quick Start

### Install

```bash
# With Poetry (recommended)
poetry install

# Or with pip
pip install -r requirements.txt
```

### Full training pipeline

```python
from src.module.model.data import load_raw_data, preprocess
from src.module.model.features import build_features, get_feature_matrix, build_team_stats_lookup
from src.module.model.model import CSWinnerModel
from utils.data_processing import split_data

# 1. Load and preprocess
raw = load_raw_data()
df  = preprocess(raw)
df  = build_features(df)

# 2. Split and train
X, y = get_feature_matrix(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

model = CSWinnerModel()
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
model.evaluate(X_test, y_test)
model.save()
```

### Or via the entry point scripts

```bash
python -m src.module.model.train          # train (add --cross_val for CV)
python -m src.module.model.predict data/processed/features.parquet
```

### Run tests

```bash
pytest
pytest --cov=src --cov=utils --cov-report=term-missing
```

---

## Pre-Match Prediction API

The primary use case: **predict a future match given only team names**.

```python
from src.module.model.data import load_raw_data, preprocess
from src.module.model.features import build_features, build_team_stats_lookup
from src.module.model.model import CSWinnerModel

# Build the historical feature lookup (do this once per session)
raw    = load_raw_data()
df     = preprocess(raw)
df     = build_features(df)
lookup = build_team_stats_lookup(df)   # {team_name: {stat: value}}

# Load the trained model
model  = CSWinnerModel.load()

# Predict any future match
result = model.predict_match(
    team1="NaVi",
    team2="Astralis",
    match_format="bo3",
    stage="Grand Final",
    team_stats=lookup,
)

print(result)
# {
#   "predicted_winner": "NaVi",
#   "team1": "NaVi",
#   "team2": "Astralis",
#   "prob_team1_wins": 0.6123,
#   "prob_team2_wins": 0.3877,
#   "elo_diff": 42.3
# }
```

`build_team_stats_lookup(df)` extracts each team's most recent historical feature vector regardless of which position (team1 or team2) they occupied in their last match. Build it once and reuse it for many predictions.

### Why not just use in-match stats?

| Feature type | Example | Available pre-match? |
|---|---|---|
| Current match kills / rating | `kills_team1`, `rating_team1` | ❌ No — happens during the match |
| Current map scores | `maps_won_team1`, `map_win_rate` | ❌ No — series in progress |
| Historical rolling average | `team1_rating_ma5` | ✅ Yes — last 5 games before this match |
| ELO before this match | `team1_elo` | ✅ Yes — computed from all prior results |
| Head-to-head record | `h2h_team1_wins` | ✅ Yes — all prior meetings |

Using current-match stats during training creates **data leakage** — the model sees the match outcome hidden inside its own features, giving artificially high accuracy that collapses at prediction time.

---

## Configuration

### `conf/settings.py` — General / Infrastructure

| Variable | Default | Env override | Description |
|---|---|---|---|
| `RAW_DIR` | `datasets/` | `RAW_DIR` | Root folder with year sub-directories |
| `PROCESSED_DIR` | `data/processed/` | `PROCESSED_DIR` | Output folder |
| `MODELS_DIR` | `models/` | `MODELS_DIR` | Saved pipeline location |
| `TEST_SIZE` | `0.2` | `TEST_SIZE` | Test fraction |
| `VAL_SIZE` | `0.1` | `VAL_SIZE` | Validation fraction |
| `RANDOM_STATE` | `42` | `RANDOM_STATE` | Global seed |
| `MISSING_COL_THRESHOLD` | `0.5` | `MISSING_COL_THRESHOLD` | Drop columns above this NaN fraction |
| `IQR_OUTLIER_FACTOR` | `3.0` | `IQR_OUTLIER_FACTOR` | IQR multiplier for outlier removal |
| `ROLLING_WINDOW` | `5` | `ROLLING_WINDOW` | Default rolling window size |
| `ELO_INITIAL` | `1000.0` | `ELO_INITIAL` | Starting ELO for all teams |
| `ELO_K` | `32.0` | `ELO_K` | ELO K-factor (update magnitude per match) |
| `FORM_WINDOW_SHORT` | `3` | `FORM_WINDOW_SHORT` | Short game-count form window |
| `FORM_WINDOW_LONG` | `20` | `FORM_WINDOW_LONG` | Long game-count form window |
| `FORM_DAYS_SHORT` | `30` | `FORM_DAYS_SHORT` | 30-day calendar form window |
| `FORM_DAYS_MID` | `60` | `FORM_DAYS_MID` | 60-day calendar form window |
| `FORM_DAYS_LONG` | `90` | `FORM_DAYS_LONG` | 90-day calendar form window |
| `LOG_LEVEL` | `INFO` | `LOG_LEVEL` | Python logging level |

### `src/module/model/settings.py` — Model / Hyperparameters

| Variable | Default | Env override | Description |
|---|---|---|---|
| `XGB_N_ESTIMATORS` | `300` | `XGB_N_ESTIMATORS` | Number of XGBoost trees |
| `XGB_MAX_DEPTH` | `5` | `XGB_MAX_DEPTH` | Max tree depth |
| `XGB_LEARNING_RATE` | `0.05` | `XGB_LEARNING_RATE` | Boosting learning rate |
| `XGB_SUBSAMPLE` | `0.8` | `XGB_SUBSAMPLE` | Row subsampling ratio |
| `XGB_COLSAMPLE_BYTREE` | `0.8` | `XGB_COLSAMPLE_BYTREE` | Column subsampling ratio |
| `CV_N_SPLITS` | `5` | `CV_N_SPLITS` | Number of CV folds |
| `PIPELINE_NAME` | `xgboost_pipeline` | `PIPELINE_NAME` | Saved `.pkl` filename stem |
| `PREDICTION_THRESHOLD` | `0.5` | — | Probability threshold for label assignment |

Override example:

```bash
XGB_N_ESTIMATORS=500 ELO_K=24 python -m src.module.model.train
```

---

## Pipeline Architecture

```
Raw Data (parquet per year, 2012–2020)
        │
        ▼
  data.py: load_raw_data()
        │  concat_years() across all year folders
        │
        ▼
  data.py: preprocess()
        │  clean matches (drop draws, parse dates)
        │  aggregate maps  → maps_won, total_maps, avg_round_diff
        │  aggregate stats → mean/max/std kills/deaths/adr/kast/rating per team
        │  drop high-NaN columns, fill numeric median
        │  derive target: winner = (score_team1 > score_team2)
        │
        ▼
  features/: build_features()
        │
        │  ── raw-input helpers (NOT in FEATURE_COLS) ────────────────
        │  map_features          → map_win_rate, maps_advantage
        │  player_features       → kill_death_ratio_team1/2
        │  differential_features → rating_diff, adr_diff, kast_diff
        │
        │  ── pre-match safe features (all in FEATURE_COLS) ──────────
        │  format_features       → format_maps (1/3/5)
        │  historical_features   → ELO, overall_wr, bo1/bo3_wr, streaks, resilience
        │  team_history_features → win_rate_last5/10, h2h_*, form_advantage
        │  momentum_features     → rating_ma5, adr_ma5, rating_trend, win_streak
        │  form_features         → last3/20 win rates, 30/60/90d rates, hot/cold
        │  fatigue_features      → days_rest, matches_7d/14d/30d, fatigue_advantage
        │  event_stage_features  → stage_encoded, is_playoff, is_final, pressure_score
        │  temporal_features     → day_of_week, month, is_weekend, season
        │  consistency_features  → star_dependency, fragility, consistency, firepower
        │  round_features        → dominance_score, avg_round_diff_ma5
        │
        ▼
  CSWinnerModel.fit()
        │  sklearn Pipeline:
        │    SimpleImputer(median) → StandardScaler → XGBClassifier
        │
        ▼
  model.pkl
        │
        ├─► predict(df)             — batch prediction from feature matrix
        ├─► predict_match(t1, t2)   — pre-match: team names + historical lookup
        └─► ModelExplainer          — global SHAP summary / per-match waterfall
```

---

## Feature Overview

All features are **pre-match safe** — computable before the match using only historical data. See [`src/module/model/features/README.md`](src/module/model/features/README.md) for full per-feature documentation.

| Module | Key Features |
|---|---|
| `format_features` | `format_maps` |
| `historical_features` | `team1/2_elo`, `elo_diff`, `elo_ratio`, `overall_wr`, `bo1/bo3_wr`, win/loss streaks, `resilience` |
| `team_history_features` | `win_rate_last5/10`, `h2h_team1/2_wins`, `h2h_team1_win_rate`, `form_advantage` |
| `momentum_features` | `rating_ma5`, `adr_ma5`, `rating_trend`, `on_win_streak`, `momentum_diff`, `rating_ma_diff` |
| `form_features` | `win_rate_last3/20`, `win_rate_30d/60d/90d`, `rating_ma20`, `hot_cold`, `form3_diff`, `form_30d_diff` |
| `fatigue_features` | `days_rest`, `matches_7d/14d/30d`, `rest_advantage`, `fatigue_advantage` |
| `event_stage_features` | `stage_encoded`, `is_playoff`, `is_final`, `pressure_score` |
| `temporal_features` | `day_of_week`, `month`, `is_weekend`, `season` |
| `consistency_features` | `star_dependency`, `fragility`, `consistency`, `firepower`, `firepower_diff`, `consistency_diff` |
| `round_features` | `dominance_score`, `avg_round_diff_ma5` |

---

## Explainability (SHAP)

```python
from src.module.model.explain import ModelExplainer

explainer = ModelExplainer().load()

# Global importance (beeswarm plot)
explainer.shap_summary(df, save=True, filename="shap_summary.png")

# Per-match explanation (waterfall plot)
result = explainer.explain_match(
    match_data={"team1_elo": 1120, "team2_elo": 980, "elo_diff": 140, ...},
    save=True, filename="match_001.png",
)
print(result["top_reasons"])
# [{"feature": "elo_diff", "shap_value": 0.51, "feature_value": 140.0}, ...]
```

---

## Utilities

### `utils/data_processing.py` — All data operations in one file

| Function | Description |
|---|---|
| `drop_missing(df, threshold)` | Drop columns exceeding NaN fraction threshold |
| `fill_numeric_median(df)` | Fill NaN with column median |
| `encode_categorical(df, cols)` | Label-encode categorical columns |
| `add_rolling_stats(df, group, cols, window)` | Rolling mean per group |
| `split_data(X, y, test_size, val_size)` | Stratified train / val / test split |
| `remove_outliers_iqr(df, cols, factor)` | Remove IQR-based outlier rows |
| `clip_outliers(df, cols, lo, hi)` | Clip extreme values to percentile bounds |
| `log_transform(df, cols)` | `log1p` for right-skewed columns |
| `get_scaler(kind)` | Unfitted scaler: `standard`, `minmax`, `robust` |
| `fit_scaler(df, cols, kind)` | Fit + transform, return `(df, scaler)` |
| `apply_scaler(df, cols, scaler)` | Apply a fitted scaler at inference time |
| `build_column_transformer(numeric, cat)` | sklearn `ColumnTransformer` |
| `build_preprocessing_pipeline(numeric, cat)` | Standalone preprocessing Pipeline |

### `utils/file_utils.py`

| Function | Description |
|---|---|
| `load_parquet` / `save_parquet` | Parquet I/O |
| `load_pkl` / `save_pkl` | Pickle I/O (model persistence) |
| `load_csv` / `save_csv` | CSV I/O |
| `load_json` / `save_json` | JSON I/O |
| `concat_years(base_dir, filename)` | Concatenate a file across all year sub-folders |

### `utils/decorators.py`

| Decorator | Description |
|---|---|
| `@timer` | Log elapsed time |
| `@log_call` | Log function entry/exit |
| `@validate_dataframe(*cols)` | Raise `ValueError` if required columns are missing |

---

## Testing

```bash
pytest                         # run all tests
pytest tests/test_features.py  # run one module
pytest -v --tb=short
```

---

## Development

```bash
ruff check .           # lint
ruff format --check .  # format check
ruff check --fix .     # auto-fix
```

### Adding a new feature theme

1. Create `src/module/model/features/my_theme.py` with `REQUIRED_COLS`, `FEATURE_COLS`, and `add_my_theme_features(df)`.
2. Import in `features/__init__.py` and add to `FEATURE_COLS` + `build_features()`.

**Rule:** only add to `FEATURE_COLS` if the feature is computable **before** the match starts. In-match stats must not appear there.
