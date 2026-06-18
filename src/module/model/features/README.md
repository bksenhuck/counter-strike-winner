# Feature Engineering — Module Documentation

This package implements all feature engineering for the CS match winner predictor.
Features are organised by **theme**: each file owns one logical group, defines its
required inputs, and exposes a single `add_*_features(df)` function.
The `__init__.py` acts as the pipeline orchestrator that calls them in order.

> **Pre-match safety rule:** Every feature in `FEATURE_COLS` must be computable
> from historical data alone — before the match starts. Features that use
> current-match stats (kills, map scores, live ratings) are computed as raw inputs
> for other modules but do **not** appear in `FEATURE_COLS`.

---

## Table of Contents

- [Package Layout](#package-layout)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Feature Reference](#feature-reference)
  - [Format Features](#1-format-features)
  - [Historical Features](#2-historical-features-elo--win-rates--streaks)
  - [Team History Features](#3-team-history-features)
  - [Momentum Features](#4-momentum-features)
  - [Form Features](#5-form-features)
  - [Fatigue Features](#6-fatigue-features)
  - [Event / Stage Features](#7-event--stage-features)
  - [Temporal Features](#8-temporal-features)
  - [Consistency Features](#9-consistency-features)
  - [Round Features](#10-round-features)
  - [Helper Modules](#helper-modules-not-in-feature_cols)
- [Base Columns (Data Contract)](#base-columns-data-contract)
- [Adding a New Feature Theme](#adding-a-new-feature-theme)

---

## Package Layout

```
features/
├── __init__.py                  # Orchestrator: build_features(), FEATURE_COLS, build_team_stats_lookup()
│
├── format_features.py           # 1. Match format context
├── historical_features.py       # 2. ELO, all-time win rates, streaks, resilience
├── team_history_features.py     # 3. Rolling win rates, head-to-head
├── momentum_features.py         # 4. Rating trends, win-streak indicators
├── form_features.py             # 5. Multi-window form (last-3/20, 30/60/90 days)
├── fatigue_features.py          # 6. Rest days and recent game load
├── event_stage_features.py      # 7. Stage encoding, pressure score
├── temporal_features.py         # 8. Day of week, month, season
├── consistency_features.py      # 9. Star dependency, fragility, firepower
├── round_features.py            # 10. Dominance score, round margins
│
├── map_features.py              # Helper — not in FEATURE_COLS
├── player_features.py           # Helper — not in FEATURE_COLS
├── differential_features.py     # Helper — not in FEATURE_COLS
└── README.md                    # This file
```

---

## How the Pipeline Works

`build_features(df)` in `__init__.py` applies each module sequentially:

```
df (preprocessed dataset)
    │
    ├─► add_map_features()          raw helper: map_win_rate, maps_advantage
    ├─► add_player_features()       raw helper: kill_death_ratio_team1/2
    ├─► add_differential_features() raw helper: rating_diff, adr_diff, kast_diff
    │
    ├─► add_format_features()       → format_maps
    ├─► add_historical_features()   → ELO, win rates, streaks, resilience
    ├─► add_team_history_features() → win_rate_last5/10, h2h_*, form_advantage
    ├─► add_momentum_features()     → rating_ma5, rating_trend, on_win_streak
    ├─► add_form_features()         → last3/20/30d/60d/90d win rates, hot/cold
    ├─► add_fatigue_features()      → days_rest, matches_7d/14d/30d
    ├─► add_event_stage_features()  → stage_encoded, pressure_score
    ├─► add_temporal_features()     → day_of_week, month, season
    ├─► add_consistency_features()  → star_dependency, fragility, firepower
    └─► add_round_features()        → dominance_score
    │
    ▼
df_enriched  (FEATURE_COLS ready for CSWinnerModel)
```

`FEATURE_COLS` is assembled automatically from each module's `FEATURE_COLS` list using `dict.fromkeys` to preserve order and deduplicate. You never need to edit it manually.

---

## Feature Reference

### 1. Format Features

**File:** `format_features.py`

| Feature | Values | Why it matters |
|---|---|---|
| `format_maps` | 1 (bo1), 3 (bo3), 5 (bo5) | Bo1 is far more volatile than bo5; the model needs this to calibrate confidence |

Source: raw `format` string column. Defaults to 1 if absent or unknown.

---

### 2. Historical Features (ELO + win rates + streaks)

**File:** `historical_features.py`

Cumulative statistics from all prior matches. Fully pre-match safe.

| Feature | Formula | Why it matters |
|---|---|---|
| `team1_elo` | ELO before this match (K=32, initial=1000) | Single best proxy for current team strength |
| `team2_elo` | Same for team2 | — |
| `elo_diff` | `team1_elo − team2_elo` | Relative strength; one of the most predictive features |
| `elo_ratio` | `team1_elo / team2_elo` | Non-linear view of the same gap |
| `team1_overall_wr` | Expanding mean win rate from all prior matches | Long-run baseline quality |
| `team2_overall_wr` | Same for team2 | — |
| `overall_wr_diff` | `team1_overall_wr − team2_overall_wr` | Head-to-head quality gap |
| `team1_total_matches` | Count of prior matches | Experience / data confidence |
| `team2_total_matches` | Same for team2 | — |
| `team1_win_streak_all` | Consecutive wins before this match | Psychological momentum |
| `team2_win_streak_all` | Same for team2 | — |
| `team1_loss_streak` | Consecutive losses before this match | Slump indicator |
| `team2_loss_streak` | Same for team2 | — |
| `team1_resilience` | WR in matches immediately after a loss | Mental strength — bounce back? |
| `team2_resilience` | Same for team2 | — |
| `team1_bo1_wr` | Expanding WR in bo1 format only | Format-specific strength |
| `team2_bo1_wr` | Same for team2 | — |
| `team1_bo3_wr` | Expanding WR in bo3 format only | Most common pro format |
| `team2_bo3_wr` | Same for team2 | — |

**ELO:** Standard Elo, `E = 1/(1+10^((R_opp−R)/400))`. Updated after each match, so the value stored for a match reflects the rating *before* it was played.

---

### 3. Team History Features

**File:** `team_history_features.py`

Rolling win rates and head-to-head records. All use `shift(1)` before rolling.

| Feature | Description | Why it matters |
|---|---|---|
| `team1_win_rate_last10` | WR over last 10 matches | 10-game form signal |
| `team2_win_rate_last10` | Same for team2 | — |
| `team1_win_rate_last5` | WR over last 5 matches | More sensitive to hot/cold streaks |
| `team2_win_rate_last5` | Same for team2 | — |
| `h2h_team1_wins` | Times team1 beat team2 in prior meetings | Direct dominance record |
| `h2h_team2_wins` | Times team2 beat team1 | — |
| `h2h_total` | Total prior meetings | How reliable is the h2h record? |
| `h2h_team1_win_rate` | `h2h_team1_wins / h2h_total` | Normalised h2h advantage; NaN if no prior meetings |
| `form_advantage` | `team1_win_rate_last10 − team2_win_rate_last10` | Which team enters in better recent form |

---

### 4. Momentum Features

**File:** `momentum_features.py`

Rating and performance trends. All computed with `shift(1).rolling(N)`.

| Feature | Description | Why it matters |
|---|---|---|
| `team1_rating_ma5` | Rolling mean HLTV rating (last 5 matches) | Smoothed performance level |
| `team2_rating_ma5` | Same for team2 | — |
| `team1_adr_ma5` | Rolling mean ADR (last 5 matches) | Sustained damage output |
| `team2_adr_ma5` | Same for team2 | — |
| `team1_rating_trend` | Linear slope of rating over last 5 matches | Positive = improving; negative = declining |
| `team2_rating_trend` | Same for team2 | — |
| `team1_on_win_streak` | 1 if team1 won their last 3+ consecutive matches | Hot-streak indicator |
| `team2_on_win_streak` | Same for team2 | — |
| `momentum_diff` | `team1_rating_trend − team2_rating_trend` | Who is on the steeper upward trajectory |
| `rating_ma_diff` | `team1_rating_ma5 − team2_rating_ma5` | Rating differential using smoothed recent form |

---

### 5. Form Features

**File:** `form_features.py`

Multi-window form — game-count and calendar-day windows.

| Feature | Window | Why it matters |
|---|---|---|
| `team1_win_rate_last3` | Last 3 games | Captures the very latest hot/cold state |
| `team2_win_rate_last3` | Last 3 games | — |
| `team1_win_rate_last20` | Last 20 games | Long baseline form |
| `team2_win_rate_last20` | Last 20 games | — |
| `team1_win_rate_30d` | Last 30 calendar days | Period-specific form |
| `team2_win_rate_30d` | Last 30 days | — |
| `team1_win_rate_60d` | Last 60 days | Mid-term form |
| `team2_win_rate_60d` | Last 60 days | — |
| `team1_win_rate_90d` | Last 90 days | Quarterly performance |
| `team2_win_rate_90d` | Last 90 days | — |
| `team1_rating_ma20` | 20-game rolling rating MA | Long-term performance baseline |
| `team2_rating_ma20` | Same for team2 | — |
| `team1_hot_cold` | `rating_ma5 − rating_ma20` | Positive = team is improving vs their own baseline |
| `team2_hot_cold` | Same for team2 | — |
| `form3_diff` | `team1_last3 − team2_last3` | Short-term form gap |
| `form_30d_diff` | `team1_30d − team2_30d` | Calendar-based form gap |

---

### 6. Fatigue Features

**File:** `fatigue_features.py`

Game load and recovery time — all from prior match dates only.

| Feature | Description | Why it matters |
|---|---|---|
| `team1_days_rest` | Days since team1's last match | Well-rested teams can perform better |
| `team2_days_rest` | Same for team2 | — |
| `team1_matches_7d` | Matches in last 7 days | Short-term fatigue |
| `team2_matches_7d` | Same for team2 | — |
| `team1_matches_14d` | Matches in last 14 days | 2-week workload |
| `team2_matches_14d` | Same for team2 | — |
| `team1_matches_30d` | Matches in last 30 days | Monthly game load |
| `team2_matches_30d` | Same for team2 | — |
| `rest_advantage` | `team1_days_rest − team2_days_rest` | Positive = team1 more rested |
| `fatigue_advantage` | `team2_matches_30d − team1_matches_30d` | Positive = team2 more fatigued |

---

### 7. Event / Stage Features

**File:** `event_stage_features.py`

| Feature | Values | Why it matters |
|---|---|---|
| `stage_encoded` | 0=Group, 1=Quarter/Playoff, 2=Semi, 3=Final, 4=Grand Final | High-stakes matches behave differently |
| `is_playoff` | 0 or 1 | Playoff teams tend to be more prepared |
| `is_final` | 0 or 1 | Finals show maximum team focus |
| `pressure_score` | `stage_encoded / 4` (0→1) | Continuous pressure proxy for the model |

---

### 8. Temporal Features

**File:** `temporal_features.py`

| Feature | Values | Why it matters |
|---|---|---|
| `day_of_week` | 0=Mon … 6=Sun | Weekend vs weekday scheduling effects |
| `month` | 1–12 | Seasonal meta and tournament cycle |
| `is_weekend` | 0 or 1 | Most online matches are on weekends |
| `season` | 1=Winter, 2=Spring, 3=Summer, 4=Autumn | Roster windows and meta shifts by season |

---

### 9. Consistency Features

**File:** `consistency_features.py`

Characterises *how* a team wins, using per-player max/std stats from `data.py`.

| Feature | Formula | Why it matters |
|---|---|---|
| `team1_star_dependency` | `rating_max_team1 / rating_team1` | How much the team relies on its best player |
| `team2_star_dependency` | Same for team2 | — |
| `team1_fragility` | `rating_max_team1 / (rating_team1 + ε)` | Whether one star dominates the squad |
| `team2_fragility` | Same for team2 | — |
| `team1_consistency` | `1 / (rating_std_team1 + ε)` | Higher = more uniform ratings across squad |
| `team2_consistency` | Same for team2 | — |
| `team1_firepower` | `0.5×rating + 0.3×(adr/100) + 0.2×(kills/20)` | Composite offensive power index |
| `team2_firepower` | Same for team2 | — |
| `firepower_diff` | `team1_firepower − team2_firepower` | Which team has superior firepower |
| `star_dependency_diff` | `team1_star_dependency − team2_star_dependency` | — |
| `consistency_diff` | `team1_consistency − team2_consistency` | — |

---

### 10. Round Features

**File:** `round_features.py`

| Feature | Description | Why it matters |
|---|---|---|
| `dominance_score` | `avg_round_diff` for this match (team1 perspective) | 16-3 is more dominant than 16-14 |
| `avg_round_diff_ma5` | Rolling 5-match mean of `avg_round_diff` per team1 | Does this team consistently blow out opponents? |

---

### Helper Modules (not in FEATURE_COLS)

These modules add columns used as raw inputs by rolling/trend modules, but the columns are in-match statistics — unavailable before the match starts.

| Module | Adds | Note |
|---|---|---|
| `map_features.py` | `map_win_rate`, `maps_advantage`, `maps_won_team1/2`, `total_maps` | Current series results |
| `player_features.py` | `kill_death_ratio_team1/2` | Current match derived stats |
| `differential_features.py` | `rating_diff`, `adr_diff`, `kast_diff` | Current match differentials |

---

## Base Columns (Data Contract)

Must be present before `build_features()` is called. Produced by `data.py:preprocess()`.

| Column | Source | Type |
|---|---|---|
| `match_id`, `date` | `matches.parquet` | int, datetime |
| `team1`, `team2` | `matches.parquet` | str |
| `format`, `stage`, `event` | `matches.parquet` | str (optional) |
| `score_team1`, `score_team2` | `matches.parquet` | int |
| `maps_won_team1/2`, `total_maps` | aggregated from `maps.parquet` | int |
| `avg_round_diff` | aggregated from `maps.parquet` | float |
| `kills_team1/2` … `rating_team1/2` | aggregated mean from `player_stats.parquet` | float |
| `rating_max_team1/2`, `rating_std_team1/2` | aggregated max/std from `player_stats.parquet` | float |
| `winner` | derived: `score_team1 > score_team2` | int (0/1) |

---

## Adding a New Feature Theme

1. Create `src/module/model/features/my_theme.py`:

```python
import pandas as pd
from utils.data_processing import fill_feature_means  # centralised NaN fill

REQUIRED_COLS: list[str] = ["col_that_must_exist"]
FEATURE_COLS:  list[str] = ["my_new_feature"]

def add_my_theme_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["my_new_feature"] = ...
    fill_feature_means(df, FEATURE_COLS)
    return df
```

2. Register in `__init__.py`: import, add to `FEATURE_COLS`, call in `build_features()`.

3. **Only add to `FEATURE_COLS`** if the feature is computable before the match starts.
