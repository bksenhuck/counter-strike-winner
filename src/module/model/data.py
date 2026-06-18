"""Data loading and pre-processing pipeline."""

from pathlib import Path

import pandas as pd

from conf.settings import (
    DATASET_FILE,
    MAPS_FILE,
    MATCHES_FILE,
    MISSING_COL_THRESHOLD,
    PLAYER_STATS_FILE,
    PROCESSED_DIR,
    RAW_DIR,
)
from utils.data_processing import drop_missing, fill_numeric_median
from utils.decorators import log_call, timer
from utils.file_utils import concat_years, save_parquet
from utils.logger import get_logger

_logger = get_logger(__name__)


@timer
@log_call
def load_raw_data(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load and concatenate matches, maps and player_stats across all year folders."""
    _logger.info("Loading raw data from %s", raw_dir)

    _logger.info("  [1/3] matches...")
    matches = concat_years(raw_dir, MATCHES_FILE)
    _logger.info("        %d rows", len(matches))

    _logger.info("  [2/3] maps...")
    maps = concat_years(raw_dir, MAPS_FILE)
    _logger.info("        %d rows", len(maps))

    _logger.info("  [3/3] player_stats...")
    player_stats = concat_years(raw_dir, PLAYER_STATS_FILE)
    _logger.info("        %d rows", len(player_stats))

    return {"matches": matches, "maps": maps, "player_stats": player_stats}


@timer
@log_call
def preprocess(
    raw: dict[str, pd.DataFrame],
    output_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Merge tables, derive target column and persist the processed dataset.

    Steps:
        1. Clean matches (parse dates, drop draws/nulls).
        2. Aggregate per-map stats per match.
        3. Aggregate per-player stats to team averages per match.
        4. Merge all three tables on match_id.
        5. Encode match format (bo1/bo3/bo5).
        6. Drop high-NaN columns; fill remaining NaN with median.
        7. Save to output_dir/DATASET_FILE.
    """
    _logger.info("  [1/7] Cleaning matches (parse dates, drop draws/nulls)...")
    matches = _clean_matches(raw["matches"])
    _logger.info("        %d valid matches", len(matches))

    _logger.info("  [2/7] Aggregating map stats (rounds won/lost, round diff)...")
    maps_agg = _aggregate_maps(raw["maps"])

    _logger.info("  [3/7] Aggregating player stats (mean/max/std per team)...")
    player_agg = _aggregate_player_stats(raw["player_stats"])

    _logger.info("  [4/7] Merging matches + maps + player_stats on match_id...")
    df = matches.merge(maps_agg, on="match_id", how="left")
    df = df.merge(player_agg, on="match_id", how="left")
    _logger.info("        merged shape: %d rows × %d cols", *df.shape)

    _logger.info("  [5/7] Dropping high-NaN columns (threshold=%.0f%%)...", MISSING_COL_THRESHOLD * 100)
    df = drop_missing(df, threshold=MISSING_COL_THRESHOLD)

    _logger.info("  [6/7] Filling remaining NaN with column median...")
    df = fill_numeric_median(df)

    _logger.info("  [7/7] Deriving target column (winner = score_team1 > score_team2)...")
    df["winner"] = (df["score_team1"] > df["score_team2"]).astype(int)

    save_parquet(df, output_dir / DATASET_FILE)
    _logger.info("Preprocessed dataset ready: %d rows × %d cols", len(df), df.shape[1])
    return df


def _clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["team1", "team2", "score_team1", "score_team2"])
    df = df[df["score_team1"] != df["score_team2"]]
    return df.reset_index(drop=True)


def _aggregate_maps(df: pd.DataFrame) -> pd.DataFrame:
    """Maps won/lost and round score stats per match_id."""
    def _per_match(g: pd.DataFrame) -> pd.Series:
        won = g[g["score_team1"] > g["score_team2"]]
        lost = g[g["score_team1"] < g["score_team2"]]
        round_diff = (g["score_team1"] - g["score_team2"]).mean()
        return pd.Series(
            {
                "maps_won_team1": len(won),
                "maps_won_team2": len(lost),
                "total_maps": len(g),
                "avg_rounds_won": won["score_team1"].mean() if len(won) else float("nan"),
                "avg_rounds_lost": lost["score_team1"].mean() if len(lost) else float("nan"),
                "avg_round_diff": round_diff,
            }
        )

    return (
        df.groupby("match_id")
        .apply(_per_match, include_groups=False)
        .reset_index()
    )


def _aggregate_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/max/std kills/deaths/adr/kast/rating per team per match, wide format.

    Suffix conventions (per team position):
      *_team1  — mean (e.g. rating_team1 = mean rating of team1 players)
      *_max_team1  — best player value (star proxy)
      *_std_team1  — spread within team (consistency proxy)
    """
    stat_cols = ["kills", "deaths", "adr", "kast", "rating"]
    base = df.groupby(["match_id", "team"])

    mean_df = base[stat_cols].mean().round(4).reset_index()
    max_df = (
        base[stat_cols].max().round(4).reset_index()
        .rename(columns={c: f"{c}_max" for c in stat_cols})
    )
    std_df = (
        base[stat_cols].std().round(4).reset_index()
        .rename(columns={c: f"{c}_std" for c in stat_cols})
    )

    all_stat_cols = stat_cols + [f"{c}_max" for c in stat_cols] + [f"{c}_std" for c in stat_cols]
    agg = (
        mean_df
        .merge(max_df, on=["match_id", "team"])
        .merge(std_df, on=["match_id", "team"])
    )

    team1 = (
        agg[agg["team"] == 1]
        .drop(columns="team")
        .add_suffix("_team1")
        .rename(columns={"match_id_team1": "match_id"})
    )
    team2 = (
        agg[agg["team"] == 2]
        .drop(columns="team")
        .add_suffix("_team2")
        .rename(columns={"match_id_team2": "match_id"})
    )
    return team1.merge(team2, on="match_id", how="outer")
