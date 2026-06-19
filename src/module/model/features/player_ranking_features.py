"""HLTV official player ranking features — pre-match safe.

Source: data/datasets/player_stats/YYYY/player_stats.parquet
Filter: match_type='all', map_name='all', cs_version='both'

For each match we parse team1_lineup / team2_lineup (pipe-separated player names),
look up each player's HLTV ranking from year Y-1, then aggregate per team:
  - best_player_rank  : rank of the team's highest-ranked player (lower = better)
  - avg_player_rating : mean rating across all 5 players
  - top3_avg_rating   : mean rating of the 3 highest-rated players (star power proxy)
  - player_rank_diff  : team2_best_rank - team1_best_rank (positive → team1 has better star)
  - player_rating_diff: team1_avg - team2_avg

AT PREDICTION TIME:
  These team-level aggregates are captured by build_team_stats_lookup() (as
  team1_best_player_rank → best_player_rank in the lookup). If the user does not
  provide lineup data, predict_match() uses the last known values from the lookup.
  player_rank_diff / player_rating_diff are recomputed there from those lookup values.

GRACEFUL DEGRADATION:
  If team1_lineup / team2_lineup columns are absent or ranking data is missing,
  all features are filled with column means (via fill_feature_means).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from conf.settings import PLAYER_RANKING_DIR
from utils.data_processing import fill_feature_means
from utils.logger import get_logger

_logger = get_logger(__name__)

REQUIRED_COLS: list[str] = ["date", "team1", "team2"]

FEATURE_COLS: list[str] = [
    "team1_best_player_rank",
    "team2_best_player_rank",
    "team1_avg_player_rating",
    "team2_avg_player_rating",
    "team1_top3_avg_rating",
    "team2_top3_avg_rating",
    "player_rank_diff",     # team2_best_rank - team1_best_rank (positive → team1 has better star)
    "player_rating_diff",   # team1_avg - team2_avg
]


def add_player_ranking_features(
    df: pd.DataFrame,
    ranking_dir: Path = PLAYER_RANKING_DIR,
) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    has_lineups = "team1_lineup" in df.columns and "team2_lineup" in df.columns
    player_df = _load_all_player_rankings(ranking_dir)

    if not has_lineups or player_df.empty:
        if not has_lineups:
            _logger.warning("Lineup columns absent — player ranking features will use imputed means.")
        if player_df.empty:
            _logger.warning("No player ranking data found in %s.", ranking_dir)
        for col in FEATURE_COLS:
            df[col] = float("nan")
        fill_feature_means(df, FEATURE_COLS)
        return df

    df["_year"] = pd.to_datetime(df["date"]).dt.year - 1

    for side in ("team1", "team2"):
        lineup_col = f"{side}_lineup"
        agg = _aggregate_lineup_stats(df, lineup_col, "_year", player_df)
        df[f"{side}_best_player_rank"] = agg["best_rank"].values
        df[f"{side}_avg_player_rating"] = agg["avg_rating"].values
        df[f"{side}_top3_avg_rating"] = agg["top3_rating"].values

    df["player_rank_diff"] = df["team2_best_player_rank"] - df["team1_best_player_rank"]
    df["player_rating_diff"] = df["team1_avg_player_rating"] - df["team2_avg_player_rating"]

    df = df.drop(columns=["_year"])
    fill_feature_means(df, FEATURE_COLS)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_lineup_stats(
    df: pd.DataFrame,
    lineup_col: str,
    year_col: str,
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    """Explode lineups, join to player_df, aggregate back to match-level."""
    exploded = (
        df[[lineup_col, year_col]]
        .copy()
        .assign(_idx=range(len(df)))
        .assign(_players=df[lineup_col].str.split("|"))
        .explode("_players")
        .rename(columns={"_players": "_player_lower"})
    )
    exploded["_player_lower"] = exploded["_player_lower"].str.lower().str.strip()

    joined = exploded.merge(
        player_df,
        left_on=[year_col, "_player_lower"],
        right_on=["_year", "_player_lower"],
        how="left",
    )

    def _agg(g: pd.DataFrame) -> pd.Series:
        ranks = g["rank"].dropna()
        ratings = g["rating"].dropna()
        return pd.Series({
            "best_rank": ranks.min() if len(ranks) else float("nan"),
            "avg_rating": ratings.mean() if len(ratings) else float("nan"),
            "top3_rating": ratings.nlargest(3).mean() if len(ratings) >= 3 else ratings.mean(),
        })

    return joined.groupby("_idx").apply(_agg, include_groups=False).reset_index(drop=True)


def _load_all_player_rankings(ranking_dir: Path) -> pd.DataFrame:
    """Concat all year parquets into one long DataFrame for vectorised joins."""
    if not ranking_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for year_dir in sorted(ranking_dir.iterdir()):
        if not year_dir.is_dir() or year_dir.name == "all":
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        pq = year_dir / "player_stats.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        sub = raw[
            (raw["match_type"] == "all")
            & (raw["map_name"] == "all")
            & (raw["cs_version"] == "both")
        ][["player_name", "rank", "rating"]].copy()
        sub["_year"] = year
        sub["_player_lower"] = sub["player_name"].str.lower().str.strip()
        frames.append(sub.drop(columns=["player_name"]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
