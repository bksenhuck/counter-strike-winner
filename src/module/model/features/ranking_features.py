"""HLTV official team ranking features — pre-match safe.

Source: data/datasets/team_stats/YYYY/team_stats.parquet
Filter: match_type='all', map_name='all', cs_version='both'

For each match in year Y we join the team's HLTV ranking stats from year Y-1
to avoid data leakage (we can't know the final-year ranking before it ends).

At PREDICTION TIME these features are captured by build_team_stats_lookup()
as part of each team's historical profile, so no external data is needed.
Diffs (hltv_rank_diff, hltv_rating_diff) must be recomputed in predict_match().
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from conf.settings import TEAM_RANKING_DIR
from utils.data_processing import fill_feature_means
from utils.logger import get_logger

_logger = get_logger(__name__)

REQUIRED_COLS: list[str] = ["date", "team1", "team2"]

FEATURE_COLS: list[str] = [
    "team1_hltv_rank",
    "team2_hltv_rank",
    "team1_hltv_rating",
    "team2_hltv_rating",
    "team1_hltv_kd",
    "team2_hltv_kd",
    "team1_hltv_maps",
    "team2_hltv_maps",
    "hltv_rank_diff",     # team2_rank - team1_rank  (positive → team1 is ranked higher)
    "hltv_rating_diff",   # team1_rating - team2_rating
]


def add_ranking_features(
    df: pd.DataFrame,
    ranking_dir: Path = TEAM_RANKING_DIR,
) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    rank_df = _load_all_rankings(ranking_dir)
    if rank_df.empty:
        _logger.warning("No team ranking data found in %s — skipping ranking features.", ranking_dir)
        for col in FEATURE_COLS:
            df[col] = float("nan")
        fill_feature_means(df, FEATURE_COLS)
        return df

    # Use Y-1 rankings for matches in year Y
    df["_year"] = pd.to_datetime(df["date"]).dt.year - 1
    df["_t1_key"] = df["team1"].str.lower().str.strip()
    df["_t2_key"] = df["team2"].str.lower().str.strip()

    for side, key_col in [("team1", "_t1_key"), ("team2", "_t2_key")]:
        joined = (
            df[["_year", key_col]]
            .rename(columns={key_col: "_team"})
            .merge(
                rank_df,
                left_on=["_year", "_team"],
                right_on=["_year", "_name_lower"],
                how="left",
            )
        )
        df[f"{side}_hltv_rank"] = joined["rank"].values
        df[f"{side}_hltv_rating"] = joined["rating"].values
        df[f"{side}_hltv_kd"] = joined["kd_ratio"].values
        df[f"{side}_hltv_maps"] = joined["maps_played"].values

    df["hltv_rank_diff"] = df["team2_hltv_rank"] - df["team1_hltv_rank"]
    df["hltv_rating_diff"] = df["team1_hltv_rating"] - df["team2_hltv_rating"]

    df = df.drop(columns=["_year", "_t1_key", "_t2_key"])
    fill_feature_means(df, FEATURE_COLS)
    return df


def _load_all_rankings(ranking_dir: Path) -> pd.DataFrame:
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
        pq = year_dir / "team_stats.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        sub = raw[
            (raw["match_type"] == "all")
            & (raw["map_name"] == "all")
            & (raw["cs_version"] == "both")
        ][["team_name", "rank", "rating", "kd_ratio", "maps_played"]].copy()
        sub["_year"] = year
        sub["_name_lower"] = sub["team_name"].str.lower().str.strip()
        frames.append(sub.drop(columns=["team_name"]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
