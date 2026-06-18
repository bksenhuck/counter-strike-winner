"""Historical team statistics and ELO rating system.

All features are computed retrospectively — each match uses only results
from *prior* matches, so they are safe to use at prediction time (pre-match).

ELO is updated AFTER each match; the ELO stored for a match reflects the
team's rating BEFORE that match was played.
"""

from __future__ import annotations

import pandas as pd

from conf.settings import ELO_INITIAL, ELO_K, FORM_WINDOW_SHORT
from utils.data_processing import fill_feature_means

REQUIRED_COLS: list[str] = ["match_id", "date", "team1", "team2", "winner"]

FEATURE_COLS: list[str] = [
    "team1_elo",
    "team2_elo",
    "elo_diff",
    "elo_ratio",
    "team1_overall_wr",
    "team2_overall_wr",
    "overall_wr_diff",
    "team1_total_matches",
    "team2_total_matches",
    "team1_win_streak_all",
    "team2_win_streak_all",
    "team1_loss_streak",
    "team2_loss_streak",
    "team1_resilience",
    "team2_resilience",
    "team1_bo1_wr",
    "team2_bo1_wr",
    "team1_bo3_wr",
    "team2_bo3_wr",
]


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ELO, all-time win rates, format win rates and streaks."""
    df = df.sort_values("date").copy()
    df = _add_elo(df)
    df = _add_cumulative_win_rates(df)
    df = _add_format_win_rates(df)
    df = _add_streaks(df)
    df = _add_resilience(df)
    df["overall_wr_diff"] = df["team1_overall_wr"] - df["team2_overall_wr"]
    fill_feature_means(df, FEATURE_COLS)
    return df


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------

def _add_elo(
    df: pd.DataFrame,
    k: float = ELO_K,
    initial: float = ELO_INITIAL,
) -> pd.DataFrame:
    """Compute ELO for every team before each match (Elo 400-point scaling)."""
    elo: dict[str, float] = {}
    t1_elo, t2_elo = [], []

    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        r1 = elo.get(t1, initial)
        r2 = elo.get(t2, initial)

        t1_elo.append(r1)
        t2_elo.append(r2)

        e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        s1 = float(row["winner"])

        elo[t1] = r1 + k * (s1 - e1)
        elo[t2] = r2 + k * ((1.0 - s1) - (1.0 - e1))

    df["team1_elo"] = t1_elo
    df["team2_elo"] = t2_elo
    df["elo_diff"] = df["team1_elo"] - df["team2_elo"]
    df["elo_ratio"] = df["team1_elo"] / df["team2_elo"].replace(0, float("nan"))
    return df


# ---------------------------------------------------------------------------
# Cumulative win rates
# ---------------------------------------------------------------------------

def _add_cumulative_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """All-time win rate and total match count per team, using only prior results."""
    wr_t1 = _expanding_wr(df, team_col="team1", winner_val=1)
    wr_t2 = _expanding_wr(df, team_col="team2", winner_val=0)
    cnt_t1 = _expanding_count(df, team_col="team1")
    cnt_t2 = _expanding_count(df, team_col="team2")

    df["team1_overall_wr"] = wr_t1
    df["team2_overall_wr"] = wr_t2
    df["team1_total_matches"] = cnt_t1
    df["team2_total_matches"] = cnt_t2
    return df


def _expanding_wr(df: pd.DataFrame, team_col: str, winner_val: int) -> pd.Series:
    won = (df["winner"] == winner_val).astype(float)
    return (
        won.groupby(df[team_col])
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )


def _expanding_count(df: pd.DataFrame, team_col: str) -> pd.Series:
    ones = pd.Series(1.0, index=df.index)
    return (
        ones.groupby(df[team_col])
        .transform(lambda s: s.shift(1).expanding(min_periods=1).sum())
    )


# ---------------------------------------------------------------------------
# Format-specific win rates (bo1 / bo3)
# ---------------------------------------------------------------------------

def _add_format_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Win rates split by format, using only prior matches in that format."""
    if "format" not in df.columns:
        for col in ["team1_bo1_wr", "team2_bo1_wr", "team1_bo3_wr", "team2_bo3_wr"]:
            df[col] = float("nan")
        return df

    fmt = df["format"].str.lower()
    df["team1_bo1_wr"] = _format_expanding_wr(df, "team1", 1, fmt, "bo1")
    df["team2_bo1_wr"] = _format_expanding_wr(df, "team2", 0, fmt, "bo1")
    df["team1_bo3_wr"] = _format_expanding_wr(df, "team1", 1, fmt, "bo3")
    df["team2_bo3_wr"] = _format_expanding_wr(df, "team2", 0, fmt, "bo3")
    return df


def _format_expanding_wr(
    df: pd.DataFrame, team_col: str, winner_val: int, fmt: pd.Series, target_fmt: str
) -> pd.Series:
    won = (df["winner"] == winner_val).astype(float)
    result = pd.Series(float("nan"), index=df.index)
    for team, group_idx in df.groupby(df[team_col]).groups.items():
        sub = df.loc[group_idx].sort_values("date")
        sub_won = won.loc[group_idx]
        sub_fmt = fmt.loc[group_idx]
        vals = pd.Series(float("nan"), index=sub.index)
        running_sum, running_count = 0.0, 0
        for i, idx in enumerate(sub.index):
            if running_count > 0:
                vals.iloc[i] = running_sum / running_count
            if sub_fmt.loc[idx] == target_fmt:
                running_sum += sub_won.loc[idx]
                running_count += 1
        result.loc[group_idx] = vals
    return result


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------

def _add_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Current win streak and loss streak per team before each match."""
    df["team1_win_streak_all"] = _current_streak(df, "team1", win=True)
    df["team2_win_streak_all"] = _current_streak(df, "team2", win=False)
    df["team1_loss_streak"] = _current_streak(df, "team1", win=False, loss=True)
    df["team2_loss_streak"] = _current_streak(df, "team2", win=True, loss=True)
    return df


def _current_streak(
    df: pd.DataFrame,
    team_col: str,
    win: bool,
    loss: bool = False,
) -> pd.Series:
    """Count consecutive wins (or losses) for each team immediately before each match."""
    if not loss:
        won = (df["winner"] == (1 if team_col == "team1" else 0)).astype(int)
    else:
        won = (df["winner"] == (0 if team_col == "team1" else 1)).astype(int)

    result = pd.Series(0, index=df.index)
    for team, group_idx in df.groupby(df[team_col]).groups.items():
        sub_won = won.loc[group_idx].sort_index()
        streak = 0
        streaks = []
        for v in sub_won.values:
            streaks.append(streak)
            streak = streak + 1 if v == 1 else 0
        result.loc[group_idx] = streaks
    return result


# ---------------------------------------------------------------------------
# Resilience (win rate after a loss)
# ---------------------------------------------------------------------------

def _add_resilience(df: pd.DataFrame, window: int = FORM_WINDOW_SHORT) -> pd.DataFrame:
    """Fraction of the team's post-loss matches that they won (short window)."""
    df["team1_resilience"] = _resilience(df, "team1", winner_val=1, window=window)
    df["team2_resilience"] = _resilience(df, "team2", winner_val=0, window=window)
    return df


def _resilience(
    df: pd.DataFrame,
    team_col: str,
    winner_val: int,
    window: int,
) -> pd.Series:
    won = (df["winner"] == winner_val).astype(int)
    result = pd.Series(float("nan"), index=df.index)
    for _, group_idx in df.groupby(df[team_col]).groups.items():
        sub_won = won.loc[group_idx].sort_index().values
        vals = []
        post_loss_results: list[int] = []
        for i, w in enumerate(sub_won):
            if len(post_loss_results) > 0:
                vals.append(sum(post_loss_results[-window:]) / len(post_loss_results[-window:]))
            else:
                vals.append(float("nan"))
            if i > 0 and sub_won[i - 1] == 0:
                post_loss_results.append(w)
        result.loc[group_idx] = vals
    return result


