"""Map-level features: dominance and control derived from per-map scores."""

import pandas as pd

REQUIRED_COLS: list[str] = ["maps_won_team1", "maps_won_team2", "total_maps"]

FEATURE_COLS: list[str] = [
    "maps_won_team1",
    "maps_won_team2",
    "total_maps",
    "map_win_rate",
    "maps_advantage",
]


def add_map_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add map-dominance columns to `df`.

    Features added:
        map_win_rate   — maps_won_team1 / total_maps  (team1 win fraction in series)
        maps_advantage — maps_won_team1 - maps_won_team2 (net map lead)
    """
    df = df.copy()
    df["map_win_rate"] = df["maps_won_team1"] / df["total_maps"].replace(0, float("nan"))
    df["maps_advantage"] = df["maps_won_team1"] - df["maps_won_team2"]
    return df
