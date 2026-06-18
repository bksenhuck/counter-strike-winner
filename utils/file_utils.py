"""Centralised file I/O utilities.

Supports: Parquet, Pickle (.pkl), JSON, CSV.
All functions log the operation and raise descriptive errors on failure.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from utils.logger import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a single parquet file and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    _logger.info("Loading parquet: %s", path)
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    """Persist a DataFrame as parquet; create parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    _logger.info("Saved parquet (%d rows) -> %s", len(df), path)
    return path


def concat_years(base_dir: Path | str, filename: str) -> pd.DataFrame:
    """Concatenate one parquet file across all year sub-directories.

    Args:
        base_dir: Root directory containing year-named sub-folders.
        filename: Parquet filename to load from each year folder.

    Returns:
        Concatenated DataFrame across all years, sorted by folder name.

    Raises:
        FileNotFoundError: When no matching files are found.
    """
    base_dir = Path(base_dir)
    frames = []
    for year_dir in sorted(base_dir.iterdir()):
        if year_dir.is_dir():
            fp = year_dir / filename
            if fp.exists():
                frames.append(load_parquet(fp))
    if not frames:
        raise FileNotFoundError(f"No '{filename}' files found under {base_dir}")
    df = pd.concat(frames, ignore_index=True)
    _logger.info(
        "Concatenated %d year(s) for '%s' -> %d rows", len(frames), filename, len(df)
    )
    return df


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def load_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    _logger.info("Loading CSV: %s", path)
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path | str, index: bool = False, **kwargs) -> Path:
    """Persist a DataFrame as CSV; create parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)
    _logger.info("Saved CSV (%d rows) -> %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------------

def load_pkl(path: Path | str) -> Any:
    """Load any Python object from a pickle file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    _logger.info("Loading pickle: %s", path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj: Any, path: Path | str) -> Path:
    """Persist any Python object as a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    _logger.info("Saved pickle -> %s", path)
    return path


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def load_json(path: Path | str) -> Any:
    """Load a JSON file and return the parsed Python object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    _logger.info("Loading JSON: %s", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path | str, indent: int = 2) -> Path:
    """Persist a JSON-serialisable object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    _logger.info("Saved JSON -> %s", path)
    return path
