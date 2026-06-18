"""Backward-compat shim — all I/O logic moved to utils.file_utils."""

from utils.file_utils import (  # noqa: F401
    concat_years,
    load_csv,
    load_json,
    load_parquet,
    load_pkl,
    save_csv,
    save_json,
    save_parquet,
    save_pkl,
)
