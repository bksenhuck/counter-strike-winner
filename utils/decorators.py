import functools
import time
from typing import Callable

import pandas as pd

from utils.logger import get_logger

_logger = get_logger(__name__)


def log_call(func: Callable) -> Callable:
    """Log function entry and exit with arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _logger.debug("-> %s called with args=%s kwargs=%s", func.__qualname__, args, kwargs)
        result = func(*args, **kwargs)
        _logger.debug("<- %s returned", func.__qualname__)
        return result
    return wrapper


def timer(func: Callable) -> Callable:
    """Log elapsed wall-clock time for a function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _logger.info("%s finished in %.3fs", func.__qualname__, elapsed)
        return result
    return wrapper


def validate_dataframe(*required_cols: str):
    """Raise ValueError if the first DataFrame argument is missing required columns."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            df = next((a for a in args if isinstance(a, pd.DataFrame)), None)
            if df is None:
                df = next(
                    (v for v in kwargs.values() if isinstance(v, pd.DataFrame)), None
                )
            if df is not None:
                missing = set(required_cols) - set(df.columns)
                if missing:
                    raise ValueError(
                        f"{func.__qualname__}: DataFrame missing columns {missing}"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator
