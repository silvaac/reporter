"""Centralized timezone utilities for portable datetime handling.

All internal timestamps in this project should be UTC-aware. This module
provides safe helpers that handle naive, already-aware, and mixed inputs
without crashing—regardless of Python (3.10+) or pandas (2.x+) version.

No external dependencies beyond the standard library and pandas.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Union

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9

logger = logging.getLogger(__name__)

# Pre-built timezone objects
UTC = timezone.utc
EASTERN = ZoneInfo("US/Eastern")


def utc_now() -> datetime:
    """Return the current time as a UTC-aware datetime.

    Replaces all uses of ``datetime.now()`` and the deprecated
    ``datetime.utcnow()`` throughout the codebase.
    """
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Return *dt* as a UTC-aware datetime.

    - If *dt* is naive, it is assumed to already represent UTC and is
      localized accordingly.
    - If *dt* is aware but in another timezone, it is converted to UTC.
    - If *dt* is already UTC-aware, it is returned unchanged.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure *df* has a UTC-aware ``DatetimeIndex``.

    Handles every combination safely:

    * **Naive index** → localized to UTC.
    * **Already UTC** → no-op.
    * **Other timezone** → converted to UTC.
    * **Not a DatetimeIndex / empty** → returned as-is.

    The dataframe is modified **in-place** for efficiency and also returned.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize("UTC")
        except TypeError:
            # Edge-case: mixed naive/aware after concat – force conversion
            df.index = pd.to_datetime(df.index, utc=True)
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    return df


def safe_strip_tz(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Remove timezone info from *index* without crashing.

    - If already naive, returns as-is.
    - If tz-aware, calls ``tz_localize(None)``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return index
    if index.tz is None:
        return index
    return index.tz_localize(None)


def to_eastern(ts: Union[pd.Timestamp, datetime]) -> Union[pd.Timestamp, datetime]:
    """Convert a UTC-aware timestamp to US/Eastern (handles EST/EDT).

    If *ts* is naive, it is assumed UTC before conversion.
    Never falls back to a hard-coded offset.
    """
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("US/Eastern")

    # stdlib datetime
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(EASTERN)
