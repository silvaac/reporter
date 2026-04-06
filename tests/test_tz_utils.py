"""Comprehensive tests for tz_utils – the central timezone helper module.

Every helper is tested with naive, UTC-aware, other-tz-aware, empty, and
edge-case inputs to guarantee crash-free behaviour across Python 3.10+ and
pandas 2.x+.
"""

import pytest
from datetime import datetime, timezone, timedelta

import pandas as pd

from tz_utils import (
    ensure_utc,
    ensure_utc_index,
    safe_strip_tz,
    to_eastern,
    utc_now,
)


# ---------------------------------------------------------------------------
# utc_now
# ---------------------------------------------------------------------------

class TestUtcNow:
    def test_returns_aware_datetime(self):
        result = utc_now()
        assert result.tzinfo is not None

    def test_is_utc(self):
        result = utc_now()
        assert result.tzinfo == timezone.utc or str(result.tzinfo) == "UTC"

    def test_close_to_real_time(self):
        before = datetime.now(timezone.utc)
        result = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= result <= after


# ---------------------------------------------------------------------------
# ensure_utc  (scalar datetime)
# ---------------------------------------------------------------------------

class TestEnsureUtc:
    def test_naive_assumed_utc(self):
        dt = datetime(2024, 6, 15, 12, 0, 0)
        result = ensure_utc(dt)
        assert result.tzinfo is not None
        assert result.hour == 12  # unchanged

    def test_already_utc_no_change(self):
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_utc(dt)
        assert result == dt

    def test_other_tz_converted(self):
        eastern = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 15, 7, 0, 0, tzinfo=eastern)
        result = ensure_utc(dt)
        assert result.hour == 12  # 07:00 EST == 12:00 UTC


# ---------------------------------------------------------------------------
# ensure_utc_index  (DataFrame)
# ---------------------------------------------------------------------------

class TestEnsureUtcIndex:
    def test_naive_index_localized(self):
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="ts")
        df = pd.DataFrame({"v": [1, 2]}, index=idx)
        ensure_utc_index(df)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_already_utc_no_op(self):
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC", name="ts")
        df = pd.DataFrame({"v": [1, 2]}, index=idx)
        ensure_utc_index(df)
        assert str(df.index.tz) == "UTC"

    def test_other_tz_converted_to_utc(self):
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="US/Eastern", name="ts")
        df = pd.DataFrame({"v": [1, 2]}, index=idx)
        ensure_utc_index(df)
        assert str(df.index.tz) == "UTC"

    def test_empty_dataframe_no_crash(self):
        df = pd.DataFrame(columns=["v"])
        result = ensure_utc_index(df)
        assert result.empty

    def test_non_datetime_index_no_crash(self):
        df = pd.DataFrame({"v": [1, 2]}, index=[0, 1])
        result = ensure_utc_index(df)
        assert list(result.index) == [0, 1]

    def test_single_element_index(self):
        idx = pd.DatetimeIndex(["2024-06-15"], name="ts")
        df = pd.DataFrame({"v": [1]}, index=idx)
        ensure_utc_index(df)
        assert str(df.index.tz) == "UTC"

    def test_returns_same_dataframe(self):
        idx = pd.DatetimeIndex(["2024-01-01"], name="ts")
        df = pd.DataFrame({"v": [1]}, index=idx)
        result = ensure_utc_index(df)
        assert result is df  # in-place modification

    def test_double_call_does_not_crash(self):
        """Calling ensure_utc_index twice must not raise (double-localize bug)."""
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="ts")
        df = pd.DataFrame({"v": [1, 2]}, index=idx)
        ensure_utc_index(df)
        ensure_utc_index(df)  # should be a no-op
        assert str(df.index.tz) == "UTC"


# ---------------------------------------------------------------------------
# safe_strip_tz
# ---------------------------------------------------------------------------

class TestSafeStripTz:
    def test_aware_stripped(self):
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC")
        result = safe_strip_tz(idx)
        assert result.tz is None

    def test_naive_no_op(self):
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
        result = safe_strip_tz(idx)
        assert result.tz is None
        assert len(result) == 2

    def test_non_datetime_index_passthrough(self):
        idx = pd.Index([0, 1, 2])
        result = safe_strip_tz(idx)
        assert list(result) == [0, 1, 2]

    def test_double_strip_does_not_crash(self):
        idx = pd.DatetimeIndex(["2024-01-01"], tz="UTC")
        result = safe_strip_tz(idx)
        result2 = safe_strip_tz(result)
        assert result2.tz is None


# ---------------------------------------------------------------------------
# to_eastern
# ---------------------------------------------------------------------------

class TestToEastern:
    def test_winter_utc_to_est(self):
        """UTC 12:00 on a winter day -> EST 07:00 (UTC-5)."""
        ts = pd.Timestamp("2024-01-15 12:00:00", tz="UTC")
        result = to_eastern(ts)
        assert result.hour == 7
        assert result.utcoffset() == timedelta(hours=-5)

    def test_summer_utc_to_edt(self):
        """UTC 12:00 on a summer day -> EDT 08:00 (UTC-4)."""
        ts = pd.Timestamp("2024-07-15 12:00:00", tz="UTC")
        result = to_eastern(ts)
        assert result.hour == 8
        assert result.utcoffset() == timedelta(hours=-4)

    def test_naive_timestamp_assumed_utc(self):
        """Naive input is assumed UTC before conversion."""
        ts = pd.Timestamp("2024-01-15 12:00:00")
        result = to_eastern(ts)
        assert result.hour == 7

    def test_stdlib_datetime_winter(self):
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = to_eastern(dt)
        assert result.hour == 7

    def test_stdlib_datetime_summer(self):
        dt = datetime(2024, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = to_eastern(dt)
        assert result.hour == 8

    def test_naive_stdlib_datetime(self):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        result = to_eastern(dt)
        assert result.hour == 7
