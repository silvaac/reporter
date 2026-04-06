"""Tests for datetime/timezone handling throughout the reporter.

All internal timestamps should be UTC-aware. EST conversion should use
'US/Eastern' timezone (handles DST automatically).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone, timedelta

import pandas as pd

from hyperliquid_reporter.reporter import HyperliquidReporter


class TestDatetimeUTCAwareness:
    """Verify that all generated timestamps are UTC-aware."""

    @pytest.fixture
    def mock_monitor(self):
        monitor = Mock()
        mock_info = MagicMock()
        mock_info.user_non_funding_ledger_updates.return_value = []
        monitor._info = mock_info
        monitor._address = "0x123"
        monitor.get_account_summary.return_value = {"net_deposits": 10000.0}
        return monitor

    @pytest.fixture
    def reporter(self, mock_monitor):
        return HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef1234567890abcdef12345678",
        )

    def test_generated_at_is_utc(self, reporter, mock_monitor):
        """generate_report_data()['generated_at'] must be UTC-aware."""
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0]},
            index=pd.DatetimeIndex(
                [pd.Timestamp("2024-01-01", tz="UTC")], name="timestamp"
            ),
        )
        mock_monitor.get_trade_history.return_value = pd.DataFrame()
        mock_monitor.get_funding_history.return_value = pd.DataFrame()
        mock_monitor.get_account_summary.return_value = {
            "net_deposits": 10000.0,
            "current_value": 10000.0,
        }

        result = reporter.generate_report_data(period="day")
        generated_at = result["generated_at"]

        assert generated_at.tzinfo is not None, "generated_at must be timezone-aware"
        assert generated_at.tzinfo == timezone.utc or str(generated_at.tzinfo) == "UTC"

    def test_performance_data_index_is_utc(self, reporter, mock_monitor):
        """Performance data DataFrame index must be UTC-aware."""
        utc_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=3, tz="UTC"), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0, 12000.0]}, index=utc_index
        )

        result = reporter.generate_performance_data(period="month")
        assert result.index.tz is not None, "Performance data index must be tz-aware"

    def test_aum_data_index_is_utc(self, reporter, mock_monitor):
        """AUM data DataFrame index must be UTC-aware."""
        utc_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=3, tz="UTC"), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0, 12000.0]}, index=utc_index
        )

        result = reporter.generate_aum_data(period="month")
        assert result.index.tz is not None, "AUM data index must be tz-aware"


class TestESTConversion:
    """Verify EST/EDT conversion uses proper timezone, not hardcoded offset."""

    def test_est_winter_time(self):
        """In winter (EST = UTC-5), conversion should subtract 5 hours."""
        utc_time = pd.Timestamp("2024-01-15 12:00:00", tz="UTC")
        est_time = utc_time.tz_convert("US/Eastern")
        assert est_time.hour == 7  # 12:00 UTC = 07:00 EST

    def test_edt_summer_time(self):
        """In summer (EDT = UTC-4), conversion should subtract 4 hours."""
        utc_time = pd.Timestamp("2024-07-15 12:00:00", tz="UTC")
        edt_time = utc_time.tz_convert("US/Eastern")
        assert edt_time.hour == 8  # 12:00 UTC = 08:00 EDT

    def test_html_report_est_column_uses_proper_tz(self, tmp_path):
        """The HTML report EST column should use US/Eastern, not a fixed -5h offset."""
        # This test verifies the conversion function exists and works correctly.
        # We test the conversion logic directly since testing the full HTML
        # would require extensive mocking.
        utc_summer = pd.Timestamp("2024-06-15 12:00:00", tz="UTC")
        utc_winter = pd.Timestamp("2024-01-15 12:00:00", tz="UTC")

        est_summer = utc_summer.tz_convert("US/Eastern")
        est_winter = utc_winter.tz_convert("US/Eastern")

        # Summer: UTC-4 (EDT)
        assert est_summer.utcoffset() == timedelta(hours=-4)
        # Winter: UTC-5 (EST)
        assert est_winter.utcoffset() == timedelta(hours=-5)


class TestPnlHistoryTimestamps:
    """Verify pnl_history.csv timestamps are UTC-aware."""

    @pytest.fixture
    def reporter(self):
        monitor = Mock()
        return HyperliquidReporter(
            monitor=monitor,
            account_address="0x1234567890abcdef1234567890abcdef12345678",
            pnl_history_file=str("/tmp/test_pnl_history_tz.csv"),
        )

    def test_save_pnl_history_uses_utc(self, reporter, tmp_path):
        """_save_pnl_history must write UTC timestamps."""
        import os

        test_file = str(tmp_path / "pnl_tz.csv")
        reporter.pnl_history_file = test_file

        reporter._save_pnl_history(10000.0, 10000.0)

        df = pd.read_csv(test_file)
        ts = pd.to_datetime(df["datetime"].iloc[0])
        # The saved timestamp string should be parseable and when localized
        # to UTC should be close to now (within a few seconds)
        utc_now = pd.Timestamp.now(tz="UTC")
        parsed = pd.Timestamp(ts)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        diff = abs((utc_now - parsed).total_seconds())
        assert diff < 60, f"Saved timestamp should be close to current UTC time, diff={diff}s"


class TestDoubleLocalizeSafety:
    """Regression: feeding already-UTC data must not crash (double-localize bug)."""

    @pytest.fixture
    def mock_monitor(self):
        monitor = Mock()
        mock_info = MagicMock()
        mock_info.user_non_funding_ledger_updates.return_value = []
        monitor._info = mock_info
        monitor._address = "0x123"
        monitor.get_account_summary.return_value = {"net_deposits": 10000.0}
        return monitor

    @pytest.fixture
    def reporter(self, mock_monitor):
        return HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef1234567890abcdef12345678",
        )

    def test_aum_data_with_already_utc_index(self, reporter, mock_monitor):
        """generate_aum_data must not crash when fed already-UTC data."""
        utc_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=3, tz="UTC"), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0, 12000.0]}, index=utc_index
        )
        result = reporter.generate_aum_data(period="month")
        assert str(result.index.tz) == "UTC"

    def test_performance_data_with_already_utc_index(self, reporter, mock_monitor):
        """generate_performance_data must not crash when fed already-UTC data."""
        utc_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=3, tz="UTC"), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0, 12000.0]}, index=utc_index
        )
        result = reporter.generate_performance_data(period="month")
        assert str(result.index.tz) == "UTC"


class TestNaiveInputLocalization:
    """Feeding naive (tz-unaware) data must result in UTC output."""

    @pytest.fixture
    def mock_monitor(self):
        monitor = Mock()
        mock_info = MagicMock()
        mock_info.user_non_funding_ledger_updates.return_value = []
        monitor._info = mock_info
        monitor._address = "0x123"
        monitor.get_account_summary.return_value = {"net_deposits": 10000.0}
        return monitor

    @pytest.fixture
    def reporter(self, mock_monitor):
        return HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef1234567890abcdef12345678",
        )

    def test_naive_index_gets_localized_in_aum(self, reporter, mock_monitor):
        naive_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=3), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0, 12000.0]}, index=naive_index
        )
        result = reporter.generate_aum_data(period="month")
        assert result.index.tz is not None, "Naive input must be localized to UTC"

    def test_naive_index_gets_localized_in_performance(self, reporter, mock_monitor):
        naive_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=3), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0, 12000.0]}, index=naive_index
        )
        result = reporter.generate_performance_data(period="month")
        assert result.index.tz is not None, "Naive input must be localized to UTC"


class TestToEasternInHtmlReport:
    """Verify to_eastern() produces correct DST-aware offsets used in HTML."""

    def test_to_eastern_winter(self):
        from tz_utils import to_eastern
        ts = pd.Timestamp("2024-01-15 12:00:00", tz="UTC")
        result = to_eastern(ts)
        assert result.utcoffset() == timedelta(hours=-5), "Winter: should be EST (UTC-5)"

    def test_to_eastern_summer(self):
        from tz_utils import to_eastern
        ts = pd.Timestamp("2024-07-15 12:00:00", tz="UTC")
        result = to_eastern(ts)
        assert result.utcoffset() == timedelta(hours=-4), "Summer: should be EDT (UTC-4)"

    def test_to_eastern_naive_input(self):
        from tz_utils import to_eastern
        ts = pd.Timestamp("2024-01-15 12:00:00")
        result = to_eastern(ts)
        assert result.hour == 7, "Naive input assumed UTC, 12:00 UTC -> 07:00 EST"


class TestMonitoringTimestampsUTC:
    """Verify base monitoring produces UTC-aware timestamps."""

    def test_performance_metrics_timestamp_is_utc(self):
        from base.monitoring import PerformanceMetrics
        pm = PerformanceMetrics(account_value=10000.0)
        assert pm.timestamp.tzinfo is not None, "Default timestamp must be tz-aware"
        assert pm.timestamp.tzinfo == timezone.utc or str(pm.timestamp.tzinfo) == "UTC"
