"""Tests for resampling P&L time series to uniform daily intervals."""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd

from hyperliquid_reporter.reporter import HyperliquidReporter


class TestDailyResample:
    """Verify performance data is resampled to uniform daily intervals."""

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

    def test_performance_data_has_daily_frequency(self, reporter, mock_monitor):
        """After resampling, performance data index should have daily frequency."""
        # Create non-uniform timestamps (gaps of varying sizes)
        timestamps = pd.DatetimeIndex(
            [
                "2024-01-01 00:00:00",
                "2024-01-01 06:00:00",
                "2024-01-02 12:00:00",
                "2024-01-05 08:00:00",  # 3-day gap
            ],
            tz="UTC",
            name="timestamp",
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 10100.0, 10200.0, 10500.0]},
            index=timestamps,
        )

        result = reporter.generate_performance_data(period="allTime")

        # Should have 5 rows (Jan 1 through Jan 5, daily)
        assert len(result) == 5, f"Expected 5 daily rows, got {len(result)}"

        # Index should be daily
        diffs = pd.Series(result.index).diff().dropna()
        for d in diffs:
            assert d == pd.Timedelta(days=1), f"Expected 1-day interval, got {d}"

    def test_resample_preserves_aum_values(self, reporter, mock_monitor):
        """Forward-filled AUM values should be correct after resampling."""
        timestamps = pd.DatetimeIndex(
            ["2024-01-01 10:00:00", "2024-01-03 14:00:00"],
            tz="UTC",
            name="timestamp",
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 10500.0]}, index=timestamps
        )

        result = reporter.generate_performance_data(period="allTime")

        # Jan 1: 10000, Jan 2: forward-filled 10000, Jan 3: 10500
        assert len(result) == 3
        assert result["aum_usd"].iloc[0] == 10000.0
        assert result["aum_usd"].iloc[1] == 10000.0  # forward-filled
        assert result["aum_usd"].iloc[2] == 10500.0

    def test_resample_single_day_data(self, reporter, mock_monitor):
        """If all data is from the same day, result should have 1 row."""
        timestamps = pd.DatetimeIndex(
            ["2024-01-01 08:00:00", "2024-01-01 16:00:00", "2024-01-01 23:00:00"],
            tz="UTC",
            name="timestamp",
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 10100.0, 10200.0]}, index=timestamps
        )

        result = reporter.generate_performance_data(period="allTime")

        # All same day -> 1 row (last value of the day)
        assert len(result) == 1
        assert result["aum_usd"].iloc[0] == 10200.0
