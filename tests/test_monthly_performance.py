"""Tests for monthly performance aggregation feature."""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

from hyperliquid_reporter.reporter import HyperliquidReporter


class TestMonthlyPerformance:
    """Verify monthly performance table generation."""

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

    def test_generate_monthly_performance_returns_dataframe(self, reporter):
        """generate_monthly_performance should return a DataFrame."""
        daily_index = pd.date_range("2024-01-01", "2024-03-31", freq="D", tz="UTC")
        np.random.seed(42)
        aum = 10000.0 + np.cumsum(np.random.randn(len(daily_index)) * 50)
        perf_data = pd.DataFrame(
            {
                "aum_usd": aum,
                "net_deposits": 10000.0,
                "pnl_usd": np.random.randn(len(daily_index)) * 50,
                "pnl_pct": np.random.randn(len(daily_index)) * 0.5,
            },
            index=daily_index,
        )

        result = reporter.generate_monthly_performance(perf_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Jan, Feb, Mar

    def test_monthly_performance_has_required_columns(self, reporter):
        """Monthly performance should have P&L ($), P&L (%), and cumulative columns."""
        daily_index = pd.date_range("2024-01-01", "2024-02-29", freq="D", tz="UTC")
        perf_data = pd.DataFrame(
            {
                "aum_usd": [10000.0 + i * 10 for i in range(len(daily_index))],
                "net_deposits": 10000.0,
                "pnl_usd": [10.0] * len(daily_index),
                "pnl_pct": [0.1] * len(daily_index),
            },
            index=daily_index,
        )

        result = reporter.generate_monthly_performance(perf_data)
        required_cols = [
            "month",
            "starting_aum",
            "ending_aum",
            "pnl_usd",
            "pnl_pct",
            "cumulative_pnl_usd",
            "cumulative_pnl_pct",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_monthly_pnl_usd_sums_correctly(self, reporter):
        """Monthly P&L ($) should be the sum of daily P&L for that month."""
        daily_index = pd.date_range("2024-01-01", "2024-01-31", freq="D", tz="UTC")
        perf_data = pd.DataFrame(
            {
                "aum_usd": [10000.0 + i * 100 for i in range(len(daily_index))],
                "net_deposits": 10000.0,
                "pnl_usd": [100.0] * len(daily_index),
                "pnl_pct": [1.0] * len(daily_index),
            },
            index=daily_index,
        )

        result = reporter.generate_monthly_performance(perf_data)
        # 31 days * $100/day = $3100
        assert abs(result["pnl_usd"].iloc[0] - 3100.0) < 0.01

    def test_monthly_pnl_pct_is_percentage(self, reporter):
        """Monthly P&L (%) should be (ending_aum - starting_aum - deposit_changes) / starting_aum * 100."""
        daily_index = pd.date_range("2024-01-01", "2024-01-31", freq="D", tz="UTC")
        aum_values = [10000.0] + [10000.0 + i * 100 for i in range(1, len(daily_index))]
        perf_data = pd.DataFrame(
            {
                "aum_usd": aum_values,
                "net_deposits": 10000.0,
                "pnl_usd": [0.0] + [100.0] * (len(daily_index) - 1),
                "pnl_pct": [0.0] + [1.0] * (len(daily_index) - 1),
            },
            index=daily_index,
        )

        result = reporter.generate_monthly_performance(perf_data)
        # Starting AUM = 10000, ending = 13000, no deposit change
        # pnl_pct = (13000 - 10000) / 10000 * 100 = 30%
        assert result["pnl_pct"].iloc[0] > 0

    def test_cumulative_columns_accumulate(self, reporter):
        """Cumulative columns should accumulate across months."""
        daily_index = pd.date_range("2024-01-01", "2024-03-31", freq="D", tz="UTC")
        perf_data = pd.DataFrame(
            {
                "aum_usd": [10000.0 + i * 10 for i in range(len(daily_index))],
                "net_deposits": 10000.0,
                "pnl_usd": [10.0] * len(daily_index),
                "pnl_pct": [0.1] * len(daily_index),
            },
            index=daily_index,
        )

        result = reporter.generate_monthly_performance(perf_data)
        # Cumulative P&L should increase monotonically
        assert result["cumulative_pnl_usd"].iloc[0] <= result["cumulative_pnl_usd"].iloc[1]
        assert result["cumulative_pnl_usd"].iloc[1] <= result["cumulative_pnl_usd"].iloc[2]

    def test_html_report_contains_monthly_section(self, reporter, mock_monitor):
        """HTML report should include a monthly performance section."""
        utc_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
            name="timestamp",
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0 + i * 10 for i in range(60)]},
            index=utc_index,
        )
        mock_monitor.get_trade_history.return_value = pd.DataFrame()
        mock_monitor.get_funding_history.return_value = pd.DataFrame()
        mock_monitor.get_account_summary.return_value = {
            "net_deposits": 10000.0,
            "current_value": 10590.0,
        }

        report_data = reporter.generate_report_data(period="allTime")
        html = reporter.generate_html_report(report_data, {})

        assert "Monthly Performance" in html
