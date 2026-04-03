"""Tests for P&L display units — no double conversion, columns labeled '%' not 'bp'."""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd

from hyperliquid_reporter.reporter import HyperliquidReporter


class TestPnlDisplayUnits:
    """Verify P&L percentage is not double-converted and columns use '%'."""

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

    def test_pnl_pct_not_double_converted(self, reporter, mock_monitor):
        """pnl_pct from generate_performance_data should be in % (e.g. 10.0 = 10%).

        The HTML rendering should NOT multiply by 100 again.
        """
        utc_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=2, tz="UTC"), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0]}, index=utc_index
        )

        result = reporter.generate_performance_data(period="month")
        # Day 2: $1000 profit on $10000 = 10%
        pnl_pct = result["pnl_pct"].iloc[1]
        assert 9.0 < pnl_pct < 11.0, (
            f"pnl_pct should be ~10.0 (meaning 10%), got {pnl_pct}"
        )

    def test_html_report_no_bp_columns(self, reporter, mock_monitor):
        """HTML report should not contain any columns labeled 'bp'."""
        utc_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=3, tz="UTC"), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0, 12000.0]}, index=utc_index
        )
        mock_monitor.get_trade_history.return_value = pd.DataFrame()
        mock_monitor.get_funding_history.return_value = pd.DataFrame()
        mock_monitor.get_account_summary.return_value = {
            "net_deposits": 10000.0,
            "current_value": 12000.0,
        }

        report_data = reporter.generate_report_data(period="day")
        visualizations = {}  # skip charts for this test

        html = reporter.generate_html_report(report_data, visualizations)

        assert "P&amp;L (bp)" not in html, "HTML should not have 'P&L (bp)' columns"
        assert "P&L (bp)" not in html, "HTML should not have 'P&L (bp)' columns"
        # Should have % columns instead
        assert "(%)" in html, "HTML should have '%' columns"

    def test_html_cumulative_pnl_pct_values_are_reasonable(self, reporter, mock_monitor):
        """Cumulative P&L % displayed in HTML should be reasonable (not 100x too large)."""
        utc_index = pd.DatetimeIndex(
            pd.date_range("2024-01-01", periods=3, tz="UTC"), name="timestamp"
        )
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame(
            {"account_value": [10000.0, 11000.0, 12100.0]}, index=utc_index
        )
        mock_monitor.get_trade_history.return_value = pd.DataFrame()
        mock_monitor.get_funding_history.return_value = pd.DataFrame()
        mock_monitor.get_account_summary.return_value = {
            "net_deposits": 10000.0,
            "current_value": 12100.0,
        }

        report_data = reporter.generate_report_data(period="day")
        perf = report_data["performance_data"]

        # Cumulative P&L %: day1=0, day2=10%, day3=10% -> cumulative ~20%
        cum_pct = perf["pnl_pct"].cumsum()
        assert cum_pct.iloc[-1] < 100, (
            f"Cumulative pnl_pct should be ~20%, got {cum_pct.iloc[-1]}"
        )
