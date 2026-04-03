"""Tests for net deposits carry-forward logic.

When net_deposits drops to 0 but previous snapshot had deposits > 0,
the last known net_deposits value should be carried forward instead.
"""

import pytest
import pandas as pd
from unittest.mock import Mock
from pathlib import Path

from hyperliquid_reporter.reporter import HyperliquidReporter


class TestNetDepositsCarryForward:
    """Verify net_deposits=0 is replaced with last known value."""

    @pytest.fixture
    def reporter(self, tmp_path):
        monitor = Mock()
        pnl_file = str(tmp_path / "pnl_history.csv")
        return HyperliquidReporter(
            monitor=monitor,
            account_address="0x1234567890abcdef1234567890abcdef12345678",
            pnl_history_file=pnl_file,
        )

    def test_carry_forward_when_net_deposits_zero(self, reporter):
        """If net_deposits=0 but previous was >0, carry forward."""
        # First save: normal deposit
        reporter._save_pnl_history(10000.0, 10000.0)
        # Second save: net_deposits drops to 0 (bug scenario)
        reporter._save_pnl_history(682.0, 0.0)

        df = pd.read_csv(reporter.pnl_history_file)
        # The second row should carry forward 10000.0, not 0.0
        assert df["net_deposits"].iloc[1] == 10000.0, (
            f"Expected net_deposits=10000.0 (carried forward), got {df['net_deposits'].iloc[1]}"
        )

    def test_allow_zero_when_no_previous(self, reporter):
        """If there's no previous snapshot, net_deposits=0 is acceptable."""
        reporter._save_pnl_history(0.0, 0.0)

        df = pd.read_csv(reporter.pnl_history_file)
        assert df["net_deposits"].iloc[0] == 0.0

    def test_allow_legitimate_decrease(self, reporter):
        """If net_deposits decreases but not to 0, allow it (withdrawal)."""
        reporter._save_pnl_history(10000.0, 10000.0)
        reporter._save_pnl_history(5000.0, 5000.0)  # withdrawal

        df = pd.read_csv(reporter.pnl_history_file)
        assert df["net_deposits"].iloc[1] == 5000.0

    def test_load_pnl_history_correct_with_carryforward(self, reporter):
        """P&L calculation from file should be correct after carry-forward."""
        reporter._save_pnl_history(10000.0, 10000.0)
        reporter._save_pnl_history(10500.0, 10000.0)  # $500 profit

        df = reporter._load_pnl_history()
        assert len(df) == 2
        # First row P&L is always 0
        assert df["pnl_usd"].iloc[0] == 0.0
        # Second row: AUM went from 10000 to 10500, no deposit change -> $500 P&L
        assert abs(df["pnl_usd"].iloc[1] - 500.0) < 0.01

    def test_pnl_not_inflated_by_zero_deposits(self, reporter):
        """The original bug: net_deposits going to 0 inflates P&L massively."""
        reporter._save_pnl_history(10000.0, 10000.0)
        # Bug scenario: AUM drops, net_deposits erroneously reported as 0
        reporter._save_pnl_history(682.0, 0.0)

        df = reporter._load_pnl_history()
        # With carry-forward fix, net_deposits stays at 10000.0
        # P&L = (682 - 10000) - (10000 - 10000) = -9318.0
        # Without fix: P&L = (682 - 10000) - (0 - 10000) = 682.0 (wrong!)
        assert df["pnl_usd"].iloc[1] < 0, (
            f"P&L should be negative (loss), got {df['pnl_usd'].iloc[1]}"
        )
        assert abs(df["pnl_usd"].iloc[1] - (-9318.0)) < 1.0
