"""Tests for position summary fields in account summary and summary stats.

Assumption under test: if there is both a spot and a perp position, they are
assumed to be for the same underlying token.  The net signed position is the
sum of the spot token balance and the signed perp position size.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, Mock, patch
import numpy as np
import pandas as pd
import pytest

from hyperliquid_reporter.reporter import HyperliquidReporter


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_reporter(mock_monitor=None):
    if mock_monitor is None:
        mock_monitor = Mock()
    return HyperliquidReporter(
        monitor=mock_monitor,
        account_address="0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    )


def _make_price_series(n_days: int = 60, last_price: float = 3000.0, seed: int = 42) -> pd.Series:
    """Generate a synthetic daily close-price series ending at last_price."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.02, n_days - 1)
    prices = [last_price * math.exp(-sum(log_returns))]
    for r in log_returns:
        prices.append(prices[-1] * math.exp(r))
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_days, freq="D")
    return pd.Series(prices, index=idx)


# ---------------------------------------------------------------------------
# Tests for get_positions_summary (new monitor-level helper)
# ---------------------------------------------------------------------------

class TestGetPositionsSummary:
    """Unit tests for HyperliquidMonitor.get_positions_summary."""

    def _make_monitor(self):
        from hyperliquid_reporter.monitoring import HyperliquidMonitor
        monitor = Mock(spec=HyperliquidMonitor)
        monitor._info = MagicMock()
        monitor._address = "0xabc"
        # Delegate the real method under test
        monitor.get_positions_summary = lambda: HyperliquidMonitor.get_positions_summary(monitor)
        return monitor

    def test_no_positions_returns_zeros(self):
        monitor = self._make_monitor()
        monitor._info.user_state.return_value = {"assetPositions": []}
        monitor._info.spot_user_state.return_value = {"balances": [
            {"coin": "USDC", "total": "10000.0", "entryNtl": "0"}
        ]}

        result = monitor.get_positions_summary()

        assert result["perp_position"] == 0.0
        assert result["spot_position"] == 0.0
        assert result["net_position"] == 0.0
        assert result["position_token"] is None

    def test_perp_only_long(self):
        monitor = self._make_monitor()
        monitor._info.user_state.return_value = {
            "assetPositions": [
                {"position": {"coin": "ETH", "szi": "2.5", "positionValue": "7500"}}
            ]
        }
        monitor._info.spot_user_state.return_value = {"balances": [
            {"coin": "USDC", "total": "5000.0", "entryNtl": "0"}
        ]}

        result = monitor.get_positions_summary()

        assert result["position_token"] == "ETH"
        assert result["perp_position"] == pytest.approx(2.5)
        assert result["spot_position"] == pytest.approx(0.0)
        assert result["net_position"] == pytest.approx(2.5)

    def test_perp_only_short(self):
        monitor = self._make_monitor()
        monitor._info.user_state.return_value = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "-0.1", "positionValue": "5000"}}
            ]
        }
        monitor._info.spot_user_state.return_value = {"balances": [
            {"coin": "USDC", "total": "10000", "entryNtl": "0"}
        ]}

        result = monitor.get_positions_summary()

        assert result["position_token"] == "BTC"
        assert result["perp_position"] == pytest.approx(-0.1)
        assert result["spot_position"] == pytest.approx(0.0)
        assert result["net_position"] == pytest.approx(-0.1)

    def test_spot_only(self):
        monitor = self._make_monitor()
        monitor._info.user_state.return_value = {"assetPositions": []}
        monitor._info.spot_user_state.return_value = {"balances": [
            {"coin": "USDC", "total": "5000", "entryNtl": "0"},
            {"coin": "ETH", "total": "3.0", "entryNtl": "6000"},
        ]}

        result = monitor.get_positions_summary()

        assert result["position_token"] == "ETH"
        assert result["spot_position"] == pytest.approx(3.0)
        assert result["perp_position"] == pytest.approx(0.0)
        assert result["net_position"] == pytest.approx(3.0)

    def test_spot_and_perp_same_token(self):
        """Spot long + perp long → net = sum."""
        monitor = self._make_monitor()
        monitor._info.user_state.return_value = {
            "assetPositions": [
                {"position": {"coin": "ETH", "szi": "1.0", "positionValue": "3000"}}
            ]
        }
        monitor._info.spot_user_state.return_value = {"balances": [
            {"coin": "USDC", "total": "1000", "entryNtl": "0"},
            {"coin": "ETH", "total": "2.0", "entryNtl": "4000"},
        ]}

        result = monitor.get_positions_summary()

        assert result["position_token"] == "ETH"
        assert result["spot_position"] == pytest.approx(2.0)
        assert result["perp_position"] == pytest.approx(1.0)
        assert result["net_position"] == pytest.approx(3.0)

    def test_spot_long_perp_short_partial_hedge(self):
        """Spot long 2 ETH, perp short -1 ETH → net = 1 ETH."""
        monitor = self._make_monitor()
        monitor._info.user_state.return_value = {
            "assetPositions": [
                {"position": {"coin": "ETH", "szi": "-1.0", "positionValue": "3000"}}
            ]
        }
        monitor._info.spot_user_state.return_value = {"balances": [
            {"coin": "ETH", "total": "2.0", "entryNtl": "6000"},
        ]}

        result = monitor.get_positions_summary()

        assert result["net_position"] == pytest.approx(1.0)

    def test_spot_api_failure_returns_zero_spot(self):
        """If spot API fails, spot_position should default to 0."""
        monitor = self._make_monitor()
        monitor._info.user_state.return_value = {
            "assetPositions": [
                {"position": {"coin": "ETH", "szi": "1.5", "positionValue": "4500"}}
            ]
        }
        monitor._info.spot_user_state.side_effect = Exception("API error")

        result = monitor.get_positions_summary()

        assert result["spot_position"] == pytest.approx(0.0)
        assert result["perp_position"] == pytest.approx(1.5)
        assert result["net_position"] == pytest.approx(1.5)

    def test_perp_api_failure_returns_zero_perp(self):
        """If perp API fails, perp_position should default to 0."""
        monitor = self._make_monitor()
        monitor._info.user_state.side_effect = Exception("API error")
        monitor._info.spot_user_state.return_value = {"balances": [
            {"coin": "ETH", "total": "1.0", "entryNtl": "3000"},
        ]}

        result = monitor.get_positions_summary()

        assert result["perp_position"] == pytest.approx(0.0)
        assert result["spot_position"] == pytest.approx(1.0)
        assert result["net_position"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests for calculate_net_position_usd (reporter-level helper)
# ---------------------------------------------------------------------------

class TestCalculateNetPositionMetrics:
    """Unit tests for HyperliquidReporter._calculate_net_position_metrics."""

    @pytest.fixture
    def reporter(self):
        return _make_reporter()

    def test_zero_net_position_returns_zeros(self, reporter):
        positions = {
            "position_token": "ETH",
            "net_position": 0.0,
            "spot_position": 0.0,
            "perp_position": 0.0,
        }
        prices = _make_price_series(60, last_price=3000.0)
        result = reporter._calculate_net_position_metrics(positions, prices)

        assert result["net_position_usd"] == pytest.approx(0.0)
        assert result["net_position_vol_usd"] == pytest.approx(0.0)

    def test_no_token_returns_zeros(self, reporter):
        positions = {
            "position_token": None,
            "net_position": 0.0,
            "spot_position": 0.0,
            "perp_position": 0.0,
        }
        result = reporter._calculate_net_position_metrics(positions, None)

        assert result["net_position_usd"] == pytest.approx(0.0)
        assert result["net_position_vol_usd"] == pytest.approx(0.0)
        assert result["position_30d_vol"] == pytest.approx(0.0)

    def test_net_position_usd_correct(self, reporter):
        """net_position_usd = net_position * last_price."""
        positions = {
            "position_token": "ETH",
            "net_position": 2.0,
            "spot_position": 2.0,
            "perp_position": 0.0,
        }
        prices = _make_price_series(60, last_price=3000.0)
        result = reporter._calculate_net_position_metrics(positions, prices)

        assert result["net_position_usd"] == pytest.approx(2.0 * 3000.0, rel=1e-6)

    def test_net_position_usd_short(self, reporter):
        """Negative net position → negative net_position_usd."""
        positions = {
            "position_token": "BTC",
            "net_position": -0.5,
            "spot_position": 0.0,
            "perp_position": -0.5,
        }
        prices = _make_price_series(60, last_price=50000.0)
        result = reporter._calculate_net_position_metrics(positions, prices)

        assert result["net_position_usd"] == pytest.approx(-0.5 * 50000.0, rel=1e-6)

    def test_30d_vol_annualised(self, reporter):
        """30-day log-return vol should be annualised (× sqrt(365)) and > 0."""
        positions = {
            "position_token": "ETH",
            "net_position": 1.0,
            "spot_position": 0.0,
            "perp_position": 1.0,
        }
        prices = _make_price_series(60, last_price=3000.0, seed=7)
        result = reporter._calculate_net_position_metrics(positions, prices)

        assert result["position_30d_vol"] > 0.0
        # Annualised vol of a ~2% daily series is roughly 2% * sqrt(365) ≈ 38%
        assert 0.01 < result["position_30d_vol"] < 5.0  # sanity bounds (as fraction)

    def test_vol_usd_equals_abs_position_times_price_times_vol(self, reporter):
        """net_position_vol_usd = |net_position| * last_price * position_30d_vol."""
        positions = {
            "position_token": "ETH",
            "net_position": -3.0,
            "spot_position": 0.0,
            "perp_position": -3.0,
        }
        prices = _make_price_series(60, last_price=2000.0, seed=99)
        result = reporter._calculate_net_position_metrics(positions, prices)

        expected = abs(-3.0) * 2000.0 * result["position_30d_vol"]
        assert result["net_position_vol_usd"] == pytest.approx(expected, rel=1e-6)

    def test_empty_price_series_returns_zero_vol(self, reporter):
        positions = {
            "position_token": "ETH",
            "net_position": 1.0,
            "spot_position": 0.0,
            "perp_position": 1.0,
        }
        result = reporter._calculate_net_position_metrics(positions, pd.Series([], dtype=float))

        assert result["position_30d_vol"] == pytest.approx(0.0)
        assert result["net_position_vol_usd"] == pytest.approx(0.0)

    def test_insufficient_price_history_returns_zero_vol(self, reporter):
        """With only 1 data point, vol cannot be computed → 0."""
        positions = {
            "position_token": "ETH",
            "net_position": 1.0,
            "spot_position": 0.0,
            "perp_position": 1.0,
        }
        prices = _make_price_series(1, last_price=3000.0)
        result = reporter._calculate_net_position_metrics(positions, prices)

        assert result["position_30d_vol"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests for _calculate_summary_stats – new position fields
# ---------------------------------------------------------------------------

class TestSummaryStatsPositionFields:
    """Ensure _calculate_summary_stats includes position fields from account_summary."""

    @pytest.fixture
    def reporter(self):
        return _make_reporter()

    def _base_account_summary(self, **overrides):
        base = {
            "net_deposits": 10000.0,
            "total_deposits": 10000.0,
            "total_withdrawals": 0.0,
            "spot_value": 6000.0,
            "perp_value": 4000.0,
            "unrealized_pnl": 0.0,
            "current_value": 10000.0,
            "spot_position": 2.0,
            "perp_position": -1.0,
            "net_position": 1.0,
            "position_token": "ETH",
            "last_perp_price": 3000.0,
            "net_position_usd": 3000.0,
            "position_30d_vol": 0.40,
            "net_position_vol_usd": 1200.0,
        }
        base.update(overrides)
        return base

    def test_position_fields_present(self, reporter):
        account_summary = self._base_account_summary()
        stats = reporter._calculate_summary_stats(
            aum_data=pd.DataFrame({"aum_usd": [10000.0]}),
            performance_data=pd.DataFrame({"pnl_usd": [0.0], "pnl_pct": [0.0]}),
            trade_analysis=pd.DataFrame(),
            funding_analysis=pd.DataFrame(),
            account_summary=account_summary,
        )

        assert stats["spot_position"] == pytest.approx(2.0)
        assert stats["perp_position"] == pytest.approx(-1.0)
        assert stats["net_position"] == pytest.approx(1.0)
        assert stats["position_token"] == "ETH"
        assert stats["last_perp_price"] == pytest.approx(3000.0)
        assert stats["net_position_usd"] == pytest.approx(3000.0)
        assert stats["position_30d_vol"] == pytest.approx(0.40)
        assert stats["net_position_vol_usd"] == pytest.approx(1200.0)

    def test_position_fields_default_to_zero_when_absent(self, reporter):
        """If account_summary has no position keys, stats should default to zero/None."""
        account_summary = {
            "net_deposits": 10000.0,
            "total_deposits": 10000.0,
            "total_withdrawals": 0.0,
            "spot_value": 0.0,
            "perp_value": 10000.0,
            "unrealized_pnl": 0.0,
            "current_value": 10000.0,
        }
        stats = reporter._calculate_summary_stats(
            aum_data=pd.DataFrame({"aum_usd": [10000.0]}),
            performance_data=pd.DataFrame({"pnl_usd": [0.0], "pnl_pct": [0.0]}),
            trade_analysis=pd.DataFrame(),
            funding_analysis=pd.DataFrame(),
            account_summary=account_summary,
        )

        assert stats["spot_position"] == pytest.approx(0.0)
        assert stats["perp_position"] == pytest.approx(0.0)
        assert stats["net_position"] == pytest.approx(0.0)
        assert stats["position_token"] is None
        assert stats["last_perp_price"] == pytest.approx(0.0)
        assert stats["net_position_usd"] == pytest.approx(0.0)
        assert stats["position_30d_vol"] == pytest.approx(0.0)
        assert stats["net_position_vol_usd"] == pytest.approx(0.0)

    def test_no_spot_no_perp_all_zero(self, reporter):
        """Account with no open positions → all position fields are zero."""
        account_summary = self._base_account_summary(
            spot_position=0.0,
            perp_position=0.0,
            net_position=0.0,
            position_token=None,
            last_perp_price=0.0,
            net_position_usd=0.0,
            position_30d_vol=0.0,
            net_position_vol_usd=0.0,
        )
        stats = reporter._calculate_summary_stats(
            aum_data=pd.DataFrame({"aum_usd": [10000.0]}),
            performance_data=pd.DataFrame({"pnl_usd": [0.0], "pnl_pct": [0.0]}),
            trade_analysis=pd.DataFrame(),
            funding_analysis=pd.DataFrame(),
            account_summary=account_summary,
        )

        assert stats["net_position_usd"] == pytest.approx(0.0)
        assert stats["net_position_vol_usd"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: get_account_summary enriches with position data
# ---------------------------------------------------------------------------

class TestGetAccountSummaryPositionEnrichment:
    """Verify that get_account_summary returns the new position keys."""

    def _make_monitor_with_positions(self, perp_szi: str, spot_balances: list):
        from hyperliquid_reporter.monitoring import HyperliquidMonitor

        info = MagicMock()
        info.user_state.return_value = {
            "marginSummary": {"accountValue": "10000", "availableBalance": "5000"},
            "assetPositions": [
                {"position": {"coin": "ETH", "szi": perp_szi, "positionValue": "3000"}}
            ] if perp_szi != "0" else [],
        }
        info.spot_user_state.return_value = {"balances": spot_balances}
        info.portfolio.return_value = [
            ("allTime", {"accountValueHistory": [[0, "10000"]], "pnlHistory": [[0, "0"]], "vlm": "0"})
        ]
        info.user_non_funding_ledger_updates.return_value = []

        monitor = HyperliquidMonitor(info=info, address="0xtest")
        return monitor

    def test_account_summary_has_position_keys(self):
        monitor = self._make_monitor_with_positions(
            perp_szi="1.5",
            spot_balances=[
                {"coin": "USDC", "total": "5000", "entryNtl": "0"},
            ],
        )
        summary = monitor.get_account_summary()

        assert "spot_position" in summary
        assert "perp_position" in summary
        assert "net_position" in summary
        assert "position_token" in summary

    def test_account_summary_perp_position_correct(self):
        monitor = self._make_monitor_with_positions(
            perp_szi="2.0",
            spot_balances=[{"coin": "USDC", "total": "5000", "entryNtl": "0"}],
        )
        summary = monitor.get_account_summary()

        assert summary["perp_position"] == pytest.approx(2.0)
        assert summary["spot_position"] == pytest.approx(0.0)
        assert summary["net_position"] == pytest.approx(2.0)
        assert summary["position_token"] == "ETH"

    def test_account_summary_spot_position_correct(self):
        monitor = self._make_monitor_with_positions(
            perp_szi="0",
            spot_balances=[
                {"coin": "USDC", "total": "2000", "entryNtl": "0"},
                {"coin": "ETH", "total": "1.5", "entryNtl": "4500"},
            ],
        )
        summary = monitor.get_account_summary()

        assert summary["spot_position"] == pytest.approx(1.5)
        assert summary["perp_position"] == pytest.approx(0.0)
        assert summary["net_position"] == pytest.approx(1.5)
        assert summary["position_token"] == "ETH"

    def test_account_summary_no_positions_returns_zeros(self):
        monitor = self._make_monitor_with_positions(
            perp_szi="0",
            spot_balances=[{"coin": "USDC", "total": "10000", "entryNtl": "0"}],
        )
        summary = monitor.get_account_summary()

        assert summary["spot_position"] == pytest.approx(0.0)
        assert summary["perp_position"] == pytest.approx(0.0)
        assert summary["net_position"] == pytest.approx(0.0)
        assert summary["position_token"] is None
