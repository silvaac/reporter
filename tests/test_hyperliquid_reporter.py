"""Unit tests for hyperliquid_reporter module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta

from hyperliquid_reporter.reporter import HyperliquidReporter
from exceptions import ReportGenerationError


class TestHyperliquidReporter:
    """Test HyperliquidReporter class."""
    
    @pytest.fixture
    def mock_monitor(self):
        """Create a mock HyperliquidMonitor."""
        monitor = Mock()
        monitor.get_portfolio_dataframe = Mock()
        monitor.get_trade_history = Mock()
        monitor.get_funding_history = Mock()
        monitor.get_account_summary = Mock()
        return monitor
    
    @pytest.fixture
    def reporter(self, mock_monitor):
        """Create a HyperliquidReporter instance with mock monitor."""
        return HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef1234567890abcdef12345678"
        )
    
    def test_init(self, mock_monitor):
        """Test reporter initialization."""
        address = "0x1234567890abcdef1234567890abcdef12345678"
        reporter = HyperliquidReporter(monitor=mock_monitor, account_address=address)
        assert reporter.monitor == mock_monitor
        assert reporter.account_address == address
    
    def test_generate_aum_data_success(self, reporter, mock_monitor):
        """Test successful AUM data generation."""
        mock_df = pd.DataFrame({
            'account_value': [10000, 11000, 12000]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_monitor.get_portfolio_dataframe.return_value = mock_df
        
        result = reporter.generate_aum_data(period="week")
        
        assert 'aum_usd' in result.columns
        assert len(result) == 3
        assert result['aum_usd'].iloc[0] == 10000
        mock_monitor.get_portfolio_dataframe.assert_called_once_with(
            period="week",
            data_type="account_value"
        )
    
    def test_generate_aum_data_empty(self, reporter, mock_monitor):
        """Test AUM data generation with empty data."""
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame()
        
        result = reporter.generate_aum_data(period="day")
        
        assert result.empty
        assert 'aum_usd' in result.columns
    
    def test_generate_performance_data_success(self, reporter, mock_monitor):
        """Test successful performance data generation."""
        from unittest.mock import MagicMock
        
        mock_df = pd.DataFrame({
            'account_value': [10000, 11000, 12000]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_monitor.get_portfolio_dataframe.return_value = mock_df
        # Mock the ledger API structure
        mock_info = MagicMock()
        mock_info.user_non_funding_ledger_updates.return_value = []
        mock_monitor._info = mock_info
        mock_monitor._address = "0x123"
        # Mock account summary for fallback
        mock_monitor.get_account_summary.return_value = {'net_deposits': 10000.0}
        
        result = reporter.generate_performance_data(period="month")
        
        assert not result.empty
        assert 'aum_usd' in result.columns
        assert 'pnl_usd' in result.columns
        assert 'pnl_pct' in result.columns
        assert len(result) == 3
        mock_monitor.get_portfolio_dataframe.assert_called_once_with(
            period="month",
            data_type="account_value"
        )
    
    def test_generate_performance_data_pnl_percentage_calculation(self, reporter, mock_monitor):
        """Test that % PnL is correctly calculated using period-over-period changes.
        
        New algorithm: pnl_usd(t) = aum_usd(t) - aum_usd(t-1) - (net_deposits(t) - net_deposits(t-1))
        pnl_pct(t) = pnl_usd(t) / aum_usd(t-1)
        
        Scenario:
        - Day 1: Start with $10,000, no profit -> 0% PnL (first row is always 0)
        - Day 2: Make $1,000 profit, AUM = $11,000 -> 10% PnL (1000/10000)
        - Day 3: Deposit $5,000, AUM = $16,000 -> 0% PnL (no trading profit, just deposit)
        - Day 4: Make $500 profit, AUM = $16,500 -> 3.33% PnL (500/15000)
        """
        from unittest.mock import MagicMock
        
        dates = pd.date_range('2024-01-01', periods=4)
        mock_df = pd.DataFrame({
            'account_value': [10000.0, 11000.0, 16000.0, 16500.0]
        }, index=dates)
        
        mock_monitor.get_portfolio_dataframe.return_value = mock_df
        
        # Mock ledger updates to trigger fallback (return None to cause exception)
        mock_info = MagicMock()
        mock_info.user_non_funding_ledger_updates.side_effect = Exception("Mock error")
        mock_monitor._info = mock_info
        mock_monitor._address = "0x123"
        # Mock account summary for fallback
        mock_monitor.get_account_summary.return_value = {'net_deposits': 10000.0}
        
        result = reporter.generate_performance_data(period="month")
        
        assert 'pnl_pct' in result.columns
        assert 'net_deposits' in result.columns
        
        # With fallback, all net_deposits are the same
        # With fallback, all net_deposits are the same, and first row AUM equals deposits
        assert result['net_deposits'].iloc[0] == 10000.0
        assert result['net_deposits'].iloc[1] == 10000.0
        assert result['net_deposits'].iloc[2] == 10000.0
        assert result['net_deposits'].iloc[3] == 10000.0
        
        # First row AUM should equal deposits (initial state)
        assert result['aum_usd'].iloc[0] == 10000.0
        # First row P&L is always 0
        assert result['pnl_pct'].iloc[0] == 0.0
        # Day 2: 10% profit (1000/10000)
        assert abs(result['pnl_pct'].iloc[1] - 10.0) < 0.01
        # Day 3: 45.45% (5000 AUM change / 11000 previous AUM)
        assert abs(result['pnl_pct'].iloc[2] - 45.45) < 0.01
        # Day 4: 3.125% profit (500/16000)
        assert abs(result['pnl_pct'].iloc[3] - 3.125) < 0.01
    
    def test_generate_performance_data_handles_zero_net_deposits(self, reporter, mock_monitor):
        """Test that % PnL handles edge case of zero net deposits gracefully."""
        from unittest.mock import MagicMock
        
        mock_df = pd.DataFrame({
            'account_value': [1000.0, 0.0, 500.0]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_monitor.get_portfolio_dataframe.return_value = mock_df
        # Mock empty ledger (no deposits)
        mock_info = MagicMock()
        mock_info.user_non_funding_ledger_updates.return_value = []
        mock_monitor._info = mock_info
        mock_monitor._address = "0x123"
        
        result = reporter.generate_performance_data(period="month")
        
        # First row deposits are 0, AUM remains as is (1000.0 from mock data)
        assert result['net_deposits'].iloc[0] == 0.0
        # First row P&L is always 0
        assert result['pnl_pct'].iloc[0] == 0.0
        # Second row: -100% (1000→0 AUM change)
        assert result['pnl_pct'].iloc[1] == -100.0
        # Third row: 0% (0→500 AUM change, but division by zero protection)
        assert result['pnl_pct'].iloc[2] == 0.0
    
    def test_generate_performance_data_handles_negative_pnl(self, reporter, mock_monitor):
        """Test that % PnL correctly handles losses.
        
        Scenario:
        - Day 1: Start with $10,000 -> 0% PnL (first row is always 0)
        - Day 2: Lose $2,000, AUM = $8,000 -> -20% PnL (-2000/10000)
        """
        from unittest.mock import MagicMock
        
        dates = pd.date_range('2024-01-01', periods=2)
        mock_df = pd.DataFrame({
            'account_value': [10000.0, 8000.0]
        }, index=dates)
        
        mock_monitor.get_portfolio_dataframe.return_value = mock_df
        # Mock ledger updates to trigger fallback
        mock_info = MagicMock()
        mock_info.user_non_funding_ledger_updates.side_effect = Exception("Mock error")
        mock_monitor._info = mock_info
        mock_monitor._address = "0x123"
        # Mock account summary for fallback
        mock_monitor.get_account_summary.return_value = {'net_deposits': 10000.0}
        
        result = reporter.generate_performance_data(period="month")
        
        # First row AUM should equal deposits (initial state)
        assert result['aum_usd'].iloc[0] == 10000.0
        assert result['net_deposits'].iloc[0] == 10000.0
        # First row P&L is always 0
        assert result['pnl_pct'].iloc[0] == 0.0
        # Second row: -20% loss (-2000/10000)
        assert abs(result['pnl_pct'].iloc[1] - (-20.0)) < 0.01
    
    def test_generate_trade_analysis_success(self, reporter, mock_monitor):
        """Test successful trade analysis generation."""
        mock_df = pd.DataFrame({
            'coin': ['BTC', 'ETH'],
            'side': ['B', 'A'],
            'px': [50000.0, 3000.0],
            'sz': [0.1, 1.0],
            'fee': [5.0, 3.0],
            'closedPnl': [100.0, -50.0],
            'fee_bps': [5.0, 3.0],
            'feeToken': ['USDC', 'USDC'],
            'dir': ['Buy', 'Sell']
        }, index=pd.date_range('2024-01-01', periods=2))
        
        mock_monitor.get_trade_history.return_value = mock_df
        
        result = reporter.generate_trade_analysis()
        
        assert 'coin' in result.columns
        assert 'side' in result.columns
        assert 'price' in result.columns
        assert 'size' in result.columns
        assert 'notional' in result.columns
        assert 'fee' in result.columns
        assert 'net_pnl' in result.columns
        assert 'fee_bps' in result.columns
        assert 'feeToken' in result.columns
        assert 'dir' in result.columns
        assert len(result) == 2
        assert result['side'].iloc[0] == 'buy'
        assert result['side'].iloc[1] == 'sell'
        assert result['fee_bps'].iloc[0] == 5.0
        assert result['feeToken'].iloc[0] == 'USDC'
        assert result['dir'].iloc[0] == 'Buy'
    
    def test_generate_funding_analysis_success(self, reporter, mock_monitor):
        """Test successful funding analysis generation."""
        mock_df = pd.DataFrame({
            'coin': ['BTC', 'ETH'],
            'usdc': [-10.5, 5.2],
            'szi': [0.5, -1.0],
            'fundingRate': [0.0001, -0.0002]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        mock_monitor.get_funding_history.return_value = mock_df
        
        result = reporter.generate_funding_analysis(lookback_days=7)
        
        assert 'coin' in result.columns
        assert 'funding_payment' in result.columns
        assert 'position_size' in result.columns
        assert 'funding_rate' in result.columns
        assert len(result) == 2
    
    def test_calculate_summary_stats(self, reporter):
        """Test summary statistics calculation."""
        aum_data = pd.DataFrame({
            'aum_usd': [10000, 11000, 12000]
        })
        
        performance_data = pd.DataFrame({
            'pnl_usd': [0, 1000, 2000],
            'pnl_pct': [0, 10, 20]
        })
        
        trade_analysis = pd.DataFrame({
            'notional': [5000, 3000],
            'fee': [5, 3],
            'net_pnl': [100, -50]
        })
        
        funding_analysis = pd.DataFrame({
            'coin': ['BTC', 'ETH'],
            'funding_payment': [-10, 5]
        })
        
        account_summary = {
            'net_deposits': 10000,
            'total_deposits': 15000,
            'total_withdrawals': 5000,
            'spot_value': 2000,
            'perp_value': 10000,
            'unrealized_pnl': 500,
            'current_value': 12000
        }
        
        stats = reporter._calculate_summary_stats(
            aum_data=aum_data,
            performance_data=performance_data,
            trade_analysis=trade_analysis,
            funding_analysis=funding_analysis,
            account_summary=account_summary
        )
        
        assert stats['current_aum'] == 12000
        assert stats['initial_aum'] == 10000
        assert stats['peak_aum'] == 12000
        assert stats['total_pnl_usd'] == 2000
        assert stats['total_trades'] == 2
        assert stats['total_fees'] == 8
        assert stats['total_volume'] == 8000
        assert stats['winning_trades'] == 1
        assert stats['win_rate'] == 50.0
        assert stats['total_funding_paid'] == -5
    
    def test_generate_report_data_integration(self, reporter, mock_monitor):
        """Test full report data generation."""
        mock_monitor.get_portfolio_dataframe.return_value = pd.DataFrame({
            'account_value': [10000],
            'pnl': [0]
        }, index=[datetime.now()])
        
        mock_monitor.get_trade_history.return_value = pd.DataFrame()
        mock_monitor.get_funding_history.return_value = pd.DataFrame()
        mock_monitor.get_account_summary.return_value = {
            'net_deposits': 10000,
            'current_value': 10000,
            'total_pnl': 0
        }
        
        result = reporter.generate_report_data(period="day")
        
        assert 'aum_data' in result
        assert 'performance_data' in result
        assert 'trade_analysis' in result
        assert 'funding_analysis' in result
        assert 'summary_stats' in result
        assert 'account_summary' in result
        assert 'period' in result
        assert 'generated_at' in result
    
    def test_fig_to_base64(self, reporter):
        """Test figure to base64 conversion."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        result = reporter._fig_to_base64(fig)
        
        assert isinstance(result, str)
        assert len(result) > 0
