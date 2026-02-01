"""Tests for configurable file paths functionality."""

import sys
from pathlib import Path
import tempfile
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from hyperliquid_reporter.monitoring import HyperliquidMonitor
from hyperliquid_reporter.reporter import HyperliquidReporter


class TestConfigurablePaths:
    """Test that file paths are configurable and work correctly."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_monitor(self):
        """Create a mock HyperliquidMonitor."""
        monitor = Mock(spec=HyperliquidMonitor)
        monitor._info = Mock()
        monitor._address = "0x1234567890abcdef"
        return monitor
    
    def test_reporter_accepts_custom_paths(self, mock_monitor, temp_dir):
        """Test that reporter accepts and stores custom file paths."""
        custom_pnl_file = str(Path(temp_dir) / "custom_pnl.csv")
        custom_cache_dir = str(Path(temp_dir) / "custom_cache")
        
        reporter = HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef",
            pnl_history_file=custom_pnl_file,
            price_cache_dir=custom_cache_dir
        )
        
        assert reporter.pnl_history_file == custom_pnl_file
        assert reporter.price_cache_dir == custom_cache_dir
    
    def test_reporter_uses_default_paths(self, mock_monitor):
        """Test that reporter uses default paths when not specified."""
        reporter = HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef"
        )
        
        assert reporter.pnl_history_file == "pnl_history.csv"
        assert reporter.price_cache_dir == "./data/hyperliquid"
    
    def test_save_pnl_history_to_custom_path(self, mock_monitor, temp_dir):
        """Test that P&L history is saved to custom path."""
        custom_pnl_file = str(Path(temp_dir) / "test_pnl.csv")
        
        reporter = HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef",
            pnl_history_file=custom_pnl_file
        )
        
        # Save P&L history
        reporter._save_pnl_history(aum_usd=10000.0, net_deposits=8000.0)
        
        # Verify file was created at custom path
        assert Path(custom_pnl_file).exists()
        
        # Verify content
        df = pd.read_csv(custom_pnl_file)
        assert len(df) == 1
        assert df['aum_usd'].iloc[0] == 10000.0
        assert df['net_deposits'].iloc[0] == 8000.0
    
    def test_save_pnl_history_creates_parent_dirs(self, mock_monitor, temp_dir):
        """Test that saving P&L history creates parent directories."""
        custom_pnl_file = str(Path(temp_dir) / "subdir" / "nested" / "pnl.csv")
        
        reporter = HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef",
            pnl_history_file=custom_pnl_file
        )
        
        # Save P&L history
        reporter._save_pnl_history(aum_usd=10000.0, net_deposits=8000.0)
        
        # Verify file and parent directories were created
        assert Path(custom_pnl_file).exists()
        assert Path(custom_pnl_file).parent.exists()
    
    def test_load_pnl_history_from_custom_path(self, mock_monitor, temp_dir):
        """Test that P&L history is loaded from custom path."""
        custom_pnl_file = str(Path(temp_dir) / "test_pnl.csv")
        
        reporter = HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef",
            pnl_history_file=custom_pnl_file
        )
        
        # Save some data first
        reporter._save_pnl_history(aum_usd=10000.0, net_deposits=8000.0)
        reporter._save_pnl_history(aum_usd=11000.0, net_deposits=8000.0)
        
        # Load P&L history
        df = reporter._load_pnl_history()
        
        # Verify data was loaded correctly
        assert len(df) == 2
        assert 'aum_usd' in df.columns
        assert 'net_deposits' in df.columns
        assert 'pnl_usd' in df.columns
        assert 'pnl_pct' in df.columns
    
    def test_load_pnl_history_nonexistent_file(self, mock_monitor, temp_dir):
        """Test that loading from nonexistent file returns empty DataFrame."""
        custom_pnl_file = str(Path(temp_dir) / "nonexistent.csv")
        
        reporter = HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef",
            pnl_history_file=custom_pnl_file
        )
        
        # Load P&L history from nonexistent file
        df = reporter._load_pnl_history()
        
        # Verify empty DataFrame is returned
        assert df.empty
    
    @pytest.mark.skipif(
        not hasattr(sys.modules.get('token_data', None), 'hyperliquid'),
        reason="token_data.hyperliquid not available"
    )
    def test_price_cache_dir_passed_to_price_manager(self, mock_monitor, temp_dir):
        """Test that custom price cache directory is passed to HyperliquidPerpManager."""
        custom_cache_dir = str(Path(temp_dir) / "price_cache")
        
        reporter = HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef",
            price_cache_dir=custom_cache_dir
        )
        
        # Create a mock funding dataframe
        funding_df = pd.DataFrame({
            'coin': ['ETH'],
            'usdc': [10.0],
            'szi': [1.0],
            'fundingRate': [0.0001]
        })
        funding_df.index = pd.DatetimeIndex([pd.Timestamp.now()])
        
        # Mock HyperliquidPerpManager - patch where it's imported
        with patch('token_data.hyperliquid.HyperliquidPerpManager') as mock_manager:
            mock_instance = MagicMock()
            mock_instance.get_data.return_value = pd.DataFrame()
            mock_manager.return_value = mock_instance
            
            # Call the method
            result = reporter._add_token_prices_to_funding(funding_df)
            
            # Verify HyperliquidPerpManager was called with custom cache directory
            mock_manager.assert_called_once()
            call_kwargs = mock_manager.call_args[1]
            assert call_kwargs['data_dir'] == custom_cache_dir
    
    def test_multiple_pnl_saves_append_correctly(self, mock_monitor, temp_dir):
        """Test that multiple P&L saves append to the same file."""
        custom_pnl_file = str(Path(temp_dir) / "test_pnl.csv")
        
        reporter = HyperliquidReporter(
            monitor=mock_monitor,
            account_address="0x1234567890abcdef",
            pnl_history_file=custom_pnl_file
        )
        
        # Save multiple entries
        reporter._save_pnl_history(aum_usd=10000.0, net_deposits=8000.0)
        reporter._save_pnl_history(aum_usd=11000.0, net_deposits=8000.0)
        reporter._save_pnl_history(aum_usd=12000.0, net_deposits=9000.0)
        
        # Load and verify
        df = pd.read_csv(custom_pnl_file)
        assert len(df) == 3
        assert df['aum_usd'].tolist() == [10000.0, 11000.0, 12000.0]
        assert df['net_deposits'].tolist() == [8000.0, 8000.0, 9000.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
