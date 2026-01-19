"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        'account_address': '0x1234567890abcdef1234567890abcdef12345678',
        'base_url': 'https://api.hyperliquid.xyz',
        'testnet': False
    }


@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing."""
    return [
        {
            'coin': 'BTC',
            'side': 'B',
            'px': 50000.0,
            'sz': 0.1,
            'fee': 5.0,
            'closedPnl': 100.0,
            'time': 1704067200000
        },
        {
            'coin': 'ETH',
            'side': 'A',
            'px': 3000.0,
            'sz': 1.0,
            'fee': 3.0,
            'closedPnl': -50.0,
            'time': 1704153600000
        }
    ]


@pytest.fixture
def sample_funding_data():
    """Sample funding data for testing."""
    return [
        {
            'coin': 'BTC',
            'usdc': -10.5,
            'szi': 0.5,
            'fundingRate': 0.0001,
            'time': 1704067200000
        },
        {
            'coin': 'ETH',
            'usdc': 5.2,
            'szi': -1.0,
            'fundingRate': -0.0002,
            'time': 1704153600000
        }
    ]


@pytest.fixture
def sample_account_summary():
    """Sample account summary for testing."""
    return {
        'total_deposits': 15000.0,
        'total_withdrawals': 5000.0,
        'net_deposits': 10000.0,
        'current_value': 12000.0,
        'spot_value': 2000.0,
        'perp_value': 10000.0,
        'perp_position_value': 5000.0,
        'total_pnl': 2000.0,
        'pnl_percentage': 20.0,
        'unrealized_pnl': 500.0,
        'cash_in_perp': 5000.0,
        'when': '2024-01-01 12:00:00'
    }
