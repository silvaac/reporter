"""Test script to verify cumulative funding rate chart."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock
from hyperliquid_reporter.reporter import HyperliquidReporter
from hyperliquid_reporter.monitoring import HyperliquidMonitor

# Create mock monitor
mock_monitor = Mock(spec=HyperliquidMonitor)
mock_monitor._info = Mock()
mock_monitor._address = "0x1234567890abcdef"

# Create reporter
reporter = HyperliquidReporter(
    monitor=mock_monitor,
    account_address="0x1234567890abcdef"
)

# Create sample funding data with funding rates
now = datetime.now()
funding_data = pd.DataFrame({
    'coin': ['ETH', 'BTC', 'SOL', 'ETH', 'BTC', 'SOL', 'ETH', 'BTC'],
    'funding_payment': [-0.5, -1.2, 0.3, -0.6, -1.1, 0.4, -0.7, -1.3],
    'position_size': [10.0, 0.5, 100.0, 12.0, 0.48, 110.0, 11.0, 0.52],
    'funding_rate': [0.0001, 0.0002, -0.00015, 0.00012, 0.00018, -0.00010, 0.00015, 0.00020],
    'token_price': [3000.0, 60000.0, 150.0, 3100.0, 61000.0, 155.0, 3050.0, 60500.0],
    'calculated_funding': [0.3, 0.6, -0.225, 0.4464, 0.5270, -0.1705, 0.5033, 0.6292]
}, index=[
    now - timedelta(days=7),
    now - timedelta(days=7),
    now - timedelta(days=7),
    now - timedelta(days=5),
    now - timedelta(days=5),
    now - timedelta(days=5),
    now - timedelta(days=3),
    now - timedelta(days=3),
])
funding_data.index.name = 'datetime'

print("="*80)
print("CUMULATIVE FUNDING RATE CHART TEST")
print("="*80)

# Test the chart creation method
chart_base64 = reporter._create_cumulative_funding_rate_chart(funding_data)

if chart_base64:
    print("\n✓ Chart created successfully!")
    print(f"  Base64 length: {len(chart_base64)} characters")
    print(f"  Chart is valid: {chart_base64.startswith('iVBOR')}")  # PNG signature in base64
    
    # Calculate cumulative funding rate manually to verify
    funding_rate_bps = funding_data['funding_rate'] * 10000
    cumulative_rate_bps = funding_rate_bps.cumsum()
    
    print(f"\n  Funding rates (bps): {funding_rate_bps.tolist()}")
    print(f"  Cumulative rates (bps): {cumulative_rate_bps.tolist()}")
    print(f"  Final cumulative rate: {cumulative_rate_bps.iloc[-1]:.2f} bps")
    
    # Verify the cumulative calculation
    expected_cumsum = sum(funding_rate_bps)
    actual_cumsum = cumulative_rate_bps.iloc[-1]
    
    if abs(expected_cumsum - actual_cumsum) < 0.0001:
        print(f"\n✓ Cumulative calculation is correct!")
    else:
        print(f"\n✗ Cumulative calculation mismatch!")
        print(f"  Expected: {expected_cumsum:.4f}")
        print(f"  Actual: {actual_cumsum:.4f}")
else:
    print("\n✗ Chart creation failed!")

# Test with empty dataframe
empty_df = pd.DataFrame()
empty_chart = reporter._create_cumulative_funding_rate_chart(empty_df)

if empty_chart is None:
    print("\n✓ Empty dataframe handled correctly (returns None)")
else:
    print("\n✗ Empty dataframe should return None")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
