"""Test script to verify funding data timezone."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from config import load_config
from hyperliquid_reporter.monitoring import HyperliquidMonitor

# Load configuration
config = load_config("config_hyperliquid.json", testnet=False)

# Initialize monitor
monitor = HyperliquidMonitor(
    info=config['_hyperliquid_info'],
    address=config['account_address']
)

# Get recent funding data
funding_df = monitor.get_funding_history(
    lookback=1,  # Last 1 day
    as_dataframe=True
)

print("="*80)
print("FUNDING DATA TIMEZONE INVESTIGATION")
print("="*80)
print(f"\nCurrent time (local): {datetime.now()}")
print(f"Current time (UTC): {datetime.utcnow()}")
print(f"\nFunding DataFrame shape: {funding_df.shape}")
print(f"Index name: {funding_df.index.name}")
print(f"Index dtype: {funding_df.index.dtype}")
print(f"Index timezone: {funding_df.index.tz}")

if not funding_df.empty:
    print(f"\nFirst entry timestamp: {funding_df.index[0]}")
    print(f"Last entry timestamp: {funding_df.index[-1]}")
    print(f"\nLast 5 entries:")
    print(funding_df.tail(5)[['coin', 'usdc', 'szi', 'fundingRate']])
    
    # Check time difference from now
    last_time = funding_df.index[-1]
    now_utc = datetime.utcnow()
    
    # If index is timezone-aware, convert now to aware
    if last_time.tzinfo is not None:
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
    
    time_diff = now_utc - last_time
    print(f"\nTime difference from last entry to now (UTC): {time_diff}")
    print(f"Hours difference: {time_diff.total_seconds() / 3600:.2f}")
    
    # Check if timestamps look like UTC or local
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    hours_diff = time_diff.total_seconds() / 3600
    if abs(hours_diff) < 2:
        print("✓ Timestamps appear to be in UTC (very recent)")
    elif 4 < hours_diff < 6:
        print("⚠ Timestamps might be in EST/EDT (5 hours behind UTC)")
    else:
        print(f"? Timestamps have unusual offset: {hours_diff:.2f} hours")
