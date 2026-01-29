"""Test script to verify price matching with UTC timestamps."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from config import load_config
from hyperliquid_reporter.monitoring import HyperliquidMonitor
from hyperliquid_reporter.reporter import HyperliquidReporter

# Load configuration
config = load_config("config_hyperliquid.json", testnet=False)

# Initialize monitor and reporter
monitor = HyperliquidMonitor(
    info=config['_hyperliquid_info'],
    address=config['account_address']
)

reporter = HyperliquidReporter(
    monitor=monitor,
    account_address=config['account_address']
)

print("="*80)
print("PRICE MATCHING TEST")
print("="*80)
print(f"\nCurrent time (local): {datetime.now()}")
print(f"Current time (UTC): {datetime.now().astimezone()}")

# Generate funding analysis with prices
print("\nGenerating funding analysis with price matching...")
funding_analysis = reporter.generate_funding_analysis(lookback_days=1)

print(f"\nFunding analysis shape: {funding_analysis.shape}")
print(f"Columns: {funding_analysis.columns.tolist()}")

if not funding_analysis.empty:
    print(f"\nLast 5 entries:")
    print(funding_analysis.tail(5))
    
    # Check price matching success rate
    total_entries = len(funding_analysis)
    matched_entries = (funding_analysis['token_price'] > 0).sum()
    match_rate = (matched_entries / total_entries * 100) if total_entries > 0 else 0
    
    print(f"\n" + "="*80)
    print("PRICE MATCHING RESULTS:")
    print("="*80)
    print(f"Total entries: {total_entries}")
    print(f"Matched entries: {matched_entries}")
    print(f"Match rate: {match_rate:.1f}%")
    
    if matched_entries > 0:
        print("\n✓ Price matching is working!")
        print("\nSample matched entry:")
        matched_sample = funding_analysis[funding_analysis['token_price'] > 0].iloc[-1]
        print(f"  Timestamp: {matched_sample.name}")
        print(f"  Coin: {matched_sample['coin']}")
        print(f"  Token Price: ${matched_sample['token_price']:,.2f}")
        print(f"  Position Size: {matched_sample['position_size']:,.4f}")
        print(f"  Funding Rate: {matched_sample['funding_rate']:.6f}")
        print(f"  Calculated Funding: ${matched_sample['calculated_funding']:,.4f}")
        print(f"  Reported Funding: ${matched_sample['funding_payment']:,.4f}")
        
        # Check if calculated matches reported (within tolerance)
        diff = abs(matched_sample['calculated_funding'] - matched_sample['funding_payment'])
        if diff < 0.01:
            print(f"\n✓ Calculated funding matches reported (diff: ${diff:.6f})")
        else:
            print(f"\n⚠ Calculated funding differs from reported (diff: ${diff:.6f})")
            print("  This is expected if exchange uses different price source (mark price vs last)")
    else:
        print("\n✗ No prices were matched!")
        print("  Check if price data is available in cache or if API is working")
else:
    print("\n⚠ No funding data available")
