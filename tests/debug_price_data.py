"""Debug script to check price data structure."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from token_data.hyperliquid import HyperliquidPerpManager
from config import load_config
import pandas as pd

# Load configuration
config = load_config("config_hyperliquid.json", testnet=False)

# Initialize price manager
price_manager = HyperliquidPerpManager(
    ticker=['ETH'],
    data_dir="./data/hyperliquid",
    interval="1h",
    file_type="parquet",
    update=True,
    save=True,
    refresh_hours=48,
    info=config['_hyperliquid_info'],
    verbose=True
)

# Get price data
price_data = price_manager.get_data('ETH')

print("="*80)
print("PRICE DATA STRUCTURE")
print("="*80)
print(f"\nType: {type(price_data)}")
print(f"Shape: {price_data.shape}")
print(f"Columns: {price_data.columns.tolist()}")
print(f"Index type: {type(price_data.index)}")
print(f"Index name: {price_data.index.name}")
print(f"Index dtype: {price_data.index.dtype}")

if 'datetime' in price_data.columns:
    print("\n'datetime' is a COLUMN")
    print(f"First datetime value: {price_data['datetime'].iloc[0]}")
    print(f"Last datetime value: {price_data['datetime'].iloc[-1]}")
else:
    print("\n'datetime' is NOT a column")

if isinstance(price_data.index, pd.DatetimeIndex):
    print("\nIndex IS a DatetimeIndex")
    print(f"First index value: {price_data.index[0]}")
    print(f"Last index value: {price_data.index[-1]}")
else:
    print(f"\nIndex is NOT DatetimeIndex, it's: {type(price_data.index)}")
    print(f"Index values: {price_data.index[:5]}")

print("\nFirst 3 rows:")
print(price_data.head(3))

print("\nLast 3 rows:")
print(price_data.tail(3))
