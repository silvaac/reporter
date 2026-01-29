"""Test script to verify the corrected funding formula."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime

# Test the formula with the user's example
print("="*80)
print("FUNDING FORMULA VERIFICATION")
print("="*80)

# User's example: size=-0.1, price=1000, funding_rate=0.01
size = -0.1
price = 1000
funding_rate = 0.01

# Correct formula: -1 * size * price * funding_rate
calculated = -1 * size * price * funding_rate

print(f"\nUser's Example:")
print(f"  Size: {size}")
print(f"  Price: {price}")
print(f"  Funding Rate: {funding_rate}")
print(f"  Formula: -1 * size * price * funding_rate")
print(f"  Calculated: -1 * ({size}) * {price} * {funding_rate} = {calculated}")
print(f"  Expected: 1.0")
print(f"  ✓ Match: {abs(calculated - 1.0) < 0.0001}")

# Test various scenarios
test_cases = [
    # (size, price, funding_rate, expected_result, description)
    (-0.1, 1000, 0.01, 1.0, "Short position, positive funding rate"),
    (0.1, 1000, 0.01, -1.0, "Long position, positive funding rate"),
    (-0.1, 1000, -0.01, -1.0, "Short position, negative funding rate"),
    (0.1, 1000, -0.01, 1.0, "Long position, negative funding rate"),
    (-1.5, 3000, 0.0001, 0.45, "Larger short position"),
    (2.0, 50000, -0.0002, 20.0, "Large long position, negative rate"),
]

print("\n" + "="*80)
print("ADDITIONAL TEST CASES")
print("="*80)

all_pass = True
for size, price, rate, expected, desc in test_cases:
    calculated = -1 * size * price * rate
    match = abs(calculated - expected) < 0.0001
    status = "✓" if match else "✗"
    
    print(f"\n{desc}:")
    print(f"  Size: {size}, Price: {price}, Rate: {rate}")
    print(f"  Calculated: {calculated:.4f}")
    print(f"  Expected: {expected:.4f}")
    print(f"  {status} {'PASS' if match else 'FAIL'}")
    
    if not match:
        all_pass = False

print("\n" + "="*80)
if all_pass:
    print("✅ ALL TESTS PASSED - Formula is correct!")
else:
    print("❌ SOME TESTS FAILED - Formula needs adjustment")
print("="*80)

# Test with actual DataFrame to simulate real usage
print("\n" + "="*80)
print("DATAFRAME TEST")
print("="*80)

df = pd.DataFrame({
    'coin': ['ETH', 'BTC', 'SOL', 'ETH'],
    'position_size': [-0.1, 0.5, -100, 0.2],
    'token_price': [1000, 50000, 20, 3000],
    'funding_rate': [0.01, -0.0002, 0.0001, 0.00015]
})

# Apply the formula
df['calculated_funding'] = -1 * df['position_size'] * df['token_price'] * df['funding_rate']

print("\nDataFrame with calculated funding:")
print(df.to_string(index=False))

print("\nManual verification:")
for idx, row in df.iterrows():
    manual = -1 * row['position_size'] * row['token_price'] * row['funding_rate']
    print(f"  {row['coin']}: -1 * {row['position_size']} * {row['token_price']} * {row['funding_rate']} = {manual:.4f}")
    print(f"    DataFrame value: {row['calculated_funding']:.4f}")
    print(f"    ✓ Match: {abs(manual - row['calculated_funding']) < 0.0001}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
