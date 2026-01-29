"""Test script to verify funding analysis enhancements."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from hyperliquid_reporter.reporter import HyperliquidReporter
from hyperliquid_reporter.monitoring import HyperliquidMonitor


def test_funding_analysis_with_prices():
    """Test that funding analysis includes price and calculated funding columns."""
    
    # Create mock monitor
    mock_monitor = Mock(spec=HyperliquidMonitor)
    mock_monitor._info = Mock()
    mock_monitor._address = "0x1234567890abcdef"
    
    # Create sample funding data
    now = datetime.now()
    funding_data = pd.DataFrame({
        'coin': ['ETH', 'BTC', 'SOL', 'ETH', 'BTC'],
        'usdc': [-0.5, -1.2, 0.3, -0.6, -1.1],
        'szi': [10.0, 0.5, 100.0, 12.0, 0.48],
        'fundingRate': [0.0001, 0.0002, -0.00015, 0.00012, 0.00018]
    }, index=[
        now - timedelta(hours=8),
        now - timedelta(hours=8),
        now - timedelta(hours=8),
        now - timedelta(hours=4),
        now - timedelta(hours=4)
    ])
    funding_data.index.name = 'datetime'
    
    mock_monitor.get_funding_history.return_value = funding_data
    
    # Create reporter
    reporter = HyperliquidReporter(
        monitor=mock_monitor,
        account_address="0x1234567890abcdef"
    )
    
    # Mock the _add_token_prices_to_funding method to simulate price data
    def mock_add_prices(funding_df):
        """Mock function to add prices without actual API calls."""
        funding_df['token_price'] = 0.0
        
        # Add sample prices for each coin
        for idx, row in funding_df.iterrows():
            if row['coin'] == 'ETH':
                funding_df.loc[idx, 'token_price'] = 3000.0
            elif row['coin'] == 'BTC':
                funding_df.loc[idx, 'token_price'] = 60000.0
            elif row['coin'] == 'SOL':
                funding_df.loc[idx, 'token_price'] = 150.0
        
        return funding_df
    
    # Patch the method
    with patch.object(reporter, '_add_token_prices_to_funding', side_effect=mock_add_prices):
        # Generate funding analysis
        result = reporter.generate_funding_analysis()
    
    # Verify the result has the expected columns
    expected_columns = [
        'coin', 'funding_payment', 'position_size', 'funding_rate',
        'token_price', 'calculated_funding'
    ]
    
    print("Testing funding analysis enhancements...")
    print(f"\nExpected columns: {expected_columns}")
    print(f"Actual columns: {result.columns.tolist()}")
    
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"
    
    print("✓ All expected columns present")
    
    # Verify calculated_funding is computed correctly
    print("\nVerifying calculated funding formula (-1 * size * price * rate):")
    for idx, row in result.iterrows():
        expected_calc = -1 * row['position_size'] * row['token_price'] * row['funding_rate']
        actual_calc = row['calculated_funding']
        
        print(f"  {row['coin']}: price={row['token_price']:.2f}, "
              f"size={row['position_size']:.4f}, rate={row['funding_rate']:.6f}")
        print(f"    Expected: {expected_calc:.6f}, Actual: {actual_calc:.6f}")
        
        # Allow small floating point differences
        assert abs(expected_calc - actual_calc) < 0.0001, \
            f"Calculated funding mismatch for {row['coin']}"
    
    print("✓ Calculated funding values are correct")
    
    # Verify token_price column has values
    assert (result['token_price'] > 0).all(), "Some token prices are zero or negative"
    print("✓ All token prices are positive")
    
    print("\n" + "="*60)
    print("Sample output:")
    print("="*60)
    print(result.to_string())
    print("\n✅ All tests passed! Funding analysis enhancements working correctly.")


if __name__ == "__main__":
    test_funding_analysis_with_prices()
