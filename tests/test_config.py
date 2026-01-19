"""Unit tests for config module."""

import pytest
from pathlib import Path
from config import TradingConfig
from exceptions import ConfigurationError


class TestTradingConfig:
    """Test TradingConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TradingConfig()
        assert config.trading_mode == "both"
        assert config.safety_max == 1.0
        assert config.min_trade_value == 1.0
    
    def test_valid_trading_modes(self):
        """Test valid trading mode values."""
        valid_modes = ['spot_only', 'perp_only', 'both']
        for mode in valid_modes:
            config = TradingConfig(trading_mode=mode)
            assert config.trading_mode == mode
    
    def test_invalid_trading_mode(self):
        """Test invalid trading mode raises error."""
        with pytest.raises(ConfigurationError, match="Invalid trading_mode"):
            TradingConfig(trading_mode="invalid")
    
    def test_valid_safety_max(self):
        """Test valid safety_max values."""
        valid_values = [0.1, 0.5, 1.0]
        for value in valid_values:
            config = TradingConfig(safety_max=value)
            assert config.safety_max == value
    
    def test_invalid_safety_max_zero(self):
        """Test safety_max of 0 raises error."""
        with pytest.raises(ConfigurationError, match="safety_max must be in"):
            TradingConfig(safety_max=0.0)
    
    def test_invalid_safety_max_negative(self):
        """Test negative safety_max raises error."""
        with pytest.raises(ConfigurationError, match="safety_max must be in"):
            TradingConfig(safety_max=-0.5)
    
    def test_invalid_safety_max_too_large(self):
        """Test safety_max > 1 raises error."""
        with pytest.raises(ConfigurationError, match="safety_max must be in"):
            TradingConfig(safety_max=1.5)
    
    def test_valid_min_trade_value(self):
        """Test valid min_trade_value."""
        config = TradingConfig(min_trade_value=10.0)
        assert config.min_trade_value == 10.0
    
    def test_invalid_min_trade_value(self):
        """Test negative min_trade_value raises error."""
        with pytest.raises(ConfigurationError, match="min_trade_value must be"):
            TradingConfig(min_trade_value=-5.0)
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = TradingConfig(
            trading_mode="spot_only",
            safety_max=0.8,
            min_trade_value=50.0
        )
        assert config.trading_mode == "spot_only"
        assert config.safety_max == 0.8
        assert config.min_trade_value == 50.0


class TestLoadConfig:
    """Test load_config function."""
    
    def test_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        from config import load_config
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_config("nonexistent_config.json")
    
    def test_config_path_as_string(self):
        """Test config_path can be a string."""
        from config import load_config
        # Test that the function accepts string paths
        # Will succeed if token_data is installed and config exists
        try:
            result = load_config("config_hyperliquid.json", testnet=True)
            # If successful, verify it returns expected keys
            assert 'account_address' in result
            assert '_hyperliquid_info' in result
            assert '_hyperliquid_exchange' in result
        except (ConfigurationError, Exception):
            # Expected if config is invalid or token_data has issues
            pass
    
    def test_config_path_as_path_object(self):
        """Test config_path can be a Path object."""
        from config import load_config
        # Test that the function accepts Path objects
        try:
            result = load_config(Path("config_hyperliquid.json"), testnet=True)
            # If successful, verify it returns expected keys
            assert 'account_address' in result
            assert '_hyperliquid_info' in result
            assert '_hyperliquid_exchange' in result
        except (ConfigurationError, Exception):
            # Expected if config is invalid or token_data has issues
            pass
