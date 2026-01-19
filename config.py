"""Configuration management for the trader package.

This module provides a simplified configuration system that uses token_data.hyperliquid.setup()
for all authentication and connection management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from exceptions import ConfigurationError

logger = logging.getLogger(__name__)




@dataclass
class TradingConfig:
    """Trading configuration parameters.
    
    Attributes:
        trading_mode: 'spot_only', 'perp_only', or 'both'.
        safety_max: Scaling factor for order sizes (0.0-1.0).
        min_trade_value: Minimum USD value for a trade.
    """
    trading_mode: str = "both"
    safety_max: float = 1.0
    min_trade_value: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate trading configuration."""
        valid_modes = {'spot_only', 'perp_only', 'both'}
        if self.trading_mode not in valid_modes:
            raise ConfigurationError(
                f"Invalid trading_mode: {self.trading_mode}. "
                f"Must be one of {valid_modes}"
            )
        if not 0.0 < self.safety_max <= 1.0:
            raise ConfigurationError(
                f"safety_max must be in (0, 1], got {self.safety_max}"
            )
        if self.min_trade_value < 0:
            raise ConfigurationError(
                f"min_trade_value must be >= 0, got {self.min_trade_value}"
            )




def load_config(
    config_path: str | Path = "config_hyperliquid.json",
    testnet: bool = True,
) -> dict[str, Any]:
    """Load configuration using token_data.hyperliquid.setup().
    
    This function uses token_data.hyperliquid.setup() for all authentication
    and connection management, matching the behavior of hyperliquid_orders_demo.py.
    
    Args:
        config_path: Path to the token_data config file.
                    Defaults to 'config_hyperliquid.json' in the current directory.
        testnet: Whether to use testnet (True) or mainnet (False).
                Defaults to True for safety.
    
    Returns:
        Configuration dictionary with injected Hyperliquid objects:
        - account_address: The trading account address
        - _hyperliquid_info: Authenticated Info object
        - _hyperliquid_exchange: Authenticated Exchange object
        - base_url: The API URL being used
        - testnet: Boolean indicating testnet/mainnet
        
    Raises:
        ConfigurationError: If token_data is not installed or setup fails.
        
    Example:
        >>> config = load_config("config_hyperliquid.json", testnet=True)
        >>> exchange = HyperliquidExchange(config)
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(
            f"Config file not found: {config_path}\n"
            f"Please create a config file with your Hyperliquid credentials."
        )
    
    try:
        from token_data.hyperliquid import constants as hl_constants
        from token_data.hyperliquid import setup as token_data_setup
    except ImportError as e:
        raise ConfigurationError(
            "Failed to import token_data.hyperliquid. \n"
            "Please install token_data: pip install token_data\n"
            f"Original error: {e}"
        ) from e
    
    # Determine base_url based on testnet flag
    base_url = hl_constants.TESTNET_API_URL if testnet else hl_constants.MAINNET_API_URL
    
    logger.info("Initializing Hyperliquid via token_data.setup()")
    logger.info("Config file: %s", config_path)
    logger.info("Network: %s", "testnet" if testnet else "mainnet")
    logger.info("Base URL: %s", base_url)
    
    try:
        address, info, exchange = token_data_setup(
            base_url=base_url,
            skip_ws=True,
            config=str(config_path),
        )
        logger.info("Successfully initialized Hyperliquid for account: %s", address[:10] + "...")
    except Exception as e:
        raise ConfigurationError(
            f"Failed to initialize Hyperliquid via token_data.setup(): {e}\n"
            f"Please check your config file: {config_path}"
        ) from e
    
    return {
        'account_address': address,
        '_hyperliquid_info': info,
        '_hyperliquid_exchange': exchange,
        'base_url': base_url,
        'testnet': testnet,
    }




