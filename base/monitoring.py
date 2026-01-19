"""Portfolio and performance monitoring base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional


class PerformanceMetrics:
    """Container for portfolio performance metrics.
    
    Attributes:
        account_value: Current total account value in USD.
        total_pnl: Total profit/loss.
        unrealized_pnl: Unrealized profit/loss from open positions.
        realized_pnl: Realized profit/loss from closed positions.
        volume: Trading volume.
        timestamp: When these metrics were captured.
    """
    
    def __init__(
        self,
        account_value: float,
        total_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        volume: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.account_value = account_value
        self.total_pnl = total_pnl
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.volume = volume
        self.timestamp = timestamp or datetime.now()
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(account_value={self.account_value:.2f}, "
            f"total_pnl={self.total_pnl:.2f}, volume={self.volume:.2f})"
        )


class PortfolioMonitor(ABC):
    """Abstract base class for portfolio monitoring and performance tracking.
    
    This class provides an interface for monitoring portfolio performance,
    tracking P&L, analyzing trades, and retrieving historical data.
    Implementations can leverage external packages like token_data for
    enhanced functionality.
    """
    
    @abstractmethod
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current portfolio performance metrics.
        
        Returns:
            PerformanceMetrics with current values.
        """
        pass
    
    @abstractmethod
    def get_portfolio_history(
        self,
        period: str = "allTime",
    ) -> dict[str, Any]:
        """Get historical portfolio data.
        
        Args:
            period: Time period for the data (e.g., "day", "week", "month", "allTime").
        
        Returns:
            Dictionary containing historical account value, P&L, and volume data.
        """
        pass
    
    @abstractmethod
    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of portfolio performance.
        
        Returns:
            Dictionary with performance metrics across different time periods.
        """
        pass
    
    def get_portfolio_dataframe(
        self,
        period: str = "allTime",
        data_type: str = "both",
    ) -> Any:
        """Get portfolio history as a pandas DataFrame.
        
        Args:
            period: Time period for the data.
            data_type: Type of data to include ("both", "account_value", "pnl").
        
        Returns:
            pandas.DataFrame with historical data.
            
        Raises:
            ImportError: If pandas is not installed.
            NotImplementedError: If the implementation doesn't support DataFrames.
        """
        raise NotImplementedError(
            "DataFrame support not implemented. Override this method in subclass."
        )
    
    @abstractmethod
    def get_trade_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        coin: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get historical trade data.
        
        Args:
            start_time: Optional start time for filtering trades.
            end_time: Optional end time for filtering trades.
            coin: Optional coin symbol to filter trades.
        
        Returns:
            List of trade dictionaries.
        """
        pass
    
    @abstractmethod
    def get_funding_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        coin: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get funding rate history for perpetual positions.
        
        Args:
            start_time: Optional start time for filtering.
            end_time: Optional end time for filtering.
            coin: Optional coin symbol to filter.
        
        Returns:
            List of funding payment dictionaries.
        """
        pass
    
    def analyze_performance(
        self,
        period: str = "allTime",
    ) -> dict[str, float]:
        """Analyze portfolio performance metrics.
        
        Args:
            period: Time period to analyze.
        
        Returns:
            Dictionary with calculated performance metrics like Sharpe ratio,
            max drawdown, win rate, etc.
        """
        raise NotImplementedError(
            "Performance analysis not implemented. Override this method in subclass."
        )
