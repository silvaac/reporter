"""Unit tests for base.monitoring module."""

import pytest
from datetime import datetime
from base.monitoring import PerformanceMetrics, PortfolioMonitor


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        metrics = PerformanceMetrics(account_value=10000.0)
        assert metrics.account_value == 10000.0
        assert metrics.total_pnl == 0.0
        assert metrics.unrealized_pnl == 0.0
        assert metrics.realized_pnl == 0.0
        assert metrics.volume == 0.0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            account_value=15000.0,
            total_pnl=5000.0,
            unrealized_pnl=1000.0,
            realized_pnl=4000.0,
            volume=50000.0,
            timestamp=timestamp
        )
        assert metrics.account_value == 15000.0
        assert metrics.total_pnl == 5000.0
        assert metrics.unrealized_pnl == 1000.0
        assert metrics.realized_pnl == 4000.0
        assert metrics.volume == 50000.0
        assert metrics.timestamp == timestamp
    
    def test_repr(self):
        """Test string representation."""
        metrics = PerformanceMetrics(
            account_value=10000.0,
            total_pnl=500.0,
            volume=25000.0
        )
        repr_str = repr(metrics)
        assert "PerformanceMetrics" in repr_str
        assert "10000.00" in repr_str
        assert "500.00" in repr_str
        assert "25000.00" in repr_str


class TestPortfolioMonitor:
    """Test PortfolioMonitor abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that PortfolioMonitor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PortfolioMonitor()
    
    def test_abstract_methods_exist(self):
        """Test that abstract methods are defined."""
        abstract_methods = PortfolioMonitor.__abstractmethods__
        expected_methods = {
            'get_current_metrics',
            'get_portfolio_history',
            'get_portfolio_summary',
            'get_trade_history',
            'get_funding_history'
        }
        assert expected_methods.issubset(abstract_methods)
