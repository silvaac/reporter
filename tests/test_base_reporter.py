"""Unit tests for base.reporter module."""

import pytest
from base.reporter import BaseReporter


class TestBaseReporter:
    """Test BaseReporter abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseReporter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseReporter()
    
    def test_abstract_methods_exist(self):
        """Test that all abstract methods are defined."""
        abstract_methods = BaseReporter.__abstractmethods__
        expected_methods = {
            'generate_aum_data',
            'generate_performance_data',
            'generate_trade_analysis',
            'generate_funding_analysis',
            'generate_report_data',
            'create_visualizations',
            'generate_html_report'
        }
        assert expected_methods.issubset(abstract_methods)
    
    def test_subclass_must_implement_all_methods(self):
        """Test that subclass must implement all abstract methods."""
        
        class IncompleteReporter(BaseReporter):
            """Incomplete implementation for testing."""
            pass
        
        with pytest.raises(TypeError):
            IncompleteReporter()
