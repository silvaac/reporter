"""Unit tests for exceptions module."""

import pytest
from exceptions import (
    ReporterError,
    ConfigurationError,
    ExchangeError,
    ReportGenerationError,
)


class TestExceptions:
    """Test custom exception classes."""
    
    def test_reporter_error_inheritance(self):
        """Test ReporterError is a base Exception."""
        assert issubclass(ReporterError, Exception)
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from ReporterError."""
        assert issubclass(ConfigurationError, ReporterError)
        assert issubclass(ConfigurationError, Exception)
    
    def test_exchange_error_inheritance(self):
        """Test ExchangeError inherits from ReporterError."""
        assert issubclass(ExchangeError, ReporterError)
        assert issubclass(ExchangeError, Exception)
    
    def test_report_generation_error_inheritance(self):
        """Test ReportGenerationError inherits from ReporterError."""
        assert issubclass(ReportGenerationError, ReporterError)
        assert issubclass(ReportGenerationError, Exception)
    
    def test_reporter_error_message(self):
        """Test ReporterError can be raised with a message."""
        msg = "Test error message"
        with pytest.raises(ReporterError, match=msg):
            raise ReporterError(msg)
    
    def test_configuration_error_message(self):
        """Test ConfigurationError can be raised with a message."""
        msg = "Configuration failed"
        with pytest.raises(ConfigurationError, match=msg):
            raise ConfigurationError(msg)
    
    def test_exchange_error_message(self):
        """Test ExchangeError can be raised with a message."""
        msg = "Exchange API failed"
        with pytest.raises(ExchangeError, match=msg):
            raise ExchangeError(msg)
    
    def test_report_generation_error_message(self):
        """Test ReportGenerationError can be raised with a message."""
        msg = "Report generation failed"
        with pytest.raises(ReportGenerationError, match=msg):
            raise ReportGenerationError(msg)
