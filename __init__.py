"""Trading Account Performance Reporter Package.

This package provides comprehensive reporting for trading account performance,
including AUM tracking, P&L analysis, trade cost analysis, and funding costs.
"""

__version__ = "1.0.0"

from .exceptions import (
    ConfigurationError,
    ExchangeError,
    ReportGenerationError,
    ReporterError,
)

__all__ = [
    'ConfigurationError',
    'ExchangeError',
    'ReportGenerationError',
    'ReporterError',
]
