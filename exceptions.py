"""Custom exceptions for the reporter package."""


class ReporterError(Exception):
    """Base exception for reporter package."""
    pass


class ConfigurationError(ReporterError):
    """Raised when there's a configuration error."""
    pass


class ExchangeError(ReporterError):
    """Raised when there's an error communicating with the exchange."""
    pass


class ReportGenerationError(ReporterError):
    """Raised when there's an error generating a report."""
    pass
