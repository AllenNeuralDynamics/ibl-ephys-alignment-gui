"""Custom exceptions for ephys alignment GUI."""

class RecoverableError(Exception):
    """Errors the GUI can recover from (e.g., bad input data)."""
    pass

class DataValidationError(RecoverableError):
    """Invalid data format or values."""
    pass

class AtlasError(RecoverableError):
    """Atlas-related errors (missing files, invalid coordinates)."""
    pass

class FatalError(Exception):
    """Errors that require operation to abort (but not GUI restart)."""
    pass

class ConfigurationError(FatalError):
    """Critical configuration missing or invalid."""
    pass
