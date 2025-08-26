class RiskLimitExceeded(Exception):
    """Raised when a requested trade risk exceeds configured thresholds."""

    def __init__(self, requested: float, max_allowed: float):
        super().__init__(
            f"Risk limit exceeded: requested={requested:.4f} > max_allowed={max_allowed:.4f}"
        )
        self.requested = requested
        self.max_allowed = max_allowed


class DataFetchError(Exception):
    """Raised when an external data source fetch fails."""

    def __init__(self, source: str, detail: str | None = None):
        msg = f"Data fetch failed from source='{source}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.source = source
        self.detail = detail


class DataValidationError(Exception):
    """Raised when ingested or engineered data violates expected schema or constraints."""

    def __init__(self, issue: str, context: dict | None = None):
        msg = f"Data validation error: {issue}"
        super().__init__(msg)
        self.issue = issue
        self.context = context or {}


class ModelError(Exception):
    """Generic model-related error across training/inference phases."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}

__all__ = [
    "RiskLimitExceeded",
    "DataFetchError",
    "DataValidationError",
    "ModelError",
]
