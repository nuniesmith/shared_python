class RiskLimitExceeded(Exception):
    """Raised when a requested trade risk exceeds configured thresholds."""

    def __init__(self, requested: float, max_allowed: float):
        super().__init__(
            f"Risk limit exceeded: requested={requested:.4f} > max_allowed={max_allowed:.4f}"
        )
        self.requested = requested
        self.max_allowed = max_allowed
