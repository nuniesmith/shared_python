from __future__ import annotations

from .config import get_settings
from .exceptions import RiskLimitExceeded


def get_risk_threshold(equity: float | int) -> float:
    s = get_settings()
    return float(equity) * float(s.RISK_MAX_PER_TRADE)


def enforce_risk(requested: float, equity: float | int) -> float:
    max_allowed = get_risk_threshold(equity)
    if requested > max_allowed:
        raise RiskLimitExceeded(requested=requested, max_allowed=max_allowed)
    return requested
