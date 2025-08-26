"""FKS shared python utilities.

Primary public surface consolidated here for convenient star-import in small scripts.
"""
from .config import load_config, get_settings
from .utils import get_risk_threshold
from .exceptions import RiskLimitExceeded
from .metrics import (
    Trade,
    win_rate_with_costs,
    cumulative_equity,
    max_drawdown,
    calmar_ratio,
    information_ratio,
    cvar,
    sharpe_ratio,
    downside_deviation,
    sortino_ratio,
)
from .simulation import (
    apply_slippage,
    simulate_gbm,
    monte_carlo_pnl,
    ood_score,
)

__all__ = [
    # config
    "load_config",
    "get_settings",
    # risk
    "get_risk_threshold",
    "RiskLimitExceeded",
    # metrics
    "Trade",
    "win_rate_with_costs",
    "cumulative_equity",
    "max_drawdown",
    "calmar_ratio",
    "information_ratio",
    "cvar",
    "sharpe_ratio",
    "downside_deviation",
    "sortino_ratio",
    # simulation
    "apply_slippage",
    "simulate_gbm",
    "monte_carlo_pnl",
    "ood_score",
]

