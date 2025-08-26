"""Alias package so both `shared_python` and `fks_shared_python` import paths work.

This avoids breaking code that already imported using either naming convention.
All symbols are re-exported from the canonical `shared_python` package.
"""

from shared_python.config import *  # type: ignore  # noqa: F401,F403
from shared_python.exceptions import *  # type: ignore  # noqa: F401,F403
from shared_python.logging import *  # type: ignore  # noqa: F401,F403
from shared_python.types import *  # type: ignore  # noqa: F401,F403
from shared_python.utils import *  # type: ignore  # noqa: F401,F403

# Re-expose submodules so `from fks_shared_python.config import get_settings` works.
import sys as _sys
import shared_python.config as _cfg  # noqa: E402
import shared_python.exceptions as _exc  # noqa: E402
import shared_python.logging as _log  # noqa: E402
import shared_python.types as _types  # noqa: E402
import shared_python.utils as _utils  # noqa: E402

_sys.modules[__name__ + ".config"] = _cfg
_sys.modules[__name__ + ".exceptions"] = _exc
_sys.modules[__name__ + ".logging"] = _log
_sys.modules[__name__ + ".types"] = _types
_sys.modules[__name__ + ".utils"] = _utils

__all__ = sorted(set([  # type: ignore[var-annotated]
    *globals().keys()
]) - {"_cfg", "_exc", "_log", "_types", "_utils", "_sys"})
"""Namespace package wrapper exposing shared modules.

This allows imports like `import fks_shared_python.metrics` while keeping single-codebase files.
"""
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

# Import root modules via absolute name since they are part of the same distribution
import config as _config  # type: ignore
import utils as _utils  # type: ignore
import exceptions as _exceptions  # type: ignore
from . import risk as risk  # re-export risk namespace
from . import types as types  # re-export types namespace

load_config = _config.load_config  # type: ignore
get_settings = _config.get_settings  # type: ignore
get_risk_threshold = _utils.get_risk_threshold  # type: ignore
RiskLimitExceeded = _exceptions.RiskLimitExceeded  # type: ignore


__all__ = [
    "load_config",
    "get_settings",
    "get_risk_threshold",
    "RiskLimitExceeded",
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
    "apply_slippage",
    "simulate_gbm",
    "monte_carlo_pnl",
    "ood_score",
    "metrics",
    "simulation",
    "risk",
    "types",
]