"""Bridge module for generated and hand-written types."""
from importlib import import_module as _im
_impl = _im("types")
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("_")})
__all__ = [k for k in globals().keys() if not k.startswith("_")]
from __future__ import annotations
try:
    from fks_shared_python import types as _m  # type: ignore
    from fks_shared_python.types import *  # type: ignore  # noqa: F401,F403
    # Explicit re-export safeguard for dynamically added models
    MarketBar = _m.MarketBar  # type: ignore
except Exception:  # pragma: no cover
    from types import *  # type: ignore  # noqa: F401,F403
