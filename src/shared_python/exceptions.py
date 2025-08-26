"""Bridge module for exceptions."""
from importlib import import_module as _im
_impl = _im("exceptions")
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("_")})
__all__ = [k for k in globals().keys() if not k.startswith("_")]
from __future__ import annotations
try:
    from fks_shared_python import exceptions as _m  # type: ignore
except Exception:  # pragma: no cover
    import exceptions as _m  # type: ignore
from exceptions import *  # type: ignore  # noqa: F401,F403
