"""Shim forwarding to central shared package.

Deprecated: update imports to `shared_python` or `fks_shared_python` from root shared module.
"""
from fks_shared_python import *  # type: ignore  # noqa: F401,F403

__all__ = []  # star import kept for backwards compatibility only
