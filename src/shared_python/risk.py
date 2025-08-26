from __future__ import annotations
try:
    from fks_shared_python import risk as _m  # type: ignore
except Exception:  # pragma: no cover
    import risk as _m  # type: ignore
from risk import *  # type: ignore  # noqa: F401,F403
