from __future__ import annotations
try:
    from fks_shared_python import simulation as _m  # type: ignore
except Exception:  # pragma: no cover
    import simulation as _m  # type: ignore
from simulation import *  # type: ignore  # noqa: F401,F403
