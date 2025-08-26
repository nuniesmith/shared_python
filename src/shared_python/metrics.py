from __future__ import annotations
try:
    from fks_shared_python import metrics as _m  # type: ignore
except Exception:  # pragma: no cover
    import metrics as _m  # type: ignore
from metrics import *  # type: ignore  # noqa: F401,F403
