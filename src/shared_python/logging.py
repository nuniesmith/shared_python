"""Bridge module for logging helpers."""
from importlib import import_module as _im
_impl = _im("logging")
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith("_")})
__all__ = [k for k in globals().keys() if not k.startswith("_")]
from __future__ import annotations

import logging
import json
import os
from datetime import datetime, UTC
from typing import Any, Dict

try:  # leverage alias config (which delegates to real config)
    from shared_python.config import get_settings  # type: ignore
except Exception:  # pragma: no cover
    from fks_shared_python import load_config as get_settings  # type: ignore

_LOGGER_INITIALIZED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in {"name","msg","args","levelname","levelno","pathname","filename","module","exc_info","exc_text","stack_info","lineno","funcName","created","msecs","relativeCreated","thread","threadName","process","processName"}:
                continue
            try:
                json.dumps(v)
                base[k] = v
            except Exception:
                base[k] = repr(v)
        return json.dumps(base, separators=(",", ":"))


def init_logging(force: bool = False) -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED and not force:
        return
    if force:
        # remove existing handlers to allow reconfiguration (important for tests)
        for h in list(logging.root.handlers):
            logging.root.removeHandler(h)
    # Fallback settings if config unavailable
    try:
        settings = get_settings()
        level_name = getattr(settings, "LOG_LEVEL", "INFO")
    except Exception:  # pragma: no cover
        level_name = os.getenv("LOG_LEVEL", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    json_logs = os.getenv("FKS_JSON_LOGS", "0").lower() in {"1", "true", "yes"}
    if json_logs:
        import sys
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logging.basicConfig(level=level, handlers=[handler])
    else:
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _LOGGER_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    if not _LOGGER_INITIALIZED:
        init_logging()
    return logging.getLogger(name)


__all__ = ["init_logging", "get_logger"]
