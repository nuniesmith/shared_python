import logging
from .config import get_settings


_LOGGER_INITIALIZED = False


def init_logging(force: bool = False) -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED and not force:
        return
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGER_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    if not _LOGGER_INITIALIZED:
        init_logging()
    return logging.getLogger(name)
