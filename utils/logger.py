"""
utils/logger.py

Centralised logging setup.

Usage:
    from utils.logger import setup_logger
    log = setup_logger(__name__)

    log.debug("Detailed trace info")
    log.info("Something happened")
    log.warning("Something looks off")
    log.error("Something failed")
"""

import logging
import sys

from config import settings

_DEFAULT_LEVEL = "INFO"
_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = None) -> logging.Logger:
    """
    Return a named logger with a stdout handler.
    Safe to call multiple times — handlers are only added once.
    """
    logger = logging.getLogger(name or "app")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    # Fall back to INFO if LOG_LEVEL is not set or invalid
    level_name = (settings.LOG_LEVEL or _DEFAULT_LEVEL).upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO
    logger.setLevel(level)

    return logger


# Module-level convenience logger
logger = setup_logger("app")
