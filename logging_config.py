# logging_config.py
import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure root logger with a console handler.

    Args:
        log_level: Optional log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   Defaults to INFO if not provided or invalid.
    """
    # Determine log level
    if log_level is not None and isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = logging.INFO

    # Formatter and handler
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers to prevent duplicate logs
    if not root.handlers:
        root.addHandler(handler)

    # Optional: silence overly verbose loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
