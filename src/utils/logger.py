"""
ClauseInsight — Logger
=======================

Centralised logging configuration for the entire project.

Call setup_logging() once at application startup (in app.py) and
every module that does `logger = logging.getLogger(__name__)` will
automatically inherit the configured handlers and level.

WHY CENTRALISED LOGGING
------------------------
Without this, each module would need to configure its own handler,
leading to duplicate log lines, inconsistent formats, and no single
place to change log level for the whole app.

With this module:
  - One call in app.py sets up everything
  - LOG_LEVEL from .env controls verbosity for all modules at once
  - File logging is opt-in via LOG_FILE in .env
  - Streamlit's own logger is quieted to WARNING so it doesn't
    flood the console with framework noise during development
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


DEFAULT_LOG_LEVEL   = "INFO"
DEFAULT_LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
    fmt: str = DEFAULT_LOG_FORMAT,
    date_fmt: str = DEFAULT_DATE_FORMAT,
) -> None:
    """
    Configure logging for the entire ClauseInsight application.

    Call this once at startup in app.py (after load_dotenv()).
    All subsequent logging.getLogger(__name__) calls in any module
    will inherit this configuration automatically.

    Args:
        level:    Log level string: DEBUG, INFO, WARNING, ERROR.
                  Defaults to LOG_LEVEL env var, then INFO.
        log_file: Path to write logs to in addition to console.
                  Defaults to LOG_FILE env var. Empty = console only.
        fmt:      Log line format string.
        date_fmt: Timestamp format string.
    """
    resolved_level = (
        level or os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    ).upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    resolved_file = (
        log_file if log_file is not None
        else os.environ.get("LOG_FILE", "")
    )

    handlers: list[logging.Handler] = []

    # Console handler — always present
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    handlers.append(console)

    # File handler — only if LOG_FILE is set
    if resolved_file:
        log_path = Path(resolved_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        handlers.append(fh)

    # Configure root logger
    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)

    # Quiet noisy third-party loggers
    for name in ("streamlit", "watchdog", "urllib3", "httpx", "chromadb"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured — level: %s%s",
        resolved_level,
        f", file: {resolved_file}" if resolved_file else "",
    )


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper — returns a named logger.

    Usage in any module:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
