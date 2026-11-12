"""
AeroSentinel Structured Logger
================================
Provides coloured console output, rotating file handler, JSON logging,
per-module log levels, and a performance-logging decorator.
"""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# ---------------------------------------------------------------------------
# Optional coloured console support
# ---------------------------------------------------------------------------
try:
    import colorlog  # type: ignore

    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOG_DIR = Path("data/logs")
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_BACKUP_COUNT = 5
_DEFAULT_LOG_LEVEL = logging.INFO

_COLOR_MAP: Dict[str, str] = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

F = TypeVar("F", bound=Callable[..., Any])

# Track whether root logging has been initialised
_logging_configured = False


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Attach any extra fields the caller passed
        standard_attrs = logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
        for key, val in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                payload[key] = val
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialise the root logger with console + rotating file handlers.

    Call this once at application startup before any :func:`get_logger` calls.

    Parameters
    ----------
    config:
        Optional dict (typically from :class:`~aerosentinel.utils.config_loader.ConfigLoader`)
        with the following keys (all optional):

        ``log_level`` (str)
            Root log level.  Default: ``"INFO"``.
        ``log_dir`` (str | Path)
            Directory where the rotating log file is written.
            Default: ``"data/logs"``.
        ``json_logging`` (bool)
            Use JSON formatter for the file handler.  Default: ``False``.
        ``max_bytes`` (int)
            Maximum size of a single log file in bytes.  Default: 10 MB.
        ``backup_count`` (int)
            Number of rotated log files to keep.  Default: 5.
        ``module_levels`` (dict)
            Per-module log levels, e.g. ``{"aerosentinel.cameras": "DEBUG"}``.
    """
    global _logging_configured

    cfg = config or {}

    root_level_str: str = str(cfg.get("log_level", "INFO")).upper()
    root_level: int = getattr(logging, root_level_str, _DEFAULT_LOG_LEVEL)

    log_dir = Path(cfg.get("log_dir", _DEFAULT_LOG_DIR))
    json_logging: bool = bool(cfg.get("json_logging", False))
    max_bytes: int = int(cfg.get("max_bytes", _DEFAULT_MAX_BYTES))
    backup_count: int = int(cfg.get("backup_count", _DEFAULT_BACKUP_COUNT))
    module_levels: Dict[str, str] = dict(cfg.get("module_levels", {}))

    # Ensure log directory exists
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.basicConfig(level=root_level)
        logging.getLogger(__name__).warning(
            "Cannot create log directory %s: %s — using stderr only", log_dir, exc
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Remove pre-existing handlers to avoid duplication on re-calls
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        h.close()

    # --- Console handler ---
    console_handler = _make_console_handler(root_level)
    root_logger.addHandler(console_handler)

    # --- Rotating file handler ---
    try:
        log_file = log_dir / "aerosentinel.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(root_level)
        if json_logging:
            file_handler.setFormatter(_JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
        root_logger.addHandler(file_handler)
    except OSError as exc:
        root_logger.warning("Cannot open log file: %s", exc)

    # --- Per-module levels ---
    for module_name, level_str in module_levels.items():
        level = getattr(logging, level_str.upper(), None)
        if level is not None:
            logging.getLogger(module_name).setLevel(level)

    _logging_configured = True
    root_logger.debug("Logging initialised (level=%s, dir=%s)", root_level_str, log_dir)


def get_logger(name: str) -> logging.Logger:
    """
    Return a :class:`logging.Logger` for *name*.

    If :func:`setup_logging` has not been called yet, a minimal
    ``basicConfig`` fallback is applied so the application does not silently
    swallow log records.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    global _logging_configured
    if not _logging_configured:
        logging.basicConfig(
            level=_DEFAULT_LOG_LEVEL,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        _logging_configured = True
    return logging.getLogger(name)


def log_performance(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    label: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator that logs the wall-clock execution time of the wrapped function.

    Parameters
    ----------
    logger:
        Logger instance to use.  Defaults to a logger named after the
        decorated function's module.
    level:
        Log level for the timing message.  Default: ``DEBUG``.
    label:
        Custom label for the log message.  Defaults to the function's
        qualified name.

    Returns
    -------
    Callable
        The wrapped function.

    Example::

        @log_performance(label="inference")
        def run_inference(frame):
            ...
    """

    def decorator(func: F) -> F:
        _logger = logger or get_logger(func.__module__)
        _label = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                _logger.log(level, "[perf] %s took %.2f ms", _label, elapsed_ms)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_console_handler(level: int) -> logging.Handler:
    """Create a console handler with optional colour support."""
    handler = logging.StreamHandler()
    handler.setLevel(level)

    if _HAS_COLORLOG:
        fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            log_colors=_COLOR_MAP,
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    handler.setFormatter(fmt)
    return handler
