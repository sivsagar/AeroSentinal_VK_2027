"""
Unit tests for src/aerosentinel/utils/logger.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aerosentinel.utils.logger import get_logger, setup_logging, log_performance, _logging_configured


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_logging():
    """Remove all handlers from root logger before and after each test."""
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level

    # Patch the global flag so setup_logging re-runs cleanly
    import aerosentinel.utils.logger as _mod
    old_flag = _mod._logging_configured
    _mod._logging_configured = False

    yield

    # Teardown
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in old_handlers:
        root.addHandler(h)
    root.setLevel(old_level)
    _mod._logging_configured = old_flag


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetLogger:
    """get_logger() returns a properly configured Logger."""

    def test_returns_logger_instance(self) -> None:
        lgr = get_logger("test.module")
        assert isinstance(lgr, logging.Logger)

    def test_name_matches(self) -> None:
        lgr = get_logger("aerosentinel.cameras")
        assert lgr.name == "aerosentinel.cameras"

    def test_different_names_are_different_loggers(self) -> None:
        a = get_logger("mod.a")
        b = get_logger("mod.b")
        assert a is not b

    def test_same_name_returns_same_logger(self) -> None:
        a = get_logger("mod.same")
        b = get_logger("mod.same")
        assert a is b


class TestSetupLogging:
    """setup_logging() initialises handlers on the root logger."""

    def test_adds_handlers(self, tmp_path: Path) -> None:
        setup_logging({"log_dir": str(tmp_path), "log_level": "DEBUG"})
        root = logging.getLogger()
        assert len(root.handlers) >= 1  # at least console handler

    def test_log_file_created(self, tmp_path: Path) -> None:
        setup_logging({"log_dir": str(tmp_path), "log_level": "INFO"})
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        assert log_files[0].name == "aerosentinel.log"

    def test_root_level_set(self, tmp_path: Path) -> None:
        setup_logging({"log_dir": str(tmp_path), "log_level": "WARNING"})
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_debug_level(self, tmp_path: Path) -> None:
        setup_logging({"log_dir": str(tmp_path), "log_level": "DEBUG"})
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_per_module_level(self, tmp_path: Path) -> None:
        setup_logging({
            "log_dir": str(tmp_path),
            "log_level": "INFO",
            "module_levels": {"aerosentinel.cameras": "DEBUG"},
        })
        cam_logger = logging.getLogger("aerosentinel.cameras")
        assert cam_logger.level == logging.DEBUG

    def test_missing_log_dir_does_not_crash(self, tmp_path: Path) -> None:
        # Point to a path we cannot create (non-existent root drive on Linux)
        # On Linux /proc is read-only; use a safely bad path
        bad_dir = "/this_path_cannot_be_created_12345/logs"
        # Should not raise — falls back gracefully
        setup_logging({"log_dir": bad_dir, "log_level": "INFO"})

    def test_idempotent_handlers(self, tmp_path: Path) -> None:
        """Calling setup_logging twice should not duplicate handlers."""
        setup_logging({"log_dir": str(tmp_path), "log_level": "INFO"})
        count_first = len(logging.getLogger().handlers)
        setup_logging({"log_dir": str(tmp_path), "log_level": "INFO"})
        count_second = len(logging.getLogger().handlers)
        assert count_second == count_first


class TestLogLevels:
    """Log level filtering works as expected."""

    def test_debug_message_emitted_at_debug_level(self, tmp_path: Path) -> None:
        setup_logging({"log_dir": str(tmp_path), "log_level": "DEBUG"})
        lgr = get_logger("test.levels")

        records: list = []

        class _Cap(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        cap = _Cap()
        lgr.addHandler(cap)
        lgr.debug("hello debug")
        assert any(r.levelno == logging.DEBUG for r in records)

    def test_debug_message_suppressed_at_info_level(self, tmp_path: Path) -> None:
        setup_logging({"log_dir": str(tmp_path), "log_level": "INFO"})
        lgr = get_logger("test.suppress")

        records: list = []

        class _Cap(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        cap = _Cap(level=logging.INFO)
        lgr.addHandler(cap)
        lgr.setLevel(logging.INFO)
        lgr.debug("should be suppressed")
        assert len(records) == 0


class TestRotatingFileHandler:
    """Rotating file handler is attached with correct parameters."""

    def test_rotating_handler_present(self, tmp_path: Path) -> None:
        import logging.handlers as lh

        setup_logging({"log_dir": str(tmp_path), "log_level": "INFO"})
        root = logging.getLogger()
        rotating = [h for h in root.handlers if isinstance(h, lh.RotatingFileHandler)]
        assert len(rotating) == 1

    def test_rotating_handler_max_bytes(self, tmp_path: Path) -> None:
        import logging.handlers as lh

        setup_logging({
            "log_dir": str(tmp_path),
            "log_level": "INFO",
            "max_bytes": 5000,
            "backup_count": 3,
        })
        root = logging.getLogger()
        rotating = [h for h in root.handlers if isinstance(h, lh.RotatingFileHandler)]
        assert rotating[0].maxBytes == 5000
        assert rotating[0].backupCount == 3


class TestLogPerformanceDecorator:
    """log_performance decorator records timing."""

    def test_function_still_returns_value(self) -> None:
        @log_performance()
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_function_executes_correctly(self) -> None:
        results = []

        @log_performance()
        def side_effect() -> None:
            results.append(1)

        side_effect()
        assert results == [1]

    def test_timing_message_logged(self) -> None:
        records: list = []
        lgr = get_logger("test.perf")

        class _Cap(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        cap = _Cap(level=logging.DEBUG)
        lgr.addHandler(cap)
        lgr.setLevel(logging.DEBUG)

        @log_performance(logger=lgr, label="myop")
        def noop() -> None:
            pass

        noop()
        assert any("myop" in r.getMessage() and "ms" in r.getMessage() for r in records)
