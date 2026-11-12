"""
AeroSentinel — Main Entry Point
=================================
Parses CLI arguments, loads configuration, initialises subsystems,
and runs the main event loop.
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure src/ is on the import path when run as a script
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from aerosentinel.utils.config_loader import ConfigLoader, ConfigError
from aerosentinel.utils.logger import get_logger, setup_logging
from aerosentinel.utils.performance import PerformanceMonitor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERSION = "0.1.0"
_TEAM_ID = "VK26-AU-PAI-0399"
_BANNER = r"""
  ___              ____            _   _            _
 / _ \  ___  _ __|  _ \ ___ _ __ | |_(_)_ __   ___| |
| | | |/ _ \| '__| |_) / _ \ '_ \| __| | '_ \ / _ \ |
| |_| |  __/| |  |  _ <  __/ |_) | |_| | | | |  __/ |
 \___/ \___||_|  |_| \_\___| .__/ \__|_|_| |_|\___|_|
                             |_|
  AeroSentinel UAV Disaster Search & Rescue System
  Version {version}  |  Team {team}
"""

_CONFIG_FILES = [
    "system.yaml",
    "camera.yaml",
    "depth.yaml",
    "ai.yaml",
    "fusion.yaml",
    "navigation.yaml",
    "mavlink.yaml",
    "mission.yaml",
]

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_running = True
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv:
        Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="aerosentinel",
        description="AeroSentinel UAV Disaster Search & Rescue System",
    )
    parser.add_argument(
        "--config",
        metavar="DIR",
        default="config/",
        help="Path to the config directory (default: config/).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/simulation mode (no real hardware required).",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the log level set in system.yaml.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the web dashboard.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def _print_banner(version: str = _VERSION, team: str = _TEAM_ID) -> None:
    """Print the ASCII startup banner."""
    print(_BANNER.format(version=version, team=team))
    print(f"  Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python   : {platform.python_version()}")
    print(f"  PID      : {os.getpid()}")
    print()


def _load_config(config_dir: str) -> ConfigLoader:
    """
    Load all YAML config files from *config_dir*.

    Parameters
    ----------
    config_dir:
        Directory that contains the YAML files.

    Returns
    -------
    ConfigLoader
        Populated configuration object.

    Raises
    ------
    SystemExit
        If a file is missing or contains invalid YAML.
    """
    loader = ConfigLoader(config_dir=config_dir)
    for filename in _CONFIG_FILES:
        path = Path(config_dir) / filename
        if path.exists():
            try:
                loader.load(filename)
                logger.debug("Loaded %s", filename)
            except ConfigError as exc:
                logger.warning("Could not load %s: %s", filename, exc)
        else:
            logger.warning("Config file not found, skipping: %s", path)
    return loader


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


def _register_signal_handlers() -> None:
    """Register SIGINT and SIGTERM handlers for graceful shutdown."""
    global _running

    def _handle_signal(signum: int, frame: Any) -> None:
        global _running
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown…", sig_name)
        _running = False

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (OSError, AttributeError):
        # SIGTERM not available on Windows
        pass


# ---------------------------------------------------------------------------
# Module init stubs
# ---------------------------------------------------------------------------


def _init_cameras(cfg: ConfigLoader, mock: bool) -> Optional[Any]:
    """Initialise camera subsystem (stub)."""
    try:
        logger.info("Initialising camera subsystem (mock=%s)…", mock)
        # TODO: Phase 2 — import and initialise CameraManager
        return None
    except Exception as exc:
        logger.error("Camera init failed: %s", exc)
        return None


def _init_perception(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise AI perception pipeline (stub)."""
    try:
        logger.info("Initialising AI perception pipeline…")
        # TODO: Phase 3 — import and initialise Detector
        return None
    except Exception as exc:
        logger.error("Perception init failed: %s", exc)
        return None


def _init_mission(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise mission manager (stub)."""
    try:
        logger.info("Initialising mission manager…")
        # TODO: Phase 4 — import and initialise MissionManager
        return None
    except Exception as exc:
        logger.error("Mission manager init failed: %s", exc)
        return None


def _init_dashboard(cfg: ConfigLoader, enabled: bool) -> Optional[Any]:
    """Initialise web dashboard (stub)."""
    if not enabled:
        logger.info("Dashboard disabled (--no-dashboard).")
        return None
    try:
        logger.info("Initialising web dashboard…")
        # TODO: Phase 5 — import and start Dashboard
        return None
    except Exception as exc:
        logger.error("Dashboard init failed: %s", exc)
        return None


def _init_mavlink(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise MAVLink telemetry client (stub)."""
    mavlink_enabled = cfg.get("mavlink.enabled", False)
    if not mavlink_enabled:
        logger.info("MAVLink disabled in config.")
        return None
    try:
        logger.info("Initialising MAVLink client…")
        # TODO: Phase 6 — import and initialise MAVLinkClient
        return None
    except Exception as exc:
        logger.error("MAVLink init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------


def _main_loop(monitor: PerformanceMonitor, loop_hz: float = 10.0) -> None:
    """
    Main application event loop.

    Parameters
    ----------
    monitor:
        Active :class:`PerformanceMonitor` instance.
    loop_hz:
        Target loop frequency in Hz.
    """
    global _running
    period = 1.0 / max(loop_hz, 1.0)
    logger.info("Main loop running at %.1f Hz (press Ctrl-C to stop)…", loop_hz)

    while _running:
        t0 = time.perf_counter()

        # TODO: Phase 2–6 — call subsystem update methods here
        monitor.tick()

        elapsed = time.perf_counter() - t0
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            monitor.drop_frame()
            logger.debug("Loop over-budget by %.1f ms", abs(sleep_time) * 1000)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """
    AeroSentinel application entry point.

    Parameters
    ----------
    argv:
        CLI argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 = success, non-zero = error).
    """
    args = _parse_args(argv)

    # --- Banner ---
    _print_banner()

    # --- Config ---
    cfg = _load_config(args.config)

    # Apply CLI overrides to config
    if args.mock:
        cfg.set("system.mock_mode", True)
    if args.log_level:
        cfg.set("system.log_level", args.log_level)

    # --- Logging ---
    log_config: Dict[str, Any] = {
        "log_level": cfg.get("system.log_level", "INFO"),
        "log_dir": cfg.get("system.log_dir", "data/logs"),
    }
    setup_logging(log_config)
    logger.info("AeroSentinel v%s starting up…", _VERSION)
    logger.info(
        "Config dir: %s | Mock mode: %s",
        args.config,
        cfg.get("system.mock_mode", False),
    )

    # --- Signal handlers ---
    _register_signal_handlers()

    # --- Performance monitor ---
    monitor = PerformanceMonitor(enable_background=True)
    monitor.start()

    try:
        # --- Module initialisation ---
        _init_cameras(cfg, mock=cfg.get("system.mock_mode", False))
        _init_perception(cfg)
        _init_mission(cfg)
        _init_mavlink(cfg)
        _init_dashboard(cfg, enabled=not args.no_dashboard)

        logger.info("All modules initialised. Starting main loop.")

        # --- Main loop ---
        _main_loop(monitor)

    except Exception as exc:
        logger.exception("Fatal error in main: %s", exc)
        return 1
    finally:
        monitor.stop()
        logger.info("AeroSentinel shutdown complete.")
        report = monitor.report()
        logger.info(
            "Session summary — uptime: %.1fs | FPS: %.2f | dropped: %d",
            report["uptime_s"],
            report["fps"],
            report["dropped_frames"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
