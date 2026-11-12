"""
AeroSentinel Performance Monitor
==================================
Tracks FPS, latency, CPU/RAM usage, temperature, and dropped frames.
Optionally runs a background monitoring thread and exports metrics as dict/JSON.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logging.getLogger(__name__).warning(
        "psutil not installed — CPU/RAM monitoring disabled."
    )

logger = logging.getLogger(__name__)

_THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


# ---------------------------------------------------------------------------
# FPS Tracker
# ---------------------------------------------------------------------------


class FPSTracker:
    """
    Rolling-window average frames-per-second tracker.

    Parameters
    ----------
    window_size:
        Number of frame timestamps to retain for the rolling average.
        A larger window smooths out spikes; a smaller window is more reactive.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        self._timestamps: Deque[float] = collections.deque(maxlen=window_size)
        self._lock = threading.Lock()

    def tick(self) -> None:
        """Record the timestamp of the current frame."""
        with self._lock:
            self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Current rolling-average FPS (0.0 if fewer than 2 frames recorded)."""
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        with self._lock:
            self._timestamps.clear()


# ---------------------------------------------------------------------------
# Latency Tracker
# ---------------------------------------------------------------------------


class LatencyTracker:
    """
    Records latency samples (in milliseconds) and computes statistics.

    Parameters
    ----------
    name:
        Human-readable name for this tracker (e.g. ``"inference"``).
    max_samples:
        Maximum number of samples to retain.
    """

    def __init__(self, name: str, max_samples: int = 100) -> None:
        self.name = name
        self._samples: Deque[float] = collections.deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        """
        Store a latency measurement.

        Parameters
        ----------
        latency_ms:
            Latency in milliseconds.
        """
        with self._lock:
            self._samples.append(latency_ms)

    def stats(self) -> Dict[str, float]:
        """
        Return a dict with ``min``, ``max``, ``mean``, ``last`` (ms) and
        ``count``.  All float values are 0.0 when no samples exist.
        """
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "last": 0.0, "count": 0}
        return {
            "min": min(samples),
            "max": max(samples),
            "mean": sum(samples) / len(samples),
            "last": samples[-1],
            "count": len(samples),
        }

    def reset(self) -> None:
        """Clear all samples."""
        with self._lock:
            self._samples.clear()


# ---------------------------------------------------------------------------
# System resource readers
# ---------------------------------------------------------------------------


def read_cpu_usage_percent(interval: float = 0.1) -> Optional[float]:
    """
    Return instantaneous CPU usage as a percentage across all cores.

    Returns
    -------
    float or None
        Percentage value, or *None* if psutil is unavailable.
    """
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.cpu_percent(interval=interval)
    except Exception as exc:
        logger.debug("CPU usage read error: %s", exc)
        return None


def read_ram_usage() -> Dict[str, Optional[float]]:
    """
    Return RAM usage statistics.

    Returns
    -------
    dict with keys ``total_mb``, ``used_mb``, ``available_mb``, ``percent``.
    All values are *None* if psutil is unavailable.
    """
    if not _HAS_PSUTIL:
        return {"total_mb": None, "used_mb": None, "available_mb": None, "percent": None}
    try:
        vm = psutil.virtual_memory()
        mb = 1024 * 1024
        return {
            "total_mb": vm.total / mb,
            "used_mb": vm.used / mb,
            "available_mb": vm.available / mb,
            "percent": vm.percent,
        }
    except Exception as exc:
        logger.debug("RAM usage read error: %s", exc)
        return {"total_mb": None, "used_mb": None, "available_mb": None, "percent": None}


def read_cpu_temperature() -> Optional[float]:
    """
    Read CPU temperature in degrees Celsius.

    Uses ``/sys/class/thermal/thermal_zone0/temp`` (RPi / Linux).
    Falls back to ``psutil.sensors_temperatures()`` on other platforms.

    Returns
    -------
    float or None
        Temperature in °C, or *None* on unsupported platforms or on error.
    """
    # Raspberry Pi / generic Linux thermal zone
    if _THERMAL_PATH.exists():
        try:
            raw = _THERMAL_PATH.read_text(encoding="utf-8").strip()
            return int(raw) / 1000.0
        except (OSError, ValueError) as exc:
            logger.debug("Thermal zone read error: %s", exc)

    # psutil fallback (macOS, some Linux)
    if _HAS_PSUTIL and hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            # Prefer coretemp, then first available
            for key in ("coretemp", "cpu_thermal", "cpu-thermal"):
                if key in temps and temps[key]:
                    return temps[key][0].current
            for entries in temps.values():
                if entries:
                    return entries[0].current
        except Exception as exc:
            logger.debug("psutil temperature read error: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Performance Monitor
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """
    Central performance monitor for the AeroSentinel system.

    Aggregates FPS, latency, CPU, RAM, temperature, and dropped-frame data.
    Optionally runs a background thread that samples system resources
    periodically.

    Parameters
    ----------
    fps_window:
        Rolling-window size for FPS calculation.
    sample_interval_s:
        How often the background thread samples CPU/RAM/temperature.
    enable_background:
        Start the background sampling thread on :meth:`start`.

    Example::

        monitor = PerformanceMonitor()
        monitor.start()
        monitor.fps_tracker.tick()
        monitor.latency("inference").record(220.5)
        print(monitor.report())
        monitor.stop()
    """

    def __init__(
        self,
        fps_window: int = 30,
        sample_interval_s: float = 1.0,
        enable_background: bool = False,
    ) -> None:
        self.fps_tracker = FPSTracker(window_size=fps_window)
        self._latency_trackers: Dict[str, LatencyTracker] = {}
        self._latency_lock = threading.Lock()

        self._sample_interval = sample_interval_s
        self._enable_background = enable_background

        self._dropped_frames: int = 0
        self._dropped_lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Cached system samples (updated by background thread or on demand)
        self._last_cpu_percent: Optional[float] = None
        self._last_ram: Dict[str, Optional[float]] = {}
        self._last_temp_c: Optional[float] = None
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start performance monitoring.

        If *enable_background* is ``True``, launches a daemon thread that
        periodically samples CPU, RAM, and temperature.
        """
        self._start_time = time.perf_counter()
        self._running = True
        if self._enable_background:
            self._thread = threading.Thread(
                target=self._background_loop,
                name="perf-monitor",
                daemon=True,
            )
            self._thread.start()
            logger.debug("PerformanceMonitor background thread started")

    def stop(self) -> None:
        """Stop the background monitoring thread (if running)."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._sample_interval * 2)
            self._thread = None
        logger.debug("PerformanceMonitor stopped")

    # ------------------------------------------------------------------
    # Frame tracking
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Record a processed frame (shortcut to ``fps_tracker.tick()``)."""
        self.fps_tracker.tick()

    def drop_frame(self, count: int = 1) -> None:
        """
        Increment the dropped-frame counter.

        Parameters
        ----------
        count:
            Number of frames to add.  Default: 1.
        """
        with self._dropped_lock:
            self._dropped_frames += count

    @property
    def dropped_frames(self) -> int:
        """Total number of dropped frames since the last :meth:`reset`."""
        with self._dropped_lock:
            return self._dropped_frames

    # ------------------------------------------------------------------
    # Latency tracking
    # ------------------------------------------------------------------

    def latency(self, name: str) -> LatencyTracker:
        """
        Return (or create) the :class:`LatencyTracker` for *name*.

        Parameters
        ----------
        name:
            Identifier (e.g. ``"inference"``, ``"processing"``, ``"fusion"``).
        """
        with self._latency_lock:
            if name not in self._latency_trackers:
                self._latency_trackers[name] = LatencyTracker(name)
            return self._latency_trackers[name]

    # ------------------------------------------------------------------
    # System readings
    # ------------------------------------------------------------------

    def sample_system(self) -> None:
        """Take a single sample of CPU, RAM, and temperature (synchronous)."""
        self._last_cpu_percent = read_cpu_usage_percent(interval=0.0)
        self._last_ram = read_ram_usage()
        self._last_temp_c = read_cpu_temperature()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """
        Return a snapshot of all performance metrics as a plain dict.

        Keys
        ----
        ``fps``
            Current rolling-average FPS.
        ``dropped_frames``
            Total dropped frame count.
        ``uptime_s``
            Seconds since :meth:`start` was called.
        ``cpu_percent``
            Last sampled CPU usage percentage (or *None*).
        ``ram``
            Dict from :func:`read_ram_usage`.
        ``cpu_temp_c``
            Last sampled CPU temperature in °C (or *None*).
        ``latency``
            Dict mapping tracker name → stats dict.
        """
        # On-demand system sample if background thread is not running
        if not self._enable_background:
            self.sample_system()

        with self._latency_lock:
            latency_report = {
                name: tracker.stats()
                for name, tracker in self._latency_trackers.items()
            }

        uptime = (
            time.perf_counter() - self._start_time
            if self._start_time is not None
            else 0.0
        )

        return {
            "fps": round(self.fps_tracker.fps, 2),
            "dropped_frames": self.dropped_frames,
            "uptime_s": round(uptime, 2),
            "cpu_percent": self._last_cpu_percent,
            "ram": self._last_ram,
            "cpu_temp_c": self._last_temp_c,
            "latency": latency_report,
        }

    def report_json(self, indent: int = 2) -> str:
        """Return :meth:`report` serialised as a JSON string."""
        return json.dumps(self.report(), indent=indent, default=str)

    def reset(self) -> None:
        """Reset all counters and trackers (does not stop the background thread)."""
        self.fps_tracker.reset()
        with self._dropped_lock:
            self._dropped_frames = 0
        with self._latency_lock:
            for tracker in self._latency_trackers.values():
                tracker.reset()
        self._start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _background_loop(self) -> None:
        """Periodically sample system resources until :meth:`stop` is called."""
        while self._running:
            try:
                self.sample_system()
            except Exception as exc:
                logger.debug("Background sample error: %s", exc)
            time.sleep(self._sample_interval)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PerformanceMonitor":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
