"""
sensor_health.py — Sensor health monitoring for AeroSentinel.

Monitors RGB camera, depth camera, and IMU independently on a background
thread.  State changes are reported via callbacks.

Usage
-----
::

    monitor = SensorHealthMonitor(camera_manager, imu_processor)
    monitor.add_callback(lambda event: print(event))
    monitor.start()
    ...
    monitor.stop()
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CHECK_INTERVAL = 2.0   # seconds


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class SensorStatus(enum.Enum):
    """Operational state of an individual sensor."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthEvent:
    """Emitted when a sensor's status changes."""

    sensor: str            # "rgb", "depth", or "imu"
    previous: SensorStatus
    current: SensorStatus
    timestamp: float       # time.monotonic()
    message: str = ""


# ---------------------------------------------------------------------------
# SensorHealthMonitor
# ---------------------------------------------------------------------------

class SensorHealthMonitor:
    """
    Polls camera and IMU health at a fixed interval.

    Parameters
    ----------
    camera_manager : CameraManager, optional
        If provided, its ``status()`` method is polled.
    imu_processor : IMUProcessor, optional
        If provided, its ``is_healthy`` property is polled.
    check_interval_s : float
        Seconds between health checks.
    """

    def __init__(
        self,
        camera_manager=None,
        imu_processor=None,
        check_interval_s: float = _DEFAULT_CHECK_INTERVAL,
    ) -> None:
        self._cam = camera_manager
        self._imu = imu_processor
        self._interval = check_interval_s

        self._status: Dict[str, SensorStatus] = {
            "rgb": SensorStatus.UNKNOWN,
            "depth": SensorStatus.UNKNOWN,
            "imu": SensorStatus.UNKNOWN,
        }
        self._callbacks: List[Callable[[HealthEvent], None]] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def add_callback(self, cb: Callable[[HealthEvent], None]) -> None:
        """Register a callback invoked on every status change."""
        with self._lock:
            self._callbacks.append(cb)

    def remove_callback(self, cb: Callable[[HealthEvent], None]) -> None:
        """Unregister a previously added callback."""
        with self._lock:
            try:
                self._callbacks.remove(cb)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="sensor-health",
            daemon=True,
        )
        self._running = True
        self._thread.start()
        logger.info("SensorHealthMonitor started (interval=%.1fs).", self._interval)

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("SensorHealthMonitor stopped.")

    # ------------------------------------------------------------------
    # Status access
    # ------------------------------------------------------------------

    def health_report(self) -> Dict[str, str]:
        """Return snapshot of current sensor statuses (name → status value)."""
        with self._lock:
            return {k: v.value for k, v in self._status.items()}

    def get_status(self, sensor: str) -> SensorStatus:
        """Return current SensorStatus for a named sensor."""
        with self._lock:
            return self._status.get(sensor, SensorStatus.UNKNOWN)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Poll all sensors and emit events on state changes."""
        while not self._stop_event.is_set():
            self._check_cameras()
            self._check_imu()
            self._stop_event.wait(timeout=self._interval)

    def _check_cameras(self) -> None:
        """Poll CameraManager.status() and update internal state."""
        if self._cam is None:
            return
        try:
            cam_status = self._cam.status()
        except Exception as exc:  # noqa: BLE001
            logger.debug("SensorHealthMonitor: camera status error: %s", exc)
            return

        for sensor_key in ("rgb", "depth"):
            info = cam_status.get(sensor_key, {})
            if not info.get("enabled", False):
                new_status = SensorStatus.UNKNOWN
            elif info.get("connected", False):
                new_status = SensorStatus.CONNECTED
            else:
                new_status = SensorStatus.DISCONNECTED
            self._update(sensor_key, new_status)

    def _check_imu(self) -> None:
        """Poll IMUProcessor.is_healthy and update internal state."""
        if self._imu is None:
            return
        try:
            healthy = self._imu.is_healthy
        except Exception as exc:  # noqa: BLE001
            logger.debug("SensorHealthMonitor: IMU health error: %s", exc)
            return

        if healthy:
            new_status = SensorStatus.CONNECTED
        else:
            # Distinguish between "never received data" and dropout
            age = getattr(self._imu, "last_update_age_s", None)
            if age is None:
                new_status = SensorStatus.UNKNOWN
            else:
                new_status = SensorStatus.DISCONNECTED
        self._update("imu", new_status)

    def _update(self, sensor: str, new_status: SensorStatus) -> None:
        """Thread-safely update status and fire callbacks on change."""
        with self._lock:
            prev = self._status.get(sensor, SensorStatus.UNKNOWN)
            if prev == new_status:
                return
            self._status[sensor] = new_status
            callbacks = list(self._callbacks)

        event = HealthEvent(
            sensor=sensor,
            previous=prev,
            current=new_status,
            timestamp=time.monotonic(),
            message=f"{sensor} changed {prev.value} → {new_status.value}",
        )
        logger.info("SensorHealth: %s", event.message)
        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SensorHealthMonitor callback error: %s", exc)
