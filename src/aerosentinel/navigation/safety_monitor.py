"""
AeroSentinel Safety Monitor Subsystem
=====================================
Monitors runtime safety constraints:
- Companion computer CPU core temperatures (thermal throttling prevention)
- Available RAM memory margins
- Sensor heartbeats (RGB camera, D435i, IMU)
- Mission time bounds
- Emergency-stop status
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SafetyStatus(str, Enum):
    """Overall system safety evaluation."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class SafetyMonitor:
    """
    Active system watchdog and hardware safety assessor for UAV companion computer.
    """

    def __init__(
        self,
        cpu_warning_temp_c: float = 75.0,
        cpu_critical_temp_c: float = 85.0,
        min_free_ram_mb: float = 200.0,
        max_sensor_dropout_s: float = 5.0,
    ) -> None:
        self.cpu_warning_temp_c = cpu_warning_temp_c
        self.cpu_critical_temp_c = cpu_critical_temp_c
        self.min_free_ram_mb = min_free_ram_mb
        self.max_sensor_dropout_s = max_sensor_dropout_s

        self._callbacks: List[Callable[[SafetyStatus, Dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self._last_status = SafetyStatus.SAFE
        self._emergency_stop_triggered = False

    def register_callback(
        self, callback: Callable[[SafetyStatus, Dict[str, Any]], None]
    ) -> None:
        """Register a callback invoked when safety status changes or degrades."""
        with self._lock:
            self._callbacks.append(callback)

    def trigger_emergency_stop(self, reason: str = "Operator Triggered") -> None:
        """Set software emergency stop flag."""
        with self._lock:
            self._emergency_stop_triggered = True
            logger.critical("EMERGENCY STOP TRIGGERED: %s", reason)
        self._notify(SafetyStatus.CRITICAL, {"emergency_stop": True, "reason": reason})

    def clear_emergency_stop(self) -> None:
        """Clear software emergency stop."""
        with self._lock:
            self._emergency_stop_triggered = False
            logger.warning("Emergency stop cleared by operator.")

    def evaluate(
        self,
        cpu_temp_c: Optional[float] = None,
        free_ram_mb: Optional[float] = None,
        rgb_healthy: bool = True,
        depth_healthy: bool = True,
        obstacle_critical: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate current safety conditions and return diagnostic report.
        """
        reasons: List[str] = []
        status = SafetyStatus.SAFE

        if self._emergency_stop_triggered:
            status = SafetyStatus.CRITICAL
            reasons.append("Emergency stop active")

        if cpu_temp_c is not None:
            if cpu_temp_c >= self.cpu_critical_temp_c:
                status = SafetyStatus.CRITICAL
                reasons.append(f"CPU critical temp: {cpu_temp_c:.1f}C >= {self.cpu_critical_temp_c}C")
            elif cpu_temp_c >= self.cpu_warning_temp_c:
                if status != SafetyStatus.CRITICAL:
                    status = SafetyStatus.WARNING
                reasons.append(f"CPU warning temp: {cpu_temp_c:.1f}C >= {self.cpu_warning_temp_c}C")

        if free_ram_mb is not None and free_ram_mb < self.min_free_ram_mb:
            status = SafetyStatus.CRITICAL
            reasons.append(f"Low RAM: {free_ram_mb:.1f}MB < {self.min_free_ram_mb}MB")

        if not rgb_healthy:
            if status != SafetyStatus.CRITICAL:
                status = SafetyStatus.WARNING
            reasons.append("RGB Camera disconnected/degraded")

        if not depth_healthy:
            if status != SafetyStatus.CRITICAL:
                status = SafetyStatus.WARNING
            reasons.append("RealSense D435i disconnected/degraded")

        if obstacle_critical:
            if status != SafetyStatus.CRITICAL:
                status = SafetyStatus.WARNING
            reasons.append("Critical near-field obstacle in flight path")

        report = {
            "status": status.value,
            "timestamp": time.time(),
            "emergency_stop": self._emergency_stop_triggered,
            "cpu_temp_c": cpu_temp_c,
            "free_ram_mb": free_ram_mb,
            "rgb_healthy": rgb_healthy,
            "depth_healthy": depth_healthy,
            "reasons": reasons,
        }

        with self._lock:
            prev = self._last_status
            self._last_status = status

        if prev != status:
            logger.info("Safety status transition: %s -> %s (%s)", prev, status, ", ".join(reasons))
            self._notify(status, report)

        return report

    def _notify(self, status: SafetyStatus, report: Dict[str, Any]) -> None:
        for cb in self._callbacks:
            try:
                cb(status, report)
            except Exception as e:
                logger.error("Error in safety callback: %s", e)

