"""
AeroSentinel Flight Status Monitor
Wraps MAVLinkClient to compute link quality and detect anomalies.
"""

import logging
import time
from typing import Callable, Dict, List, Optional

from aerosentinel.flight.telemetry import TelemetryData

logger = logging.getLogger(__name__)

LOW_BATTERY_THRESHOLD = 20.0   # percent
LINK_LOSS_TIMEOUT_S = 10.0     # seconds


class FlightStatusMonitor:
    """
    Monitors flight status using MAVLinkClient.
    Detects link loss, low battery, and mode changes.
    Fires callbacks on anomalies.
    """

    def __init__(self, mavlink_client=None):
        """
        :param mavlink_client: MAVLinkClient instance (or MockMAVLinkClient).
                               If None, monitor runs without a live connection.
        """
        self._client = mavlink_client
        self._last_flight_mode: Optional[str] = None

        # Callbacks
        self.on_link_loss: List[Callable] = []
        self.on_low_battery: List[Callable] = []
        self.on_mode_change: List[Callable] = []

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def compute_link_quality(self, heartbeat_timestamps: List[float]) -> float:
        """
        Estimate link quality (0–100 %) from recent heartbeat timestamps.
        Expects ~1 Hz heartbeats. Uses the last 10 seconds as reference window.
        """
        if not heartbeat_timestamps:
            return 0.0
        now = time.time()
        window = 10.0  # seconds
        recent = [t for t in heartbeat_timestamps if now - t <= window]
        # Ideal: 10 heartbeats in 10 s
        expected = window  # 1 Hz * 10 s = 10
        quality = min(100.0, (len(recent) / expected) * 100.0)
        return round(quality, 1)

    def detect_low_battery(self, telemetry: TelemetryData) -> bool:
        """Return True if battery percentage is below 20 %."""
        if telemetry.battery_pct is None:
            return False
        return telemetry.battery_pct < LOW_BATTERY_THRESHOLD

    def detect_link_loss(self, telemetry: TelemetryData) -> bool:
        """Return True if no heartbeat received for more than 10 s."""
        if telemetry.last_heartbeat is None:
            return not telemetry.is_connected
        return (time.time() - telemetry.last_heartbeat) > LINK_LOSS_TIMEOUT_S

    # ------------------------------------------------------------------
    # Status summary
    # ------------------------------------------------------------------

    def get_status_summary(self) -> dict:
        """Return a dict summarising current flight status."""
        if self._client is None:
            return {"error": "No MAVLink client attached"}

        raw = self._client.get_telemetry()
        heartbeats = []
        if hasattr(self._client, "get_heartbeat_timestamps"):
            heartbeats = self._client.get_heartbeat_timestamps()

        tel = TelemetryData(
            timestamp=raw.get("timestamp") or time.time(),
            battery_pct=raw.get("battery_pct"),
            battery_voltage_v=raw.get("battery_voltage_v"),
            battery_current_a=raw.get("battery_current_a"),
            lat=raw.get("lat"),
            lon=raw.get("lon"),
            alt_m=raw.get("alt_m"),
            gps_fix_type=raw.get("gps_fix_type"),
            gps_satellites=raw.get("gps_satellites"),
            roll_deg=raw.get("roll_deg"),
            pitch_deg=raw.get("pitch_deg"),
            yaw_deg=raw.get("yaw_deg"),
            groundspeed_ms=raw.get("groundspeed_ms"),
            flight_mode=raw.get("flight_mode"),
            armed=raw.get("armed"),
            link_quality=raw.get("link_quality"),
            is_connected=raw.get("is_connected", False),
            last_heartbeat=raw.get("last_heartbeat"),
        )

        link_quality = self.compute_link_quality(heartbeats)
        low_battery = self.detect_low_battery(tel)
        link_loss = self.detect_link_loss(tel)

        # Fire callbacks
        if link_loss:
            self._emit(self.on_link_loss, tel)
        if low_battery:
            self._emit(self.on_low_battery, tel)

        current_mode = tel.flight_mode
        if current_mode != self._last_flight_mode:
            if self._last_flight_mode is not None:
                self._emit(self.on_mode_change, self._last_flight_mode, current_mode)
            self._last_flight_mode = current_mode

        return {
            "is_connected": tel.is_connected,
            "link_quality": link_quality,
            "link_loss": link_loss,
            "battery_pct": tel.battery_pct,
            "low_battery": low_battery,
            "flight_mode": tel.flight_mode,
            "armed": tel.armed,
            "lat": tel.lat,
            "lon": tel.lon,
            "alt_m": tel.alt_m,
            "roll_deg": tel.roll_deg,
            "pitch_deg": tel.pitch_deg,
            "yaw_deg": tel.yaw_deg,
        }

    @staticmethod
    def _emit(callbacks: List[Callable], *args, **kwargs) -> None:
        for cb in callbacks:
            try:
                cb(*args, **kwargs)
            except Exception as exc:
                logger.warning("FlightStatusMonitor callback error: %s", exc)
