"""
AeroSentinel Mock MAVLink Client
Provides synthetic telemetry for testing without real hardware.
"""

import math
import random
import threading
import time
from typing import Optional


class MockMAVLinkClient:
    """
    Mock MAVLink client with the same interface as MAVLinkClient.
    Returns synthetic telemetry: battery draining, sinusoidal attitude,
    random GPS walk. Supports configurable failure modes.
    """

    def __init__(
        self,
        connection_string: str = "mock://",
        baud_rate: int = 57600,
        read_only: bool = True,
        # Failure modes
        link_loss_after_n_messages: Optional[int] = None,
        low_battery_at_n: Optional[int] = None,
    ):
        self._connection_string = connection_string
        self._baud_rate = baud_rate
        self._read_only = read_only

        self._link_loss_after = link_loss_after_n_messages
        self._low_battery_at = low_battery_at_n

        self._message_count = 0
        self._start_time: Optional[float] = None
        self._running = False
        self._lock = threading.Lock()
        self._heartbeat_timestamps: list = []

        # Synthetic state
        self._battery = 100.0
        self._gps_lat = 28.6139   # Delhi (synthetic — not real mission coords)
        self._gps_lon = 77.2090
        self._alt = 50.0

        self.latest_telemetry: dict = self._empty_telemetry()

    # ------------------------------------------------------------------
    # Connection interface (matches MAVLinkClient)
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._start_time = time.time()
        self._running = True
        threading.Thread(
            target=self._update_loop, daemon=True, name="MockMAVLinkUpdate"
        ).start()

    def disconnect(self) -> None:
        self._running = False
        with self._lock:
            self.latest_telemetry["is_connected"] = False

    def reconnect(self) -> None:
        self.disconnect()
        time.sleep(0.05)
        self.connect()

    def get_telemetry(self) -> dict:
        with self._lock:
            return dict(self.latest_telemetry)

    def get_heartbeat_timestamps(self) -> list:
        with self._lock:
            return list(self._heartbeat_timestamps)

    # ------------------------------------------------------------------
    # Synthetic update loop
    # ------------------------------------------------------------------

    def _update_loop(self) -> None:
        while self._running:
            self._tick()
            time.sleep(0.1)  # ~10 Hz updates

    def _tick(self) -> None:
        with self._lock:
            self._message_count += 1
            n = self._message_count
            t = time.time()
            elapsed = t - self._start_time

            # Trigger link loss
            if self._link_loss_after is not None and n > self._link_loss_after:
                self.latest_telemetry["is_connected"] = False
                self.latest_telemetry["last_heartbeat"] = t - 15.0  # expired
                return

            # Trigger low battery
            if self._low_battery_at is not None and n >= self._low_battery_at:
                self._battery = 10.0
            else:
                # Drain battery slowly (1% per 60 messages)
                self._battery = max(0.0, 100.0 - elapsed / 60.0)

            # Heartbeat every ~10 ticks (1 s at 10 Hz)
            if n % 10 == 0:
                self._heartbeat_timestamps.append(t)
                self._heartbeat_timestamps = self._heartbeat_timestamps[-20:]

            # Sinusoidal attitude
            roll_deg = 10.0 * math.sin(elapsed * 0.5)
            pitch_deg = 5.0 * math.sin(elapsed * 0.3 + 1.0)
            yaw_deg = (elapsed * 15.0) % 360.0

            # Random GPS walk (tiny step each tick)
            self._gps_lat += random.uniform(-0.00001, 0.00001)
            self._gps_lon += random.uniform(-0.00001, 0.00001)

            self.latest_telemetry.update(
                {
                    "timestamp": t,
                    "battery_pct": round(self._battery, 1),
                    "battery_voltage_v": round(11.1 + (self._battery / 100) * 1.5, 2),
                    "battery_current_a": round(5.0 + random.uniform(-0.5, 0.5), 2),
                    "lat": round(self._gps_lat, 7),
                    "lon": round(self._gps_lon, 7),
                    "alt_m": self._alt + math.sin(elapsed * 0.1) * 2.0,
                    "gps_fix_type": 3,
                    "gps_satellites": random.randint(8, 12),
                    "roll_deg": round(roll_deg, 2),
                    "pitch_deg": round(pitch_deg, 2),
                    "yaw_deg": round(yaw_deg, 2),
                    "groundspeed_ms": round(5.0 + math.sin(elapsed * 0.2), 2),
                    "flight_mode": "GUIDED",
                    "armed": True,
                    "link_quality": 95.0,
                    "is_connected": True,
                    "last_heartbeat": t if n % 10 == 0 else self.latest_telemetry.get("last_heartbeat"),
                }
            )

    # ------------------------------------------------------------------
    # SAFETY: Write / command interface — ALL DISABLED (same as real client)
    # ------------------------------------------------------------------

    def send_command(self, *args, **kwargs):
        raise NotImplementedError("SAFETY: MockMAVLinkClient is read-only.")

    def arm(self, *args, **kwargs):
        raise NotImplementedError("SAFETY: arm() is not permitted.")

    def disarm(self, *args, **kwargs):
        raise NotImplementedError("SAFETY: disarm() is not permitted.")

    def takeoff(self, *args, **kwargs):
        raise NotImplementedError("SAFETY: takeoff() is not permitted.")

    def land(self, *args, **kwargs):
        raise NotImplementedError("SAFETY: land() is not permitted.")

    def set_mode(self, *args, **kwargs):
        raise NotImplementedError("SAFETY: set_mode() is not permitted.")

    def motor_test(self, *args, **kwargs):
        raise NotImplementedError("SAFETY: motor_test() is not permitted.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_telemetry() -> dict:
        return {
            "timestamp": None,
            "battery_pct": None,
            "battery_voltage_v": None,
            "battery_current_a": None,
            "lat": None,
            "lon": None,
            "alt_m": None,
            "gps_fix_type": None,
            "gps_satellites": None,
            "roll_deg": None,
            "pitch_deg": None,
            "yaw_deg": None,
            "groundspeed_ms": None,
            "flight_mode": None,
            "armed": None,
            "link_quality": None,
            "is_connected": False,
            "last_heartbeat": None,
        }
