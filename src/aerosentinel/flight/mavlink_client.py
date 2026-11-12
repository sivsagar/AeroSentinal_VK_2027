"""
AeroSentinel MAVLink Client — READ ONLY
SAFETY: This module is read-only. No flight commands are issued.

Receives and parses MAVLink telemetry from an autopilot over serial/UDP/TCP.
All write / command methods raise NotImplementedError to prevent accidental
flight commands.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pymavlink import
# ---------------------------------------------------------------------------
try:
    from pymavlink import mavutil as _mavutil  # type: ignore
    _MAVLINK_AVAILABLE = True
except ImportError:
    _mavutil = None
    _MAVLINK_AVAILABLE = False


HEARTBEAT_TIMEOUT_S = 10.0  # seconds before link-loss declared


class MAVLinkClient:
    """
    Read-only MAVLink telemetry client.

    SAFETY: This module is read-only. No flight commands are issued.
    All command-related methods raise NotImplementedError.
    """

    def __init__(
        self,
        connection_string: str = "udp:0.0.0.0:14550",
        baud_rate: int = 57600,
        read_only: bool = True,
    ):
        if not _MAVLINK_AVAILABLE:
            raise ImportError(
                "pymavlink is not installed. Install it with: pip install pymavlink"
            )

        self._connection_string = connection_string
        self._baud_rate = baud_rate
        self._read_only = read_only  # always True; exposed for introspection

        self._conn = None
        self._lock = threading.Lock()
        self._running = False
        self._receive_thread: Optional[threading.Thread] = None

        # Thread-safe latest telemetry snapshot
        self.latest_telemetry: dict = {
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
        self._heartbeat_timestamps: list = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open MAVLink connection and start receive thread."""
        logger.info("Connecting to MAVLink: %s", self._connection_string)
        self._conn = _mavutil.mavlink_connection(
            self._connection_string, baud=self._baud_rate
        )
        self._running = True
        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="MAVLinkReceive"
        )
        self._receive_thread.start()
        logger.info("MAVLink connected.")

    def disconnect(self) -> None:
        """Stop receive thread and close connection."""
        self._running = False
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        with self._lock:
            self.latest_telemetry["is_connected"] = False
        logger.info("MAVLink disconnected.")

    def reconnect(self) -> None:
        """Attempt to reconnect after link loss."""
        logger.warning("MAVLink: attempting reconnect …")
        self.disconnect()
        time.sleep(1.0)
        self.connect()

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    def _receive_loop(self) -> None:
        """Background thread: continuously receive and parse MAVLink messages."""
        while self._running:
            try:
                msg = self._conn.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    self._check_heartbeat_timeout()
                    continue
                self._parse_message(msg)
            except Exception as exc:
                logger.error("MAVLink receive error: %s", exc)
                time.sleep(0.5)

    def _parse_message(self, msg) -> None:
        msg_type = msg.get_type()
        now = time.time()

        with self._lock:
            self.latest_telemetry["timestamp"] = now

            if msg_type == "HEARTBEAT":
                self.latest_telemetry["last_heartbeat"] = now
                self.latest_telemetry["is_connected"] = True
                self._heartbeat_timestamps.append(now)
                # Keep last 20 timestamps for link quality calc
                self._heartbeat_timestamps = self._heartbeat_timestamps[-20:]
                # Flight mode (base_mode flag 0x80 = custom mode enabled)
                armed = bool(msg.base_mode & 0x80)  # MAV_MODE_FLAG_SAFETY_ARMED is 0x80? 
                # Actually MAV_MODE_FLAG_SAFETY_ARMED = 128
                armed = bool(msg.base_mode & mavutil_const("MAV_MODE_FLAG_SAFETY_ARMED", 128))
                self.latest_telemetry["armed"] = armed
                try:
                    self.latest_telemetry["flight_mode"] = self._conn.flightmode
                except Exception:
                    pass

            elif msg_type == "BATTERY_STATUS":
                battery_remaining = msg.battery_remaining  # 0-100 or -1
                if battery_remaining != -1:
                    self.latest_telemetry["battery_pct"] = float(battery_remaining)
                voltages = msg.voltages  # list of millivolts
                if voltages:
                    # First cell group
                    v_mv = voltages[0]
                    if v_mv != 65535:
                        self.latest_telemetry["battery_voltage_v"] = v_mv / 1000.0
                current = msg.current_battery  # cA
                if current != -1:
                    self.latest_telemetry["battery_current_a"] = current / 100.0

            elif msg_type == "GPS_RAW_INT":
                fix_type = msg.fix_type
                self.latest_telemetry["gps_fix_type"] = fix_type
                self.latest_telemetry["gps_satellites"] = msg.satellites_visible
                if fix_type >= 3:
                    # Genuine fix — convert from 1e7 degrees
                    self.latest_telemetry["lat"] = msg.lat / 1e7
                    self.latest_telemetry["lon"] = msg.lon / 1e7
                    self.latest_telemetry["alt_m"] = msg.alt / 1000.0
                else:
                    # No fix — do not fabricate GPS coordinates
                    self.latest_telemetry["lat"] = None
                    self.latest_telemetry["lon"] = None

            elif msg_type == "ATTITUDE":
                import math
                self.latest_telemetry["roll_deg"] = math.degrees(msg.roll)
                self.latest_telemetry["pitch_deg"] = math.degrees(msg.pitch)
                self.latest_telemetry["yaw_deg"] = math.degrees(msg.yaw)

            elif msg_type == "VFR_HUD":
                self.latest_telemetry["groundspeed_ms"] = msg.groundspeed
                self.latest_telemetry["alt_m"] = msg.alt

            elif msg_type == "SYS_STATUS":
                # onboard_control_sensors_health bitmask — stored for future use
                self.latest_telemetry["sensors_health"] = msg.onboard_control_sensors_health

    def _check_heartbeat_timeout(self) -> None:
        with self._lock:
            last = self.latest_telemetry.get("last_heartbeat")
        if last is not None and (time.time() - last) > HEARTBEAT_TIMEOUT_S:
            logger.warning("MAVLink: heartbeat timeout — link loss detected")
            with self._lock:
                self.latest_telemetry["is_connected"] = False

    def get_telemetry(self) -> dict:
        """Return a snapshot of the latest telemetry."""
        with self._lock:
            return dict(self.latest_telemetry)

    def get_heartbeat_timestamps(self) -> list:
        with self._lock:
            return list(self._heartbeat_timestamps)

    # ------------------------------------------------------------------
    # SAFETY: Write / command interface — ALL DISABLED
    # These placeholders ensure no accidental commands are sent.
    # ------------------------------------------------------------------

    def send_command(self, *args, **kwargs):
        """SAFETY: Command sending is disabled in read-only mode."""
        raise NotImplementedError(
            "SAFETY: MAVLinkClient is read-only. Command sending is not permitted."
        )

    def arm(self, *args, **kwargs):
        """SAFETY: Arming is disabled. This module is read-only."""
        raise NotImplementedError(
            "SAFETY: arm() is not permitted. This module is read-only."
        )

    def disarm(self, *args, **kwargs):
        """SAFETY: Disarming is disabled. This module is read-only."""
        raise NotImplementedError(
            "SAFETY: disarm() is not permitted. This module is read-only."
        )

    def takeoff(self, *args, **kwargs):
        """SAFETY: Takeoff is disabled. This module is read-only."""
        raise NotImplementedError(
            "SAFETY: takeoff() is not permitted. This module is read-only."
        )

    def land(self, *args, **kwargs):
        """SAFETY: Land command is disabled. This module is read-only."""
        raise NotImplementedError(
            "SAFETY: land() is not permitted. This module is read-only."
        )

    def set_mode(self, *args, **kwargs):
        """SAFETY: Mode changes are disabled. This module is read-only."""
        raise NotImplementedError(
            "SAFETY: set_mode() is not permitted. This module is read-only."
        )

    def motor_test(self, *args, **kwargs):
        """SAFETY: Motor test is disabled. This module is read-only."""
        raise NotImplementedError(
            "SAFETY: motor_test() is not permitted. This module is read-only."
        )


def mavutil_const(name: str, default: int) -> int:
    """Safely retrieve a MAVLink constant, falling back to default."""
    try:
        from pymavlink import mavutil
        return getattr(mavutil.mavlink, name, default)
    except Exception:
        return default
