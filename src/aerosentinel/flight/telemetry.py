"""
AeroSentinel Telemetry Data Model
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TelemetryData:
    timestamp: float
    battery_pct: Optional[float] = None
    battery_voltage_v: Optional[float] = None
    battery_current_a: Optional[float] = None
    lat: Optional[float] = None          # None when no GPS
    lon: Optional[float] = None          # None when no GPS
    alt_m: Optional[float] = None
    gps_fix_type: Optional[int] = None   # 0=no fix, 3=3D fix
    gps_satellites: Optional[int] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    groundspeed_ms: Optional[float] = None
    flight_mode: Optional[str] = None
    armed: Optional[bool] = None
    link_quality: Optional[float] = None
    is_connected: bool = False
    last_heartbeat: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert telemetry to a plain dictionary."""
        return {
            "timestamp": self.timestamp,
            "battery_pct": self.battery_pct,
            "battery_voltage_v": self.battery_voltage_v,
            "battery_current_a": self.battery_current_a,
            "lat": self.lat,
            "lon": self.lon,
            "alt_m": self.alt_m,
            "gps_fix_type": self.gps_fix_type,
            "gps_satellites": self.gps_satellites,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "groundspeed_ms": self.groundspeed_ms,
            "flight_mode": self.flight_mode,
            "armed": self.armed,
            "link_quality": self.link_quality,
            "is_connected": self.is_connected,
            "last_heartbeat": self.last_heartbeat,
        }
