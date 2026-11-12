"""
AeroSentinel Localization Interface
=====================================
Provides the abstraction layer for all localization sources (GPS via MAVLink,
visual odometry, SLAM, etc.).

Design goals:
- Single interface that the rest of the system depends on.
- Graceful fallback when no reliable localization is available (NullLocalizationSource).
- Thread-safe access to the latest LocalizationData.
- Easy swap of underlying source without changing consumers.

Usage::

    from aerosentinel.localization.localization_interface import LocalizationManager

    mgr = LocalizationManager(config={"source": "mavlink_gps"})
    mgr.set_telemetry(telemetry_data)
    loc = mgr.get_latest()
    if loc.position_valid:
        print(f"Position: {loc.lat:.6f}, {loc.lon:.6f}, alt={loc.alt_m:.1f}m")
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LocalizationData:
    """
    Unified localization snapshot produced by any LocalizationSource.

    All position fields are Optional; consumers must check ``position_valid``
    before trusting numeric values.

    Coordinate frames
    -----------------
    * Local frame (x_m, y_m, z_m): metres from an arbitrary origin, NED or ENU
      depending on the source; documented per-source.
    * WGS84 (lat, lon, alt_m): decimal degrees / metres MSL; None when GPS is
      unavailable.

    Attributes
    ----------
    timestamp : float
        UNIX epoch seconds (time.time()) when the measurement was taken.
    x_m, y_m, z_m : Optional[float]
        Position in local frame (metres).
    lat, lon : Optional[float]
        WGS84 latitude / longitude (decimal degrees). None when GPS unavailable.
    alt_m : Optional[float]
        Altitude in metres above mean sea level (WGS84). None when unavailable.
    roll_deg, pitch_deg, yaw_deg : Optional[float]
        Euler angles in degrees (NED convention). None when unavailable.
    vx_ms, vy_ms, vz_ms : Optional[float]
        Velocity in local frame (m/s). None when unavailable.
    confidence : float
        Source-estimated confidence in [0.0, 1.0].  0.0 = unknown / invalid.
    source : str
        One of ``'none'``, ``'mavlink_gps'``, ``'visual_odometry'``, ``'slam'``.
    gps_available : bool
        True when a valid GPS fix (fix_type >= 3) is present.
    position_valid : bool
        True when the position fields can be trusted by consumers.
    """

    timestamp: float

    # Local frame
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    z_m: Optional[float] = None

    # WGS84
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_m: Optional[float] = None

    # Orientation (degrees)
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None

    # Velocity (m/s, local frame)
    vx_ms: Optional[float] = None
    vy_ms: Optional[float] = None
    vz_ms: Optional[float] = None

    # Meta
    confidence: float = 0.0          # 0.0 to 1.0
    source: str = "none"             # 'none', 'mavlink_gps', 'visual_odometry', 'slam'
    gps_available: bool = False
    position_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of this snapshot."""
        return {
            "timestamp": self.timestamp,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "lat": self.lat,
            "lon": self.lon,
            "alt_m": self.alt_m,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "vx_ms": self.vx_ms,
            "vy_ms": self.vy_ms,
            "vz_ms": self.vz_ms,
            "confidence": self.confidence,
            "source": self.source,
            "gps_available": self.gps_available,
            "position_valid": self.position_valid,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LocalizationSource(ABC):
    """
    Abstract base class for all localization providers.

    Implementors must override all abstract methods. Each call should return
    the most recent estimate; implementations are responsible for caching if the
    underlying sensor is asynchronous.
    """

    @abstractmethod
    def get_position(self) -> Dict[str, Optional[float]]:
        """Return position estimate: x_m, y_m, z_m, lat, lon, alt_m."""

    @abstractmethod
    def get_orientation(self) -> Dict[str, Optional[float]]:
        """Return orientation: roll_deg, pitch_deg, yaw_deg."""

    @abstractmethod
    def get_velocity(self) -> Dict[str, Optional[float]]:
        """Return velocity in local frame: vx_ms, vy_ms, vz_ms."""

    @abstractmethod
    def get_confidence(self) -> float:
        """Return confidence score in [0.0, 1.0]."""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return health status dict: source, gps_available, position_valid, last_update."""

    def get_snapshot(self) -> LocalizationData:
        """Combine all abstract methods into a LocalizationData snapshot."""
        pos = self.get_position()
        ori = self.get_orientation()
        vel = self.get_velocity()
        status = self.get_status()
        return LocalizationData(
            timestamp=time.time(),
            x_m=pos.get("x_m"),
            y_m=pos.get("y_m"),
            z_m=pos.get("z_m"),
            lat=pos.get("lat"),
            lon=pos.get("lon"),
            alt_m=pos.get("alt_m"),
            roll_deg=ori.get("roll_deg"),
            pitch_deg=ori.get("pitch_deg"),
            yaw_deg=ori.get("yaw_deg"),
            vx_ms=vel.get("vx_ms"),
            vy_ms=vel.get("vy_ms"),
            vz_ms=vel.get("vz_ms"),
            confidence=self.get_confidence(),
            source=status.get("source", "none"),
            gps_available=status.get("gps_available", False),
            position_valid=status.get("position_valid", False),
        )


# ---------------------------------------------------------------------------
# Null source (safe default)
# ---------------------------------------------------------------------------

class NullLocalizationSource(LocalizationSource):
    """
    Safe no-op localization source used when no real source is configured.

    All position fields return None; confidence is 0.0; position_valid is False.
    This is the default fallback so that the rest of the system can run safely
    without crashing when localisation is unavailable.
    """

    def get_position(self) -> Dict[str, Optional[float]]:
        return {"x_m": None, "y_m": None, "z_m": None,
                "lat": None, "lon": None, "alt_m": None}

    def get_orientation(self) -> Dict[str, Optional[float]]:
        return {"roll_deg": None, "pitch_deg": None, "yaw_deg": None}

    def get_velocity(self) -> Dict[str, Optional[float]]:
        return {"vx_ms": None, "vy_ms": None, "vz_ms": None}

    def get_confidence(self) -> float:
        return 0.0

    def get_status(self) -> Dict[str, Any]:
        return {
            "source": "none",
            "gps_available": False,
            "position_valid": False,
            "last_update": None,
        }


# ---------------------------------------------------------------------------
# MAVLink GPS source
# ---------------------------------------------------------------------------

class MAVLinkLocalizationSource(LocalizationSource):
    """
    Localization source backed by MAVLink telemetry (GPS + attitude).

    Wraps a TelemetryData object and exposes position, orientation and
    confidence. Call ``update(telemetry)`` whenever fresh telemetry arrives.

    GPS fix quality thresholds:
    - fix_type >= 3 (3D fix) → position_valid = True, confidence ~ 0.7
    - fix_type >= 3 and satellites >= 8 → confidence ~ 0.9
    - No fix → position_valid = False, confidence = 0.0
    """

    _MIN_FIX_TYPE: int = 3
    _CONF_LOW_SATS: int = 5
    _CONF_HIGH_SATS: int = 8

    def __init__(self) -> None:
        self._telemetry = None
        self._last_update: Optional[float] = None
        self._lock = threading.Lock()

    def update(self, telemetry: Any) -> None:
        """Push a fresh TelemetryData snapshot."""
        with self._lock:
            self._telemetry = telemetry
            self._last_update = time.time()

    def _has_fix(self) -> bool:
        t = self._telemetry
        if t is None:
            return False
        fix = getattr(t, "gps_fix_type", None)
        return fix is not None and fix >= self._MIN_FIX_TYPE

    def get_position(self) -> Dict[str, Optional[float]]:
        with self._lock:
            t = self._telemetry
        if t is None or not self._has_fix():
            return {"x_m": None, "y_m": None, "z_m": None,
                    "lat": None, "lon": None, "alt_m": None}
        return {
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "lat": getattr(t, "lat", None),
            "lon": getattr(t, "lon", None),
            "alt_m": getattr(t, "alt_m", None),
        }

    def get_orientation(self) -> Dict[str, Optional[float]]:
        with self._lock:
            t = self._telemetry
        if t is None:
            return {"roll_deg": None, "pitch_deg": None, "yaw_deg": None}
        return {
            "roll_deg": getattr(t, "roll_deg", None),
            "pitch_deg": getattr(t, "pitch_deg", None),
            "yaw_deg": getattr(t, "yaw_deg", None),
        }

    def get_velocity(self) -> Dict[str, Optional[float]]:
        return {"vx_ms": None, "vy_ms": None, "vz_ms": None}

    def get_confidence(self) -> float:
        with self._lock:
            t = self._telemetry
        if t is None or not self._has_fix():
            return 0.0
        sats = getattr(t, "gps_satellites", None) or 0
        link = getattr(t, "link_quality", None)
        link_factor = (link / 100.0) if link is not None else 0.8
        if sats >= self._CONF_HIGH_SATS:
            base = 0.9
        elif sats >= self._CONF_LOW_SATS:
            base = 0.7
        else:
            base = 0.5
        return round(base * link_factor, 3)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            t = self._telemetry
            lu = self._last_update
        has_fix = self._has_fix()
        return {
            "source": "mavlink_gps",
            "gps_available": has_fix,
            "position_valid": has_fix,
            "last_update": lu,
            "connected": getattr(t, "is_connected", False) if t is not None else False,
            "gps_fix_type": getattr(t, "gps_fix_type", None) if t else None,
            "gps_satellites": getattr(t, "gps_satellites", None) if t else None,
        }


# ---------------------------------------------------------------------------
# Localization Manager
# ---------------------------------------------------------------------------

class LocalizationManager:
    """
    Central localization manager with configurable source and fallback chain.

    Configuration dict example::

        {"source": "mavlink_gps"}
        # or nested:
        {"localization": {"source": "mavlink_gps"}}

    Thread-safe: get_latest() and set_telemetry() are safe from any thread.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[LocalizationData], None]] = []

        source_name = (
            self._config.get("localization", {}).get("source", "none")
            if isinstance(self._config.get("localization"), dict)
            else self._config.get("source", "none")
        )

        self._source = self._build_source(source_name)
        self._latest: Optional[LocalizationData] = None
        logger.info("LocalizationManager: active source = %s", source_name)

    def _build_source(self, name: str) -> LocalizationSource:
        name = (name or "none").lower()
        if name == "mavlink_gps":
            return MAVLinkLocalizationSource()
        if name in ("visual_odometry", "vo"):
            # Raises NotImplementedError — intentional; caller should catch
            from aerosentinel.localization.visual_odometry import VisualOdometry  # noqa: PLC0415
            return VisualOdometry()
        if name == "slam":
            from aerosentinel.localization.slam_adapter import SLAMAdapter  # noqa: PLC0415
            return SLAMAdapter()
        if name != "none":
            logger.warning("Unknown localization source '%s'; using NullLocalizationSource", name)
        return NullLocalizationSource()

    def switch_source(self, name: str) -> None:
        """Hot-swap the active localization source (thread-safe)."""
        new_source = self._build_source(name)
        with self._lock:
            self._source = new_source
        logger.info("LocalizationManager: switched to source '%s'", name)

    def set_telemetry(self, telemetry: Any) -> None:
        """Push fresh MAVLink telemetry into MAVLinkLocalizationSource (no-op otherwise)."""
        with self._lock:
            src = self._source
        if isinstance(src, MAVLinkLocalizationSource):
            src.update(telemetry)

    def get_latest(self) -> LocalizationData:
        """Return the most recent LocalizationData; always returns a valid object."""
        with self._lock:
            src = self._source
        try:
            snap = src.get_snapshot()
        except Exception as exc:
            logger.error("LocalizationManager: source error (%s); returning null data", exc)
            snap = NullLocalizationSource().get_snapshot()

        with self._lock:
            self._latest = snap

        for cb in self._callbacks:
            try:
                cb(snap)
            except Exception as exc:
                logger.warning("LocalizationManager: callback error: %s", exc)

        return snap

    def register_callback(self, fn: Callable[[LocalizationData], None]) -> None:
        """Register a callback invoked on every get_latest() call."""
        self._callbacks.append(fn)

    def get_source_status(self) -> Dict[str, Any]:
        """Return the raw status dict from the active source."""
        with self._lock:
            src = self._source
        try:
            return src.get_status()
        except Exception as exc:
            return {"source": "error", "error": str(exc)}
