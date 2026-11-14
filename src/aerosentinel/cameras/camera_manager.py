"""
camera_manager.py — CameraManager and BaseCameraDriver for AeroSentinel.

Manages PiCamera (RGB) and RealSense (depth + color + IMU) drivers.
Supports mock mode for development on Windows / Linux without hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IMUData dataclass (shared by all sensor modules)
# ---------------------------------------------------------------------------

@dataclass
class IMUData:
    """Raw + derived IMU reading from RealSense or mock source."""

    timestamp: float          # time.monotonic()
    accel_x: float            # m/s²
    accel_y: float            # m/s²
    accel_z: float            # m/s²
    gyro_x: float             # rad/s
    gyro_y: float             # rad/s
    gyro_z: float             # rad/s
    roll_estimate: float = 0.0   # degrees (tilt-only, NOT navigation-grade)
    pitch_estimate: float = 0.0  # degrees (tilt-only, NOT navigation-grade)
    is_valid: bool = True


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseCameraDriver(ABC):
    """Abstract interface every camera driver must implement."""

    @abstractmethod
    def start(self) -> None:
        """Start the camera acquisition thread."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera acquisition thread and release resources."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the camera is reachable and operating."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """True while the acquisition thread is active."""


class BaseRGBDriver(BaseCameraDriver):
    """Extended interface for RGB-only cameras."""

    @abstractmethod
    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Return (BGR frame, monotonic timestamp) or None if no frame ready."""


class BaseDepthDriver(BaseCameraDriver):
    """Extended interface for depth + color cameras."""

    @abstractmethod
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Return (depth_uint16_mm, color_BGR, monotonic_timestamp) or None."""

    @abstractmethod
    def get_imu(self) -> Optional[IMUData]:
        """Return latest IMUData or None."""

    @abstractmethod
    def depth_at_pixel(self, x: int, y: int) -> Optional[float]:
        """Return depth in metres at pixel (x, y), or None if invalid."""

    @abstractmethod
    def depth_in_region(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Dict[str, object]:
        """Return dict with min, max, mean, valid_count for the region (metres)."""


# ---------------------------------------------------------------------------
# CameraManager
# ---------------------------------------------------------------------------

class CameraManager:
    """
    Orchestrates PiCamera (RGB) and RealSense (depth) drivers.

    Parameters
    ----------
    config : dict
        Parsed camera.yaml content.
    mock : bool
        When True, use mock drivers regardless of config flags.
    """

    def __init__(self, config: Optional[Dict] = None, mock: bool = False) -> None:
        self._config: Dict = config or {}
        self._mock = mock
        self._rgb_driver: Optional[BaseRGBDriver] = None
        self._depth_driver: Optional[BaseDepthDriver] = None
        self._lock = threading.Lock()
        self._init_drivers()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_drivers(self) -> None:
        """Lazily import and instantiate drivers based on config / mock flag."""
        rgb_cfg = self._config.get("rgb_camera")
        if rgb_cfg is None and ("resolution" in self._config or "format" in self._config):
            rgb_cfg = self._config
        rgb_cfg = rgb_cfg or {}

        depth_cfg = self._config.get("depth_camera")
        if depth_cfg is None and ("depth_resolution" in self._config or "color_resolution" in self._config):
            depth_cfg = self._config
        depth_cfg = depth_cfg or {}

        use_mock = self._mock or rgb_cfg.get("mock", False)
        use_mock_depth = self._mock or depth_cfg.get("mock", False)

        if use_mock:
            try:
                from mock_camera import MockRGBCamera
            except ModuleNotFoundError:
                try:
                    from simulation.mock_camera import MockRGBCamera
                except ModuleNotFoundError:
                    import sys
                    from pathlib import Path
                    sim_dir = Path(__file__).resolve().parents[3] / "simulation"
                    if str(sim_dir) not in sys.path:
                        sys.path.insert(0, str(sim_dir))
                    from mock_camera import MockRGBCamera
            self._rgb_driver = MockRGBCamera(config=rgb_cfg)
            logger.info("CameraManager: RGB driver → MockRGBCamera")
        elif rgb_cfg.get("enabled", True):
            try:
                from aerosentinel.cameras.picamera_driver import PiCameraDriver
                self._rgb_driver = PiCameraDriver(config=rgb_cfg)
                logger.info("CameraManager: RGB driver → PiCameraDriver")
            except ImportError as exc:
                logger.warning("PiCameraDriver unavailable (%s); RGB disabled.", exc)

        if use_mock_depth:
            try:
                from mock_camera import MockDepthCamera
            except ModuleNotFoundError:
                try:
                    from simulation.mock_camera import MockDepthCamera
                except ModuleNotFoundError:
                    import sys
                    from pathlib import Path
                    sim_dir = Path(__file__).resolve().parents[3] / "simulation"
                    if str(sim_dir) not in sys.path:
                        sys.path.insert(0, str(sim_dir))
                    from mock_camera import MockDepthCamera
            self._depth_driver = MockDepthCamera(config=depth_cfg)
            logger.info("CameraManager: Depth driver → MockDepthCamera")
        elif depth_cfg.get("enabled", True):
            try:
                from aerosentinel.cameras.realsense_driver import RealSenseCameraDriver
                self._depth_driver = RealSenseCameraDriver(config=depth_cfg)
                logger.info("CameraManager: Depth driver → RealSenseCameraDriver")
            except ImportError as exc:
                logger.warning("RealSenseCameraDriver unavailable (%s); Depth disabled.", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self) -> None:
        """Start all enabled camera drivers (independent — one failure won't block)."""
        for name, driver in [("RGB", self._rgb_driver), ("Depth", self._depth_driver)]:
            if driver is None:
                continue
            try:
                driver.start()
                logger.info("CameraManager: %s driver started.", name)
            except Exception as exc:  # noqa: BLE001
                logger.error("CameraManager: Failed to start %s driver: %s", name, exc)

    def stop_all(self) -> None:
        """Stop all enabled camera drivers gracefully."""
        for name, driver in [("RGB", self._rgb_driver), ("Depth", self._depth_driver)]:
            if driver is None:
                continue
            try:
                driver.stop()
                logger.info("CameraManager: %s driver stopped.", name)
            except Exception as exc:  # noqa: BLE001
                logger.error("CameraManager: Error stopping %s driver: %s", name, exc)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_rgb_frame(self, timeout_s: float = 0.0) -> Optional[Tuple[np.ndarray, float]]:
        """Return (BGR frame, timestamp) or None."""
        if self._rgb_driver is None:
            return None
        try:
            return self._rgb_driver.get_frame(timeout_s=timeout_s)
        except TypeError:
            return self._rgb_driver.get_frame()
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_rgb_frame error: %s", exc)
            return None

    def get_depth_frame(self, timeout_s: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Return (depth_uint16_mm, color_BGR, timestamp) or None."""
        if self._depth_driver is None:
            return None
        try:
            return self._depth_driver.get_frames(timeout_s=timeout_s)
        except TypeError:
            return self._depth_driver.get_frames()
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_depth_frame error: %s", exc)
            return None

    def get_imu_data(self) -> Optional[IMUData]:
        """Return latest IMUData from depth camera, or None."""
        if self._depth_driver is None:
            return None
        try:
            return self._depth_driver.get_imu()
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_imu_data error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Status / health
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, object]:
        """Return health dict for all cameras."""
        def _driver_status(driver: Optional[BaseCameraDriver], name: str) -> Dict:
            if driver is None:
                return {"enabled": False, "connected": False, "running": False}
            try:
                return {
                    "enabled": True,
                    "connected": driver.is_connected(),
                    "running": driver.is_running,
                }
            except Exception as exc:  # noqa: BLE001
                return {"enabled": True, "connected": False, "running": False, "error": str(exc)}

        return {
            "rgb": _driver_status(self._rgb_driver, "RGB"),
            "depth": _driver_status(self._depth_driver, "Depth"),
            "timestamp": time.monotonic(),
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraManager":
        self.start_all()
        return self

    def __exit__(self, *_) -> None:
        self.stop_all()
