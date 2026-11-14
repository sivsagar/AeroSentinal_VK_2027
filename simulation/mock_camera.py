"""
mock_camera.py — Software-only camera and IMU simulators for AeroSentinel.

Provides MockRGBCamera, MockDepthCamera, and MockIMU that implement the same
interfaces as the hardware drivers so the full pipeline can be tested on any
development machine (Windows / Linux) without physical hardware.

Failure injection
-----------------
Pass ``fail_after_frames=N`` to simulate a USB disconnection after N frames.
"""

from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

# Local import of base classes (no hardware dependency)
try:
    from aerosentinel.cameras.camera_manager import (
        BaseDepthDriver,
        BaseRGBDriver,
        IMUData,
    )
except ImportError:
    from src.aerosentinel.cameras.camera_manager import (
        BaseDepthDriver,
        BaseRGBDriver,
        IMUData,
    )

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_FPS = 15


# ---------------------------------------------------------------------------
# MockRGBCamera
# ---------------------------------------------------------------------------

class MockRGBCamera(BaseRGBDriver):
    """
    Synthetic RGB camera that generates solid-colour frames with an
    overlaid frame counter and ISO-format timestamp.

    Parameters
    ----------
    config : dict
        Accepts ``resolution`` ([w, h]) and ``fps`` keys.
    fail_after_frames : int, optional
        When given, the camera simulates a disconnection after this many
        frames.  ``is_connected()`` returns False and frames stop.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        fail_after_frames: Optional[int] = None,
    ) -> None:
        cfg = config or {}
        res = cfg.get("resolution", [_DEFAULT_WIDTH, _DEFAULT_HEIGHT])
        self._width = int(res[0])
        self._height = int(res[1])
        self._fps = int(cfg.get("fps", _DEFAULT_FPS))
        self._fail_after = fail_after_frames

        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._connected = False
        self._frame_count = 0

    # Properties
    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running

    # Lifecycle
    def start(self) -> None:
        self._stop_event.clear()
        self._connected = True
        self._thread = threading.Thread(
            target=self._produce, name="mock-rgb", daemon=True
        )
        self._running = True
        self._thread.start()
        logger.info("MockRGBCamera started (%dx%d @ %d fps).", self._width, self._height, self._fps)

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._connected = False
        logger.info("MockRGBCamera stopped.")

    def is_connected(self) -> bool:
        return self._connected

    # Frame access
    def get_frame(self, timeout_s: float = 0.0) -> Optional[Tuple[np.ndarray, float]]:
        try:
            if timeout_s > 0:
                return self._frame_queue.get(timeout=timeout_s)
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    # Private
    def _produce(self) -> None:
        interval = 1.0 / max(self._fps, 1)
        while not self._stop_event.is_set():
            if not self._connected:
                time.sleep(0.1)
                continue

            if self._fail_after is not None and self._frame_count >= self._fail_after:
                logger.info("MockRGBCamera: simulating disconnection at frame %d.", self._frame_count)
                self._connected = False
                continue

            frame = self._generate_frame()
            ts = time.monotonic()
            try:
                self._frame_queue.put_nowait((frame, ts))
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait((frame, ts))
                except queue.Full:
                    pass

            self._frame_count += 1
            time.sleep(interval)

    def _generate_frame(self) -> np.ndarray:
        """Create a solid-colour BGR frame with text overlay."""
        # Cycle hue over time for visual feedback
        hue = int((time.monotonic() * 30) % 180)
        # Create HSV single-colour image and convert to BGR
        hsv = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = 180
        hsv[:, :, 2] = 200

        if _CV2_AVAILABLE:
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            label = f"MockRGB #{self._frame_count} | {time.strftime('%H:%M:%S')}"
            cv2.putText(
                frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
        else:
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            frame[:, :, 0] = 100  # Simple blue channel for no-cv2 fallback
        return frame


# ---------------------------------------------------------------------------
# MockDepthCamera
# ---------------------------------------------------------------------------

class MockDepthCamera(BaseDepthDriver):
    """
    Synthetic depth + colour camera with matching IMU stream.

    Depth frames are gradient images with configurable synthetic obstacles.
    Colour frames match MockRGBCamera output.

    Parameters
    ----------
    config : dict
        Accepts ``depth_resolution``, ``color_resolution``, ``fps`` keys.
    fail_after_frames : int, optional
        Simulate USB disconnection after N frames.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        fail_after_frames: Optional[int] = None,
    ) -> None:
        try:
            from simulation.mock_depth import generate_depth_frame, generate_imu_data, DepthScenario
        except ModuleNotFoundError:
            from mock_depth import generate_depth_frame, generate_imu_data, DepthScenario

        self._gen_depth = generate_depth_frame
        self._gen_imu = generate_imu_data
        self._scenario = DepthScenario.MULTI_OBSTACLE

        cfg = config or {}
        dr = cfg.get("depth_resolution", [_DEFAULT_WIDTH, _DEFAULT_HEIGHT])
        cr = cfg.get("color_resolution", [_DEFAULT_WIDTH, _DEFAULT_HEIGHT])
        self._dw, self._dh = int(dr[0]), int(dr[1])
        self._cw, self._ch = int(cr[0]), int(cr[1])
        self._fps = int(cfg.get("fps", _DEFAULT_FPS))
        self._fail_after = fail_after_frames

        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._imu_queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._connected = False
        self._frame_count = 0
        self._latest_imu: Optional[IMUData] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._stop_event.clear()
        self._connected = True
        self._thread = threading.Thread(
            target=self._produce, name="mock-depth", daemon=True
        )
        self._running = True
        self._thread.start()
        logger.info("MockDepthCamera started.")

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._connected = False
        logger.info("MockDepthCamera stopped.")

    def is_connected(self) -> bool:
        return self._connected

    def get_frames(self, timeout_s: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        try:
            if timeout_s > 0:
                item = self._frame_queue.get(timeout=timeout_s)
            else:
                item = self._frame_queue.get_nowait()
            self._latest_depth = item[0]
            return item
        except queue.Empty:
            return None

    def get_imu(self) -> Optional[IMUData]:
        try:
            return self._imu_queue.get_nowait()
        except queue.Empty:
            return self._latest_imu

    def depth_at_pixel(self, x: int, y: int) -> Optional[float]:
        depth = getattr(self, "_latest_depth", None)
        if depth is None:
            result = self.get_frames(timeout_s=0.5)
            if result is None:
                return None
            depth = result[0]
        if y < 0 or y >= depth.shape[0] or x < 0 or x >= depth.shape[1]:
            return None
        raw = int(depth[y, x])
        return None if raw == 0 else raw / 1000.0  # mm → m

    def depth_in_region(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Dict[str, object]:
        depth = getattr(self, "_latest_depth", None)
        if depth is None:
            result = self.get_frames(timeout_s=0.5)
            if result is None:
                return {"min": None, "max": None, "mean": None, "valid_count": 0}
            depth = result[0]
        roi = depth[max(0, y1):min(depth.shape[0], y2),
                    max(0, x1):min(depth.shape[1], x2)]
        valid = roi[roi > 0].astype(np.float32) / 1000.0
        if valid.size == 0:
            return {"min": None, "max": None, "mean": None, "valid_count": 0}
        return {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "valid_count": int(valid.size),
        }

    def _produce(self) -> None:
        interval = 1.0 / max(self._fps, 1)
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            if not self._connected:
                time.sleep(0.1)
                continue

            if self._fail_after is not None and self._frame_count >= self._fail_after:
                logger.info("MockDepthCamera: simulating disconnection at frame %d.", self._frame_count)
                self._connected = False
                continue

            t = time.monotonic() - t0
            depth_arr = self._gen_depth(self._dw, self._dh)
            color_arr = np.zeros((self._ch, self._cw, 3), dtype=np.uint8)
            color_arr[:, :, 1] = 80  # green tint for depth colour
            ts = time.monotonic()

            imu = self._gen_imu(t)
            self._latest_imu = imu

            self._put(self._frame_queue, (depth_arr, color_arr, ts))
            self._put(self._imu_queue, imu)

            self._frame_count += 1
            time.sleep(interval)

    @staticmethod
    def _put(q: queue.Queue, item: object) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# MockIMU (standalone, used directly in tests)
# ---------------------------------------------------------------------------

class MockIMU:
    """
    Standalone mock IMU that generates sinusoidal accelerometer + gyro data.

    Parameters
    ----------
    fail_after : int, optional
        Stop returning valid data after this many calls.
    """

    def __init__(self, fail_after: Optional[int] = None) -> None:
        self._fail_after = fail_after
        self._count = 0
        self._t0 = time.monotonic()

    def read(self) -> Optional[IMUData]:
        """Return a synthetic IMUData reading, or None on simulated failure."""
        if self._fail_after is not None and self._count >= self._fail_after:
            return None

        t = time.monotonic() - self._t0
        # Gravity on Z axis (~9.81 m/s²) with small oscillation
        imu = IMUData(
            timestamp=time.monotonic(),
            accel_x=0.05 * math.sin(t),
            accel_y=0.05 * math.cos(t),
            accel_z=9.81 + 0.02 * math.sin(2 * t),
            gyro_x=0.01 * math.sin(t * 0.5),
            gyro_y=0.01 * math.cos(t * 0.5),
            gyro_z=0.005 * math.sin(t * 0.3),
            is_valid=True,
        )
        self._count += 1
        return imu
