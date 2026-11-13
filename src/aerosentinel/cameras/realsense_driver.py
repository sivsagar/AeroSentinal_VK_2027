"""
realsense_driver.py — Intel RealSense D435i driver for AeroSentinel.

Uses pyrealsense2 (librealsense Python bindings).  When pyrealsense2 is not
installed, importing this module raises ImportError with helpful guidance.

Streams
-------
* Depth  — 640×480 Z16 (uint16, millimetres)
* Color  — 640×480 BGR8
* IMU    — accelerometer + gyroscope via MOTION_XYZ32F

Thread safety
-------------
Three independent maxsize=2 Queues (depth/color combo, IMU).
Stale frames are silently dropped to keep latency minimal.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

from aerosentinel.cameras.camera_manager import BaseDepthDriver, IMUData

logger = logging.getLogger(__name__)

_DEFAULT_DEPTH_RES: List[int] = [640, 480]
_DEFAULT_COLOR_RES: List[int] = [640, 480]
_DEFAULT_FPS: int = 15
_DEPTH_SCALE_DEFAULT: float = 0.001  # librealsense default: 1 unit = 1 mm → 0.001 m


class RealSenseCameraDriver(BaseDepthDriver):
    """
    Driver for Intel RealSense D435i (depth + colour + IMU).

    Parameters
    ----------
    config : dict
        Section from camera.yaml under ``depth_camera``.  Recognised keys:

        * ``depth_resolution``    – [width, height], default [640, 480]
        * ``color_resolution``    – [width, height], default [640, 480]
        * ``fps``                 – integer, default 15
        * ``usb_timeout_s``       – USB wait timeout, default 10
        * ``reconnect_interval_s``– seconds between reconnection attempts

    Raises
    ------
    ImportError
        If pyrealsense2 is not installed.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if not _RS_AVAILABLE:
            raise ImportError(
                "pyrealsense2 is not installed.\n"
                "On Raspberry Pi OS Bookworm 64-bit:\n"
                "  Build librealsense from source (ARM64) then:\n"
                "    pip install pyrealsense2\n"
                "  or use a pre-built wheel from:\n"
                "    https://github.com/IntelRealSense/librealsense/releases\n"
                "On Windows dev machines, set mock=true in camera.yaml."
            )

        cfg = config or {}
        dr = cfg.get("depth_resolution", _DEFAULT_DEPTH_RES)
        cr = cfg.get("color_resolution", _DEFAULT_COLOR_RES)
        self._depth_resolution: Tuple[int, int] = (int(dr[0]), int(dr[1]))
        self._color_resolution: Tuple[int, int] = (int(cr[0]), int(cr[1]))
        self._fps: int = int(cfg.get("fps", _DEFAULT_FPS))
        self._usb_timeout: float = float(cfg.get("usb_timeout_s", 10))
        self._reconnect_interval: float = float(cfg.get("reconnect_interval_s", 5))

        self._pipeline: Optional[rs.pipeline] = None
        self._config_rs: Optional[rs.config] = None
        self._align: Optional[rs.align] = None
        self._depth_scale: float = _DEPTH_SCALE_DEFAULT

        # Filters
        self._dec_filter = rs.decimation_filter()
        self._spat_filter = rs.spatial_filter()
        self._temp_filter = rs.temporal_filter()
        self._hole_filter = rs.hole_filling_filter()

        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)   # (depth, color, ts)
        self._imu_queue: queue.Queue = queue.Queue(maxsize=2)     # IMUData

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._connected = False
        self._lock = threading.Lock()

        # Latest IMU state for filter continuity
        self._latest_imu: Optional[IMUData] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # BaseCameraDriver implementation
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Configure and start RealSense pipeline + background acquisition thread."""
        with self._lock:
            if self._running:
                logger.warning("RealSenseCameraDriver.start() called while already running.")
                return
            self._stop_event.clear()
            self._init_pipeline()
            self._thread = threading.Thread(
                target=self._acquisition_loop,
                name="rs-acquire",
                daemon=True,
            )
            self._running = True
            self._thread.start()
            logger.info("RealSenseCameraDriver started.")

    def stop(self) -> None:
        """Stop the acquisition thread and pipeline."""
        self._stop_event.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._release_pipeline()
        logger.info("RealSenseCameraDriver stopped.")

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Return (depth_uint16_mm, color_BGR, timestamp) or None."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_imu(self) -> Optional[IMUData]:
        """Return latest IMUData or None."""
        try:
            return self._imu_queue.get_nowait()
        except queue.Empty:
            return self._latest_imu  # return last known reading rather than None

    # ------------------------------------------------------------------
    # Depth helpers
    # ------------------------------------------------------------------

    def depth_at_pixel(self, x: int, y: int) -> Optional[float]:
        """
        Return depth in metres at pixel (x, y), or None if invalid.

        Uses the last queued frame; non-blocking.
        """
        result = self.get_frames()
        if result is None:
            return None
        depth_frame, _, _ = result
        if y < 0 or y >= depth_frame.shape[0] or x < 0 or x >= depth_frame.shape[1]:
            return None
        raw = int(depth_frame[y, x])
        if raw == 0:
            return None
        return raw * self._depth_scale

    def depth_in_region(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Dict[str, object]:
        """
        Return depth statistics (metres) for a rectangular region.

        Returns
        -------
        dict with keys: min, max, mean, valid_count
        """
        result = self.get_frames()
        if result is None:
            return {"min": None, "max": None, "mean": None, "valid_count": 0}
        depth_frame, _, _ = result
        roi = depth_frame[
            max(0, y1) : min(depth_frame.shape[0], y2),
            max(0, x1) : min(depth_frame.shape[1], x2),
        ]
        valid = roi[roi > 0].astype(np.float32) * self._depth_scale
        if valid.size == 0:
            return {"min": None, "max": None, "mean": None, "valid_count": 0}
        return {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "valid_count": int(valid.size),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_pipeline(self) -> None:
        """Create and start the RealSense pipeline."""
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(
                rs.stream.depth,
                self._depth_resolution[0],
                self._depth_resolution[1],
                rs.format.z16,
                self._fps,
            )
            cfg.enable_stream(
                rs.stream.color,
                self._color_resolution[0],
                self._color_resolution[1],
                rs.format.bgr8,
                self._fps,
            )
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)

            profile = pipeline.start(cfg)
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()
            self._align = rs.align(rs.stream.color)
            self._pipeline = pipeline
            self._config_rs = cfg
            self._connected = True
            logger.info(
                "RealSense pipeline started — depth_scale=%.6f m/unit", self._depth_scale
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("RealSense init failed: %s", exc)
            self._connected = False

    def _release_pipeline(self) -> None:
        """Stop the RealSense pipeline."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("RealSense release error: %s", exc)
            finally:
                self._pipeline = None
                self._connected = False

    def _apply_depth_filters(self, depth_frame: "rs.depth_frame") -> "rs.depth_frame":
        """Apply decimation → spatial → temporal → hole-filling filters."""
        frame = self._dec_filter.process(depth_frame)
        frame = self._spat_filter.process(frame)
        frame = self._temp_filter.process(frame)
        frame = self._hole_filter.process(frame)
        return frame.as_depth_frame()

    def _acquisition_loop(self) -> None:
        """Background thread: grab frameset and dispatch to queues."""
        accel: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        gyro: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}

        while not self._stop_event.is_set():
            if self._pipeline is None or not self._connected:
                logger.warning("RealSense: not connected; reconnecting in %.1fs…", self._reconnect_interval)
                time.sleep(self._reconnect_interval)
                self._init_pipeline()
                continue

            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=int(self._usb_timeout * 1000))
            except RuntimeError as exc:
                logger.error("RealSense: wait_for_frames timeout/error: %s", exc)
                self._connected = False
                self._release_pipeline()
                time.sleep(self._reconnect_interval)
                self._init_pipeline()
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error("RealSense: unexpected error: %s", exc)
                self._connected = False
                self._release_pipeline()
                time.sleep(self._reconnect_interval)
                continue

            ts = time.monotonic()

            # --- IMU frames ---
            for stream in (rs.stream.accel, rs.stream.gyro):
                try:
                    motion = frames.first_or_default(stream)
                    if motion and motion.is_motion_frame():
                        data = motion.as_motion_frame().get_motion_data()
                        if stream == rs.stream.accel:
                            accel = {"x": data.x, "y": data.y, "z": data.z}
                        else:
                            gyro = {"x": data.x, "y": data.y, "z": data.z}
                except Exception:  # noqa: BLE001
                    pass

            imu = IMUData(
                timestamp=ts,
                accel_x=accel["x"],
                accel_y=accel["y"],
                accel_z=accel["z"],
                gyro_x=gyro["x"],
                gyro_y=gyro["y"],
                gyro_z=gyro["z"],
            )
            self._latest_imu = imu
            self._put_queue(self._imu_queue, imu)

            # --- Depth + Color frames ---
            try:
                aligned = self._align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                filtered_depth = self._apply_depth_filters(depth_frame)
                depth_arr = np.asanyarray(filtered_depth.get_data())   # uint16, mm
                color_arr = np.asanyarray(color_frame.get_data())       # BGR uint8

                self._put_queue(self._frame_queue, (depth_arr, color_arr, ts))

            except Exception as exc:  # noqa: BLE001
                logger.debug("RealSense frame processing error: %s", exc)

    @staticmethod
    def _put_queue(q: queue.Queue, item: object) -> None:
        """Non-blocking put; evict oldest item on Full."""
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
