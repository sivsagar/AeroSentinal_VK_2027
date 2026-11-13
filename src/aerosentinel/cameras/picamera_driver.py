"""
picamera_driver.py — Pi Camera Module 3 driver for AeroSentinel.

Uses the picamera2 library (libcamera-based).  When picamera2 is not
installed (e.g., on a Windows dev machine), importing this module raises
an ImportError with a helpful installation message.

Thread safety
-------------
A background daemon thread feeds a maxsize=2 Queue so the consumer always
gets the most recent frame and stale frames are silently dropped.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    from picamera2 import Picamera2  # type: ignore
    from libcamera import controls as _lc_controls  # type: ignore
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False

# Deferred so the ImportError is only raised at instantiation time
from aerosentinel.cameras.camera_manager import BaseRGBDriver

logger = logging.getLogger(__name__)

_DEFAULT_RESOLUTION: List[int] = [1280, 720]
_DEFAULT_FPS: int = 15


class PiCameraDriver(BaseRGBDriver):
    """
    Driver for Raspberry Pi Camera Module 3 via picamera2 / libcamera.

    Parameters
    ----------
    config : dict
        Section from camera.yaml under ``rgb_camera``.  Recognised keys:

        * ``resolution``  – [width, height], default [1280, 720]
        * ``fps``         – integer, default 15
        * ``exposure``    – ``"auto"`` or integer microseconds
        * ``brightness``  – 0–100 (mapped to 0.0–1.0 for libcamera)
        * ``reconnect_interval_s`` – seconds between reconnection attempts

    Raises
    ------
    ImportError
        If picamera2 / libcamera are not installed on this system.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if not _PICAMERA2_AVAILABLE:
            raise ImportError(
                "picamera2 is not installed.  Install it on a Raspberry Pi with:\n"
                "  sudo apt-get install -y python3-picamera2\n"
                "or via pip (Pi OS Bookworm 64-bit):\n"
                "  pip install picamera2\n"
                "On Windows/Linux dev machines, set mock=true in camera.yaml."
            )

        cfg = config or {}
        res = cfg.get("resolution", _DEFAULT_RESOLUTION)
        self._resolution: Tuple[int, int] = (int(res[0]), int(res[1]))
        self._fps: int = int(cfg.get("fps", _DEFAULT_FPS))
        self._exposure = cfg.get("exposure", "auto")
        self._brightness: float = float(cfg.get("brightness", 50)) / 100.0
        self._reconnect_interval: float = float(cfg.get("reconnect_interval_s", 5))

        self._camera: Optional[Picamera2] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._connected = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> Tuple[int, int]:
        """Current (width, height) in pixels."""
        return self._resolution

    @property
    def fps(self) -> int:
        """Configured frames-per-second."""
        return self._fps

    @property
    def is_running(self) -> bool:
        """True while the acquisition thread is active."""
        return self._running

    # ------------------------------------------------------------------
    # BaseCameraDriver implementation
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise Picamera2 and start the background acquisition thread."""
        with self._lock:
            if self._running:
                logger.warning("PiCameraDriver.start() called while already running.")
                return
            self._stop_event.clear()
            self._init_camera()
            self._thread = threading.Thread(
                target=self._acquisition_loop,
                name="picam-acquire",
                daemon=True,
            )
            self._running = True
            self._thread.start()
            logger.info(
                "PiCameraDriver started — resolution=%s fps=%d", self._resolution, self._fps
            )

    def stop(self) -> None:
        """Signal the acquisition thread to stop and release the camera."""
        self._stop_event.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._release_camera()
        logger.info("PiCameraDriver stopped.")

    def is_connected(self) -> bool:
        """Return True if the camera has been successfully opened."""
        return self._connected

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Return the most recent (BGR frame, monotonic timestamp) or None.

        Non-blocking — returns None immediately when no frame is queued.
        """
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_camera(self) -> None:
        """Instantiate and configure Picamera2."""
        try:
            cam = Picamera2()
            config = cam.create_video_configuration(
                main={"size": self._resolution, "format": "BGR888"},
                controls={"FrameRate": float(self._fps)},
            )
            cam.configure(config)

            # Apply exposure setting
            if self._exposure != "auto":
                cam.set_controls({"ExposureTime": int(self._exposure), "AeEnable": False})
            else:
                cam.set_controls({"AeEnable": True})

            cam.set_controls({"Brightness": self._brightness})
            cam.start()
            self._camera = cam
            self._connected = True
            logger.info("PiCameraDriver: Picamera2 initialised.")
        except Exception as exc:  # noqa: BLE001
            logger.error("PiCameraDriver: Camera init failed: %s", exc)
            self._connected = False

    def _release_camera(self) -> None:
        """Stop and close the Picamera2 instance."""
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("PiCameraDriver: Error releasing camera: %s", exc)
            finally:
                self._camera = None
                self._connected = False

    def _acquisition_loop(self) -> None:
        """Background thread: continuously capture frames into the queue."""
        while not self._stop_event.is_set():
            if self._camera is None or not self._connected:
                logger.warning("PiCameraDriver: Camera not connected; attempting reconnection...")
                time.sleep(self._reconnect_interval)
                self._init_camera()
                continue

            try:
                frame = self._camera.capture_array("main")  # BGR888 → (H,W,3) uint8
                ts = time.monotonic()

                # Drain queue if full (drop oldest stale frame)
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

            except Exception as exc:  # noqa: BLE001
                logger.error("PiCameraDriver: Frame capture error: %s — reconnecting.", exc)
                self._connected = False
                self._release_camera()
                time.sleep(self._reconnect_interval)
                self._init_camera()
