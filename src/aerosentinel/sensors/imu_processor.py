"""
imu_processor.py — IMU data processing for AeroSentinel.

Applies low-pass filtering to raw accelerometer and gyroscope data from the
RealSense D435i (or a mock source), derives tilt estimates, logs to CSV, and
monitors for data dropouts.

⚠️  IMPORTANT: The roll/pitch estimates produced here are based on a simple
static tilt calculation from gravity direction.  They are NOT suitable for
flight stabilisation or navigation.  They indicate approximate sensor tilt
only and should be used for data-quality monitoring exclusively.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.aerosentinel.cameras.camera_manager import IMUData

logger = logging.getLogger(__name__)

_DEFAULT_ACCEL_ALPHA = 0.2    # low-pass coefficient: smaller = smoother
_DEFAULT_GYRO_ALPHA = 0.3
_DEFAULT_DROPOUT_TIMEOUT = 2.0   # seconds without data before dropout alarm


class IMUProcessor:
    """
    Receives raw IMUData readings and produces filtered, annotated readings.

    Parameters
    ----------
    accel_alpha : float
        Low-pass filter coefficient for accelerometer (0 < α ≤ 1).
        Smaller values → heavier smoothing.
    gyro_alpha : float
        Low-pass filter coefficient for gyroscope.
    dropout_timeout_s : float
        Seconds without a new reading before ``is_healthy`` returns False.
    log_csv_path : str, optional
        When given, raw + filtered readings are appended to this CSV file.
    """

    def __init__(
        self,
        accel_alpha: float = _DEFAULT_ACCEL_ALPHA,
        gyro_alpha: float = _DEFAULT_GYRO_ALPHA,
        dropout_timeout_s: float = _DEFAULT_DROPOUT_TIMEOUT,
        log_csv_path: Optional[str] = None,
    ) -> None:
        self._accel_alpha = max(0.0, min(1.0, accel_alpha))
        self._gyro_alpha = max(0.0, min(1.0, gyro_alpha))
        self._dropout_timeout = dropout_timeout_s
        self._log_path = log_csv_path

        # Filter state
        self._ax = self._ay = self._az = 0.0
        self._gx = self._gy = self._gz = 0.0
        self._filter_initialized = False

        # Latest processed reading (thread-safe via lock)
        self._lock = threading.Lock()
        self._latest: Optional[IMUData] = None
        self._last_update_time: Optional[float] = None

        # CSV logging
        self._csv_file = None
        self._csv_writer = None
        if log_csv_path:
            self._open_csv(log_csv_path)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _open_csv(self, path: str) -> None:
        """Open (or create) the CSV log file."""
        try:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            is_new = not os.path.exists(path)
            self._csv_file = open(path, "a", newline="")  # noqa: WPS515
            self._csv_writer = csv.writer(self._csv_file)
            if is_new:
                self._csv_writer.writerow([
                    "timestamp",
                    "accel_x", "accel_y", "accel_z",
                    "gyro_x", "gyro_y", "gyro_z",
                    "filt_ax", "filt_ay", "filt_az",
                    "filt_gx", "filt_gy", "filt_gz",
                    "roll_deg", "pitch_deg",
                    "is_valid",
                ])
            logger.info("IMUProcessor: logging to %s", path)
        except OSError as exc:
            logger.error("IMUProcessor: cannot open CSV %s: %s", path, exc)
            self._csv_writer = None

    def close(self) -> None:
        """Flush and close the CSV log file."""
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except OSError:
                pass
            self._csv_file = None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, raw: IMUData) -> IMUData:
        """
        Apply low-pass filters, compute tilt, and store the result.

        Parameters
        ----------
        raw : IMUData
            Unfiltered reading from the camera driver or mock source.

        Returns
        -------
        IMUData
            Same timestamps as *raw*, but with filtered values and
            updated roll_estimate / pitch_estimate.

        Notes
        -----
        Roll and pitch are derived from the accelerometer gravity vector.
        This method is **not** suitable for flight stabilisation.
        """
        alpha_a = self._accel_alpha
        alpha_g = self._gyro_alpha

        if not self._filter_initialized:
            self._ax, self._ay, self._az = raw.accel_x, raw.accel_y, raw.accel_z
            self._gx, self._gy, self._gz = raw.gyro_x, raw.gyro_y, raw.gyro_z
            self._filter_initialized = True
        else:
            # Exponential moving average (low-pass)
            self._ax = alpha_a * raw.accel_x + (1 - alpha_a) * self._ax
            self._ay = alpha_a * raw.accel_y + (1 - alpha_a) * self._ay
            self._az = alpha_a * raw.accel_z + (1 - alpha_a) * self._az
            self._gx = alpha_g * raw.gyro_x + (1 - alpha_g) * self._gx
            self._gy = alpha_g * raw.gyro_y + (1 - alpha_g) * self._gy
            self._gz = alpha_g * raw.gyro_z + (1 - alpha_g) * self._gz

        roll_rad, pitch_rad = self._tilt_estimate(self._ax, self._ay, self._az)

        processed = IMUData(
            timestamp=raw.timestamp,
            accel_x=self._ax,
            accel_y=self._ay,
            accel_z=self._az,
            gyro_x=self._gx,
            gyro_y=self._gy,
            gyro_z=self._gz,
            roll_estimate=math.degrees(roll_rad),
            pitch_estimate=math.degrees(pitch_rad),
            is_valid=raw.is_valid,
        )

        with self._lock:
            self._latest = processed
            self._last_update_time = time.monotonic()

        self._log(raw, processed)
        return processed

    @staticmethod
    def _tilt_estimate(ax: float, ay: float, az: float) -> tuple:
        """
        Simple static tilt from gravity vector.

        ⚠️  NOT navigation-grade.  Assumes sensor is nearly stationary.
        Returns (roll_rad, pitch_rad).
        """
        magnitude = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        if magnitude < 1e-6:
            return 0.0, 0.0
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay ** 2 + az ** 2))
        return roll, pitch

    def _log(self, raw: IMUData, filt: IMUData) -> None:
        """Write a row to the CSV log if configured."""
        if self._csv_writer is None:
            return
        try:
            self._csv_writer.writerow([
                raw.timestamp,
                raw.accel_x, raw.accel_y, raw.accel_z,
                raw.gyro_x, raw.gyro_y, raw.gyro_z,
                filt.accel_x, filt.accel_y, filt.accel_z,
                filt.gyro_x, filt.gyro_y, filt.gyro_z,
                filt.roll_estimate, filt.pitch_estimate,
                filt.is_valid,
            ])
        except OSError as exc:
            logger.debug("IMUProcessor CSV write error: %s", exc)

    # ------------------------------------------------------------------
    # Thread-safe accessors
    # ------------------------------------------------------------------

    @property
    def latest_reading(self) -> Optional[IMUData]:
        """Return the most recent processed IMUData (thread-safe)."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """True if a reading was processed within the dropout timeout window."""
        with self._lock:
            if self._last_update_time is None:
                return False
            age = time.monotonic() - self._last_update_time
        return age < self._dropout_timeout

    @property
    def last_update_age_s(self) -> Optional[float]:
        """Seconds since last successful reading, or None if never received."""
        with self._lock:
            if self._last_update_time is None:
                return None
            return time.monotonic() - self._last_update_time
