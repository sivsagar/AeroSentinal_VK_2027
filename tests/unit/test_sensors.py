"""
Unit Tests for IMUProcessor and SensorHealthMonitor (Phase 2)
=============================================================
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))

from aerosentinel.cameras.camera_manager import IMUData
from aerosentinel.sensors.imu_processor import IMUProcessor
from aerosentinel.sensors.sensor_health import SensorHealthMonitor, SensorStatus


class TestIMUProcessor:
    def test_filter_and_tilt(self):
        proc = IMUProcessor(accel_alpha=0.5, gyro_alpha=0.5, dropout_timeout_s=1.0)
        reading1 = IMUData(
            timestamp=time.time(),
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            is_valid=True,
        )
        out = proc.process(reading1)
        assert out is not None
        assert abs(out.accel_z - 9.81) < 0.1
        assert abs(out.roll_estimate) < 5.0
        assert abs(out.pitch_estimate) < 5.0
        assert proc.is_healthy

    def test_dropout_detection(self):
        proc = IMUProcessor(dropout_timeout_s=0.2)
        reading = IMUData(
            timestamp=time.time() - 1.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            is_valid=True,
        )
        proc.process(reading)
        time.sleep(0.3)
        assert not proc.is_healthy


class DummyCam:
    def __init__(self, connected=True):
        self._conn = connected

    def is_connected(self):
        return self._conn


class DummyCamManager:
    def __init__(self):
        self.rgb_driver = DummyCam(True)
        self.depth_driver = DummyCam(True)

    def status(self):
        return {
            "rgb_camera": {"connected": True},
            "depth_camera": {"connected": True},
        }


class TestSensorHealthMonitor:
    def test_health_evaluation(self):
        mgr = DummyCamManager()
        imu = IMUProcessor()
        monitor = SensorHealthMonitor(camera_manager=mgr, imu_processor=imu, check_interval_s=0.05)
        monitor.start()
        time.sleep(0.1)
        rep = monitor.health_report()
        monitor.stop()

        assert "rgb" in rep
        assert "depth" in rep
        assert "imu" in rep

