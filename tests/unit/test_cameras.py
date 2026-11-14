"""
Unit Tests for Camera Drivers and CameraManager (Phase 2)
=========================================================
Uses MockRGBCamera and MockDepthCamera to verify contracts without hardware.
"""

import sys
import os
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))

from mock_camera import MockRGBCamera, MockDepthCamera, MockIMU
from aerosentinel.cameras.camera_manager import CameraManager


class TestMockRGBCamera:
    def test_start_stop(self):
        cam = MockRGBCamera(config={"resolution": [640, 480], "fps": 15})
        assert not cam.is_connected()
        cam.start()
        assert cam.is_connected()
        frame, ts = cam.get_frame(timeout_s=1.0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert ts > 0
        cam.stop()
        assert not cam.is_connected()

    def test_fail_after_frames(self):
        cam = MockRGBCamera(config={"resolution": [320, 240], "fps": 30}, fail_after_frames=3)
        cam.start()
        time.sleep(0.3)
        # after 3 frames it should fail
        assert not cam.is_connected()
        cam.stop()


class TestMockDepthCamera:
    def test_depth_frame_acquisition(self):
        cam = MockDepthCamera(config={"depth_resolution": [320, 240], "color_resolution": [320, 240], "fps": 15})
        cam.start()
        depth, color, ts = cam.get_frames(timeout_s=1.0)
        assert depth is not None
        assert depth.shape == (240, 320)
        assert depth.dtype == np.uint16
        assert color is not None
        assert color.shape == (240, 320, 3)

        # test depth at pixel
        dist = cam.depth_at_pixel(160, 120)
        assert dist is not None
        assert 0.1 <= dist <= 20.0

        # test depth in region
        reg = cam.depth_in_region(100, 100, 200, 200)
        assert "min" in reg
        assert "mean" in reg
        assert "valid_count" in reg
        cam.stop()


class TestCameraManager:
    def test_manager_mock_mode(self):
        cfg = {
            "rgb_camera": {"enabled": True, "resolution": [320, 240], "fps": 15, "mock": True},
            "depth_camera": {"enabled": True, "depth_resolution": [320, 240], "fps": 15, "mock": True},
        }
        mgr = CameraManager(config=cfg, mock=True)
        mgr.start_all()
        status = mgr.status()
        assert "rgb" in status
        assert "depth" in status

        rgb = mgr.get_rgb_frame(timeout_s=1.0)
        assert rgb is not None
        assert rgb[0].shape == (240, 320, 3)

        depth = mgr.get_depth_frame(timeout_s=1.0)
        assert depth is not None
        assert depth[0].shape == (240, 320)

        mgr.stop_all()


class TestRealSenseIMUConfiguration:
    """Verify that RealSense D435i requests 250 Hz for accel and 400 Hz for gyro."""

    def test_default_imu_fps_constants(self):
        from aerosentinel.cameras.realsense_driver import (
            _DEFAULT_ACCEL_FPS,
            _DEFAULT_GYRO_FPS,
        )
        assert _DEFAULT_ACCEL_FPS == 250
        assert _DEFAULT_GYRO_FPS == 400

    def test_realsense_driver_imu_stream_rates(self):
        from unittest.mock import MagicMock
        import aerosentinel.cameras.realsense_driver as rs_mod

        # Construct mock pyrealsense2
        mock_rs = MagicMock()
        mock_rs.stream.depth = "STREAM_DEPTH"
        mock_rs.stream.color = "STREAM_COLOR"
        mock_rs.stream.accel = "STREAM_ACCEL"
        mock_rs.stream.gyro = "STREAM_GYRO"
        mock_rs.format.z16 = "FORMAT_Z16"
        mock_rs.format.bgr8 = "FORMAT_BGR8"
        mock_rs.format.motion_xyz32f = "FORMAT_XYZ32F"

        mock_pipeline = MagicMock()
        mock_cfg = MagicMock()
        mock_rs.pipeline.return_value = mock_pipeline
        mock_rs.config.return_value = mock_cfg

        mock_profile = MagicMock()
        mock_pipeline.start.return_value = mock_profile
        mock_depth_sensor = MagicMock()
        mock_depth_sensor.get_depth_scale.return_value = 0.001
        mock_profile.get_device.return_value.first_depth_sensor.return_value = mock_depth_sensor

        original_avail = rs_mod._RS_AVAILABLE
        original_rs = getattr(rs_mod, "rs", None)
        try:
            rs_mod._RS_AVAILABLE = True
            rs_mod.rs = mock_rs

            driver = rs_mod.RealSenseCameraDriver(config={"fps": 15})
            assert driver.accel_fps == 250
            assert driver.gyro_fps == 400

            driver._init_pipeline()

            # Find all enable_stream calls
            calls = mock_cfg.enable_stream.call_args_list
            accel_calls = [c for c in calls if len(c[0]) >= 3 and c[0][0] == "STREAM_ACCEL"]
            gyro_calls = [c for c in calls if len(c[0]) >= 3 and c[0][0] == "STREAM_GYRO"]

            assert len(accel_calls) == 1, "Expected exactly 1 enable_stream call for accel"
            assert len(gyro_calls) == 1, "Expected exactly 1 enable_stream call for gyro"

            # Check accel rate is 250 Hz (not the obsolete 200 Hz)
            assert accel_calls[0][0][2] == 250
            assert accel_calls[0][0][2] != 200

            # Check gyro rate is 400 Hz
            assert gyro_calls[0][0][2] == 400

            # Verify depth and color stream settings remain untouched
            depth_calls = [c for c in calls if len(c[0]) >= 5 and c[0][0] == "STREAM_DEPTH"]
            color_calls = [c for c in calls if len(c[0]) >= 5 and c[0][0] == "STREAM_COLOR"]
            assert len(depth_calls) == 1
            assert len(color_calls) == 1
            assert depth_calls[0][0][1:5] == (640, 480, "FORMAT_Z16", 15)
            assert color_calls[0][0][1:5] == (640, 480, "FORMAT_BGR8", 15)

        finally:
            rs_mod._RS_AVAILABLE = original_avail
            if original_rs is not None:
                rs_mod.rs = original_rs
            elif hasattr(rs_mod, "rs"):
                delattr(rs_mod, "rs")

