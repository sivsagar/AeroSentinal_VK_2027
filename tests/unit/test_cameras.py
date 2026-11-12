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

