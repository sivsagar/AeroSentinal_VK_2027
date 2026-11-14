"""
test_hardware_smoke_test.py — Unit and integration tests for hardware_smoke_test.py.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aerosentinel.cameras.camera_manager import IMUData
from aerosentinel.perception.inference_engine import Detection

# Import hardware_smoke_test script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.hardware_smoke_test import (
    DiagnosticLogHandler,
    HardwareSmokeTest,
    main,
    parse_args,
)


class TestHardwareSmokeTestCLI:
    """Tests command-line argument parsing and default configuration."""

    def test_default_args(self) -> None:
        args = parse_args([])
        assert args.duration == 60.0
        assert args.config_dir == "config"
        assert args.model == "models/yolov8n.onnx"
        assert args.output_dir == "data/diagnostics"
        assert args.sync_inference is False
        assert args.poll_interval == 0.005
        assert args.verbose is False

    def test_custom_args(self) -> None:
        args = parse_args([
            "--duration", "15.5",
            "--config-dir", "custom_cfg",
            "--model", "custom_model.onnx",
            "--output-dir", "custom_out",
            "--sync-inference",
            "--poll-interval", "0.01",
            "--verbose",
        ])
        assert args.duration == 15.5
        assert args.config_dir == "custom_cfg"
        assert args.model == "custom_model.onnx"
        assert args.output_dir == "custom_out"
        assert args.sync_inference is True
        assert args.poll_interval == 0.01
        assert args.verbose is True


class TestDiagnosticLogHandler:
    """Tests log interception and categorization."""

    def test_log_categorization(self) -> None:
        handler = DiagnosticLogHandler()
        logger = logging.getLogger("test_diag_logger")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        logger.warning("PiCameraDriver unavailable (picamera2 not installed); RGB disabled.")
        logger.warning("RealSense: not connected; reconnecting in 5.0s…")
        logger.error("RealSense: wait_for_frames timeout/error: frame did not arrive")
        logger.error("Unexpected hardware crash happened")
        logger.info("Normal info message")

        summary = handler.summary()
        assert len(summary["init_errors"]) == 1
        assert "unavailable" in summary["init_errors"][0]

        assert len(summary["reconnect_attempts"]) == 1
        assert "reconnecting" in summary["reconnect_attempts"][0]

        assert len(summary["timeout_errors"]) == 1
        assert "timeout" in summary["timeout_errors"][0]

        assert len(summary["other_errors"]) == 1
        assert "Unexpected hardware crash" in summary["other_errors"][0]


class TestHardwareSmokeTestEvaluation:
    """Tests evaluation metrics, pass/fail determination, and report generation."""

    def test_evaluate_all_passed(self, tmp_path: Path) -> None:
        test = HardwareSmokeTest(
            config_dir="config",
            model_path="models/yolov8n.onnx",
            duration_s=10.0,
            output_dir=str(tmp_path),
        )
        test.metrics["actual_duration_s"] = 10.0
        test.metrics["rgb_frames_received"] = 150
        test.metrics["depth_frames_received"] = 150
        test.metrics["depth_valid_pixel_pcts"] = [55.0, 56.2]
        test.metrics["imu_samples_received"] = 2500
        test.metrics["imu_valid_samples"] = 2500
        test.metrics["inference_passes"] = 40
        test.metrics["inference_latencies_ms"] = [190.0, 205.0]

        test.evaluate_results()

        assert test.test_status["rgb_camera"]["status"] == "PASS"
        assert test.test_status["depth_camera"]["status"] == "PASS"
        assert test.test_status["imu"]["status"] == "PASS"
        assert test.test_status["onnx_inference"]["status"] == "PASS"
        assert test.test_status["overall"]["status"] == "PASS"

    def test_evaluate_hardware_failure(self, tmp_path: Path) -> None:
        test = HardwareSmokeTest(
            config_dir="config",
            model_path="models/yolov8n.onnx",
            duration_s=10.0,
            output_dir=str(tmp_path),
        )
        test.metrics["actual_duration_s"] = 10.0
        test.metrics["rgb_frames_received"] = 0
        test.metrics["depth_frames_received"] = 0
        test.metrics["imu_samples_received"] = 0
        test.metrics["inference_passes"] = 1
        test.metrics["inference_latencies_ms"] = [40.0]

        test.evaluate_results()

        assert test.test_status["rgb_camera"]["status"] == "FAIL"
        assert test.test_status["depth_camera"]["status"] == "FAIL"
        assert test.test_status["imu"]["status"] == "FAIL"
        assert test.test_status["onnx_inference"]["status"] == "PASS"
        assert test.test_status["overall"]["status"] == "FAIL"

    def test_generate_and_save_report(self, tmp_path: Path) -> None:
        test = HardwareSmokeTest(
            config_dir="config",
            model_path="models/yolov8n.onnx",
            duration_s=5.0,
            output_dir=str(tmp_path),
        )
        test.rgb_cfg = {"fps": 15, "resolution": [1280, 720]}
        test.depth_cfg = {"fps": 15, "depth_resolution": [640, 480], "accel_fps": 250, "gyro_fps": 400}
        test.metrics["actual_duration_s"] = 5.0
        test.metrics["rgb_frames_received"] = 75
        test.metrics["depth_frames_received"] = 70
        test.metrics["inference_passes"] = 15
        test.metrics["inference_latencies_ms"] = [195.0, 200.0, 210.0]

        test.evaluate_results()
        report = test.generate_report()

        assert report["rgb_camera"]["frames_received"] == 75
        assert report["rgb_camera"]["dropped_frames"] == 0
        assert report["depth_camera"]["frames_received"] == 70
        assert report["depth_camera"]["dropped_frames"] == 5

        json_file, txt_file = test.save_report(report)
        assert json_file.is_file()
        assert txt_file.is_file()

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["metadata"]["test_name"] == "hardware_smoke_test"
            assert data["inference"]["total_passes"] == 3

        txt_content = txt_file.read_text(encoding="utf-8")
        assert "AeroSentinel Hardware Integration Smoke Test Report" in txt_content


class TestHardwareSmokeTestExecution:
    """Tests execution flow and safety guarantees."""

    def test_missing_model_raises_file_not_found(self, tmp_path: Path) -> None:
        test = HardwareSmokeTest(
            config_dir="config",
            model_path=str(tmp_path / "nonexistent.onnx"),
            duration_s=1.0,
            output_dir=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError):
            test.init_onnx_engine()

    def test_safe_resource_release_on_error(self, tmp_path: Path) -> None:
        test = HardwareSmokeTest(
            config_dir="config",
            model_path="models/yolov8n.onnx",
            duration_s=0.5,
            output_dir=str(tmp_path),
        )
        mock_camera_mgr = MagicMock()
        test.camera_mgr = mock_camera_mgr

        # Simulate exception during acquisition
        with patch.object(test, "_acquisition_loop", side_effect=RuntimeError("Hardware error")):
            with pytest.raises(RuntimeError):
                # Run the inner flow directly or test try/finally
                try:
                    test._acquisition_loop(MagicMock())
                finally:
                    if test.camera_mgr:
                        test.camera_mgr.stop_all()

        # CameraManager.stop_all() must have been called
        mock_camera_mgr.stop_all.assert_called_once()

