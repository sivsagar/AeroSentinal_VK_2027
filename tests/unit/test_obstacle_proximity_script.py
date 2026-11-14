"""
test_obstacle_proximity_script.py — Unit tests for scripts/obstacle_proximity_test.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.obstacle_proximity_test as opt_script
from scripts.obstacle_proximity_test import (
    ObstacleProximityDiagnostic,
    build_parser,
    main,
    parse_args,
)


class TestObstacleProximityScriptCLI:
    """Test argument parsing for the diagnostic script."""

    def test_default_args(self) -> None:
        args = parse_args([])
        assert args.duration == 60.0
        assert args.mock is False
        assert args.grid == "1x3"
        assert args.config_dir == "config"
        assert args.output_dir == "data/detections"
        assert args.diagnostics_dir == "data/diagnostics"
        assert args.save_frames == 5

    def test_custom_args(self) -> None:
        args = parse_args([
            "--duration", "15.0",
            "--mock",
            "--grid", "3x3",
            "--config-dir", "custom_cfg",
            "--output-dir", "custom_out",
            "--diagnostics-dir", "custom_diag",
            "--save-frames", "10",
        ])
        assert args.duration == 15.0
        assert args.mock is True
        assert args.grid == "3x3"
        assert args.config_dir == "custom_cfg"
        assert args.output_dir == "custom_out"
        assert args.diagnostics_dir == "custom_diag"
        assert args.save_frames == 10


class TestObstacleProximityDiagnosticRunner:
    """Test diagnostic runner execution, initialization, and cleanup."""

    def test_mock_initialization_and_cleanup(self, tmp_path: Path) -> None:
        diag = ObstacleProximityDiagnostic(
            duration_s=1.0,
            grid_layout="1x3",
            mock=True,
            output_dir=str(tmp_path / "detections"),
            diagnostics_dir=str(tmp_path / "diagnostics"),
        )
        diag.initialize_subsystems()
        assert diag.camera_mgr is not None
        assert diag.depth_proc is not None
        assert diag.obs_det is not None

        diag.cleanup()
        assert diag.camera_mgr is None

    def test_short_mock_diagnostic_run(self, tmp_path: Path) -> None:
        diag = ObstacleProximityDiagnostic(
            duration_s=0.5,
            grid_layout="1x3",
            mock=True,
            output_dir=str(tmp_path / "detections"),
            diagnostics_dir=str(tmp_path / "diagnostics"),
            save_frame_count=1,
        )
        diag.initialize_subsystems()
        report = diag.run()

        assert "session_metadata" in report
        assert "performance_metrics" in report
        assert "proximity_assessment_summary" in report
        assert "system_metrics" in report
        assert "verdict" in report
        assert report["performance_metrics"]["total_depth_frames_acquired"] > 0

        # Verify JSON report file was written to disk
        diag_files = list((tmp_path / "diagnostics").glob("obstacle_proximity_diagnostic_*.json"))
        assert len(diag_files) == 1
        with open(diag_files[0], "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert saved_json["session_metadata"]["test_name"] == "Obstacle Proximity Assessment Diagnostic"

    def test_main_cli_execution_success(self, tmp_path: Path) -> None:
        exit_code = main([
            "--duration", "0.5",
            "--mock",
            "--output-dir", str(tmp_path / "detections"),
            "--diagnostics-dir", str(tmp_path / "diagnostics"),
            "--save-frames", "1",
        ])
        assert exit_code == 0
