"""
test_obstacle_proximity.py — Comprehensive Unit Tests for Depth Obstacle Proximity Assessment.

Covers all 14 required milestones:
1. Clear depth regions.
2. Caution-level proximity.
3. Warning-level proximity.
4. Critical proximity.
5. Invalid depth regions.
6. Insufficient valid depth coverage.
7. Mixed near and far depth pixels.
8. Multiple occupied regions.
9. Human-associated regions.
10. Temporal smoothing.
11. Hysteresis.
12. Temporary depth dropout.
13. Stale measurement rejection.
14. JSON report generation.
"""

from __future__ import annotations

import json
import time
from typing import Tuple

import numpy as np
import pytest

from aerosentinel.perception.obstacle_detector import (
    ObstacleDetector,
    ObstacleReport,
    ObstacleStatus,
    ProximityLevel,
    RegionAnalysis,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _make_depth(
    h: int = 480,
    w: int = 640,
    left_m: float = 4.0,
    center_m: float = 4.0,
    right_m: float = 4.0,
    left_pct: float = 0.33,
    right_pct: float = 0.33,
) -> np.ndarray:
    """Create a synthetic 1x3 depth image with uniform regional distances."""
    arr = np.zeros((h, w), dtype=np.float32)
    left_end = int(w * left_pct)
    right_start = w - int(w * right_pct)

    arr[:, :left_end] = left_m
    arr[:, left_end:right_start] = center_m
    arr[:, right_start:] = right_m
    return arr


class MockTrack:
    """Lightweight duck-typed tracked person."""

    def __init__(self, track_id: int, cx: int, cy: int, w: int = 60, h: int = 120):
        self.track_id = track_id
        self.center_px = (cx, cy)
        self.bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        self.class_name = "person"


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestObstacleProximityAssessment:
    """Unit test suite for the Obstacle Proximity Assessment subsystem."""

    # 1. Clear depth regions
    def test_clear_depth_regions(self) -> None:
        od = ObstacleDetector()
        depth = _make_depth(left_m=4.5, center_m=5.0, right_m=4.2)
        report = od.analyze(depth)

        assert report.overall_status == ObstacleStatus.CLEAR
        assert report.overall_status == ProximityLevel.CLEAR
        assert report.warning_active is False
        assert report.caution_active is False
        assert report.critical_active is False
        assert report.left.status == ObstacleStatus.CLEAR
        assert report.center.status == ObstacleStatus.CLEAR
        assert report.right.status == ObstacleStatus.CLEAR
        assert report.center.is_valid is True
        assert report.center.median_distance_m == pytest.approx(5.0, abs=0.05)

    # 2. Caution-level proximity
    def test_caution_level_proximity(self) -> None:
        # Caution range: 2.0m < distance <= 3.5m
        od = ObstacleDetector(
            warning_distance_m=2.0,
            caution_distance_m=3.5,
        )
        depth = _make_depth(left_m=4.0, center_m=2.8, right_m=4.0)
        report = od.analyze(depth)

        assert report.overall_status == ObstacleStatus.CAUTION
        assert report.caution_active is True
        assert report.warning_active is False
        assert report.critical_active is False
        assert report.center.status == ObstacleStatus.CAUTION
        assert report.center.median_distance_m == pytest.approx(2.8, abs=0.05)

    # 3. Warning-level proximity
    def test_warning_level_proximity(self) -> None:
        # Warning range: 1.0m < distance <= 2.0m
        od = ObstacleDetector(
            critical_distance_m=1.0,
            warning_distance_m=2.0,
            caution_distance_m=3.5,
        )
        depth = _make_depth(left_m=4.0, center_m=1.6, right_m=4.0)
        report = od.analyze(depth)

        assert report.overall_status == ObstacleStatus.WARNING
        assert report.warning_active is True
        assert report.critical_active is False
        assert report.center.status == ObstacleStatus.WARNING
        assert report.center.median_distance_m == pytest.approx(1.6, abs=0.05)

    # 4. Critical proximity
    def test_critical_proximity(self) -> None:
        # Critical range: distance <= 1.0m
        od = ObstacleDetector(critical_distance_m=1.0)
        depth = _make_depth(left_m=0.6, center_m=3.0, right_m=3.0)
        report = od.analyze(depth)

        assert report.overall_status == ObstacleStatus.CRITICAL
        assert report.critical_active is True
        assert report.warning_active is True
        assert report.left.status == ObstacleStatus.CRITICAL
        assert report.left.median_distance_m == pytest.approx(0.6, abs=0.05)

    # 5. Invalid depth regions (zeros, NaNs, infinities)
    def test_invalid_depth_regions(self) -> None:
        od = ObstacleDetector()

        # All zeros (missing depth returns)
        depth_zeros = np.zeros((480, 640), dtype=np.float32)
        report = od.analyze(depth_zeros)
        assert report.overall_status == ObstacleStatus.UNKNOWN
        assert report.left.is_valid is False
        assert report.center.is_valid is False
        assert report.right.is_valid is False
        assert report.left.status == ObstacleStatus.UNKNOWN

        # Array with NaNs and infs
        depth_nan = np.full((480, 640), np.nan, dtype=np.float32)
        report_nan = od.analyze(depth_nan)
        assert report_nan.overall_status == ObstacleStatus.UNKNOWN
        assert report_nan.center.status == ObstacleStatus.UNKNOWN

    # 6. Insufficient valid depth coverage (< 10% valid pixels)
    def test_insufficient_valid_depth_coverage(self) -> None:
        od = ObstacleDetector(min_valid_coverage_pct=10.0)
        depth = np.zeros((480, 640), dtype=np.float32)

        # Place valid pixels in only ~2% of center region
        depth[200:220, 280:310] = 1.5

        report = od.analyze(depth)
        assert report.center.valid_depth_percent < 10.0
        assert report.center.is_valid is False
        assert report.center.status == ObstacleStatus.UNKNOWN

    # 7. Mixed near and far depth pixels & rejection of isolated noise
    def test_mixed_near_and_far_depth_pixels(self) -> None:
        od = ObstacleDetector(
            critical_distance_m=1.0,
            near_range_occupancy_threshold_pct=15.0,
        )
        depth = np.full((480, 640), 4.5, dtype=np.float32)

        # Substantial near-range obstacle: 25% of center region occupied at 0.7m
        # Exceeds near_range_occupancy_threshold_pct (15%)
        depth[100:300, 240:380] = 0.7

        report = od.analyze(depth)
        assert report.center.near_range_occupancy_percent > 15.0
        assert report.center.percentile_10_distance_m == pytest.approx(0.7, abs=0.05)
        assert report.center.status == ObstacleStatus.CRITICAL

        # Contrast with isolated noise: only 2 pixels at 0.5m
        depth_clean = np.full((480, 640), 4.5, dtype=np.float32)
        depth_clean[200, 300] = 0.5
        depth_clean[201, 300] = 0.5

        od_clean = ObstacleDetector(critical_distance_m=1.0)
        report_clean = od_clean.analyze(depth_clean)
        # Should NOT trigger critical warning on isolated noise
        assert report_clean.center.status == ObstacleStatus.CLEAR

    # 8. Multiple occupied regions
    def test_multiple_occupied_regions(self) -> None:
        od = ObstacleDetector(
            critical_distance_m=1.0,
            warning_distance_m=2.0,
            caution_distance_m=3.5,
        )
        # Left: 1.8m (WARNING), Center: 0.7m (CRITICAL), Right: 4.5m (CLEAR)
        depth = _make_depth(left_m=1.8, center_m=0.7, right_m=4.5)
        report = od.analyze(depth)

        assert report.left.status == ObstacleStatus.WARNING
        assert report.center.status == ObstacleStatus.CRITICAL
        assert report.right.status == ObstacleStatus.CLEAR
        assert report.overall_status == ObstacleStatus.CRITICAL
        assert report.closest_reliable_candidate is not None
        assert report.closest_reliable_candidate["region"] == "center"
        assert report.closest_reliable_candidate["distance_m"] == pytest.approx(0.7, abs=0.05)

    # 9. Human-associated regions
    def test_human_associated_regions(self) -> None:
        od = ObstacleDetector()
        depth = _make_depth(left_m=3.0, center_m=3.0, right_m=3.0)

        # Place track 101 in center (cx=320, cy=240) and track 102 in right (cx=520, cy=240)
        track1 = MockTrack(track_id=101, cx=320, cy=240)
        track2 = MockTrack(track_id=102, cx=520, cy=240)

        report = od.analyze(depth, tracks=[track1, track2])

        assert report.center.human_associated is True
        assert 101 in report.center.human_track_ids
        assert report.right.human_associated is True
        assert 102 in report.right.human_track_ids
        assert report.left.human_associated is False
        assert "center" in report.human_occupied_regions
        assert "right" in report.human_occupied_regions
        assert "left" not in report.human_occupied_regions

    # 10. Temporal smoothing (Exponential Moving Average)
    def test_temporal_smoothing(self) -> None:
        od = ObstacleDetector(temporal_smoothing_alpha=0.30)
        t0 = 1000.0

        # Frame 1: 4.0m
        depth1 = _make_depth(center_m=4.0)
        r1 = od.analyze(depth1, timestamp=t0)
        assert r1.center.median_distance_m == pytest.approx(4.0, abs=0.05)

        # Frame 2: distance steps to 2.0m
        depth2 = _make_depth(center_m=2.0)
        r2 = od.analyze(depth2, timestamp=t0 + 0.1)

        # Smoothed distance: 0.3 * 2.0 + 0.7 * 4.0 = 0.6 + 2.8 = 3.4m
        # Internal state should reflect EMA smoothing
        temp_state = od._temporal_state["center"]
        assert temp_state.smoothed_distance == pytest.approx(3.4, abs=0.05)

    # 11. Hysteresis (requires sustained frames to de-escalate)
    def test_hysteresis(self) -> None:
        od = ObstacleDetector(
            warning_distance_m=2.0,
            caution_distance_m=3.5,
            hysteresis_deescalate_frames=3,
        )
        t = 1000.0

        # Step 1: establish WARNING
        depth_warn = _make_depth(center_m=1.6)
        r1 = od.analyze(depth_warn, timestamp=t)
        assert r1.center.status == ObstacleStatus.WARNING

        # Step 2: feed 1 frame of CLEAR distance (4.0m)
        depth_clear = _make_depth(center_m=4.0)
        r2 = od.analyze(depth_clear, timestamp=t + 0.1)
        # Due to hysteresis, state must stay WARNING on frame 1
        assert r2.center.status == ObstacleStatus.WARNING

        # Step 3: feed second CLEAR frame
        r3 = od.analyze(depth_clear, timestamp=t + 0.2)
        assert r3.center.status == ObstacleStatus.WARNING

        # Step 4: feed third CLEAR frame (satisfies hysteresis_deescalate_frames=3)
        r4 = od.analyze(depth_clear, timestamp=t + 0.3)
        assert r4.center.status == ObstacleStatus.CLEAR

    # 12. Temporary depth dropout handling
    def test_temporary_depth_dropout(self) -> None:
        od = ObstacleDetector(max_dropout_frames=3, stale_timeout_s=2.0)
        t = 1000.0

        # Establish WARNING
        depth_warn = _make_depth(center_m=1.5)
        r1 = od.analyze(depth_warn, timestamp=t)
        assert r1.center.status == ObstacleStatus.WARNING
        assert r1.center.is_valid is True

        # Frame 2: depth dropout (zeros)
        depth_zero = np.zeros((480, 640), dtype=np.float32)
        r2 = od.analyze(depth_zero, timestamp=t + 0.1)

        # Measurement is invalid on this frame, but WARNING state and last valid distance are held
        assert r2.center.is_valid is False
        assert r2.center.status == ObstacleStatus.WARNING
        assert r2.center.last_valid_distance_m == pytest.approx(1.5, abs=0.05)

        # Frame 3: second dropout
        r3 = od.analyze(depth_zero, timestamp=t + 0.2)
        assert r3.center.status == ObstacleStatus.WARNING

    # 13. Stale measurement rejection
    def test_stale_measurement_rejection(self) -> None:
        od = ObstacleDetector(max_dropout_frames=3, stale_timeout_s=2.0)
        t = 1000.0

        # Frame 1: valid WARNING
        od.analyze(_make_depth(center_m=1.5), timestamp=t)

        # Jump forward in time by 3.5s (> stale_timeout_s = 2.0s) with invalid data
        depth_zero = np.zeros((480, 640), dtype=np.float32)
        r_stale = od.analyze(depth_zero, timestamp=t + 3.5)

        # Must be rejected as UNKNOWN with duration cleared
        assert r_stale.center.status == ObstacleStatus.UNKNOWN
        assert r_stale.center.warning_duration_s == 0.0

    # 14. Structured JSON report generation
    def test_json_report_generation(self) -> None:
        od = ObstacleDetector()
        depth = _make_depth(left_m=1.8, center_m=0.8, right_m=4.0)
        track = MockTrack(track_id=7, cx=320, cy=240)

        report = od.analyze(depth, tracks=[track])
        report_dict = report.to_dict()
        json_str = report.to_json(indent=2)

        # Verify JSON is parseable
        parsed = json.loads(json_str)

        assert "timestamp" in parsed
        assert "grid_layout" in parsed
        assert parsed["grid_layout"] == "1x3"
        assert "overall_status" in parsed
        assert parsed["overall_status"] == "CRITICAL"
        assert parsed["warning_active"] is True
        assert parsed["critical_active"] is True
        assert "total_valid_coverage_percent" in parsed
        assert parsed["total_valid_coverage_percent"] > 50.0
        assert "closest_reliable_candidate" in parsed
        assert parsed["closest_reliable_candidate"]["region"] == "center"
        assert "safety_notice" in parsed
        assert "PROVISIONAL" in parsed["safety_notice"]

        # Regional details
        assert "regions" in parsed
        assert "left" in parsed["regions"]
        assert "center" in parsed["regions"]
        assert "right" in parsed["regions"]
        assert parsed["regions"]["center"]["human_associated"] is True
        assert 7 in parsed["regions"]["center"]["human_track_ids"]

    # Extra: 3x3 Grid Layout Test
    def test_3x3_grid_layout(self) -> None:
        od = ObstacleDetector(grid_layout="3x3")
        depth = np.full((480, 640), 4.0, dtype=np.float32)
        # Put critical obstacle in top_left
        depth[100:150, 50:120] = 0.5

        report = od.analyze(depth)
        assert report.grid_layout == "3x3"
        assert len(report.regions) == 9
        assert "top_left" in report.regions
        assert "mid_center" in report.regions
        assert "bottom_right" in report.regions
        assert report.regions["top_left"].status == ObstacleStatus.CRITICAL
        assert report.overall_status == ObstacleStatus.CRITICAL

    # Extra: Visualisation Overlay & Depth Image Generation
    def test_visualisation_methods(self) -> None:
        od = ObstacleDetector()
        depth = _make_depth(left_m=1.5, center_m=0.8, right_m=4.0)
        report = od.analyze(depth)

        rgb = np.full((480, 640, 3), 100, dtype=np.uint8)
        overlay = od.draw_proximity_overlay(rgb, report)
        assert overlay.shape == (480, 640, 3)
        assert overlay.dtype == np.uint8

        depth_vis = od.create_proximity_depth_visualization(depth, report)
        assert depth_vis.shape == (480, 640, 3)
        assert depth_vis.dtype == np.uint8
