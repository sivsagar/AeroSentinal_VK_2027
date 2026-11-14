"""
test_rgbd_fusion.py — Unit tests for RGB-D Person Detection and Distance Estimation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from aerosentinel.perception.inference_engine import Detection
from aerosentinel.perception.rgbd_fusion import FusedDetection, RGBDFusion


def _create_detection(
    cx: int = 320,
    cy: int = 240,
    w: int = 80,
    h: int = 160,
    conf: float = 0.88,
    class_id: int = 0,
    class_name: str = "person",
) -> Detection:
    """Helper to create a Detection with given center and dimensions."""
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        bbox=(x1, y1, x2, y2),
        timestamp=time.time(),
    )


class TestRGBDPersonDistanceEstimation:
    """Test suite covering Milestone requirements for RGB-D fusion."""

    def test_valid_depth_roi(self) -> None:
        """Requirement 4 & 5: Valid depth ROI gives accurate distance & structured fields."""
        fusion = RGBDFusion(min_valid_depth_m=0.2, max_valid_depth_m=8.0)
        det = _create_detection(cx=300, cy=200, w=100, h=150, conf=0.92)

        # Uniform depth of 3.45 metres
        depth_m = np.full((480, 640), 3.45, dtype=np.float32)
        results = fusion.fuse([det], depth_m)

        assert len(results) == 1
        fd = results[0]

        assert fd.distance_valid is True
        assert fd.distance_m is not None
        assert abs(fd.distance_m - 3.45) < 0.01
        assert fd.depth_method == "median"
        assert fd.center_px == (300, 200)
        assert fd.valid_depth_percent == 100.0
        assert fd.detection_id == 1

        # Check structured dictionary output
        d = fd.to_dict()
        assert d["detection_id"] == 1
        assert d["class_name"] == "person"
        assert d["confidence_score"] == 0.92
        assert d["bbox"] == [250, 125, 350, 275]
        assert d["center_coordinates"] == [300, 200]
        assert d["estimated_distance_m"] == 3.45
        assert d["valid_depth_percent"] == 100.0
        assert d["depth_measurement_validity"] is True
        assert d["depth_method"] == "median"

    def test_zero_depth_pixels(self) -> None:
        """Requirement 10: Zero-depth pixels (sensor shadow/dropout) are rejected."""
        fusion = RGBDFusion()
        det = _create_detection(cx=320, cy=240, w=80, h=120)

        # Depth frame full of zeros
        depth_m = np.zeros((480, 640), dtype=np.float32)
        results = fusion.fuse([det], depth_m)

        assert len(results) == 1
        fd = results[0]
        assert fd.distance_valid is False
        assert fd.distance_m is None
        assert fd.depth_method == "invalid"
        assert fd.valid_depth_percent == 0.0
        assert fd.center_px == (320, 240)

        d = fd.to_dict()
        assert d["depth_measurement_validity"] is False
        assert d["estimated_distance_m"] is None

    def test_invalid_depth_regions(self) -> None:
        """Requirement 10: Out-of-range, NaN, and Inf depths are rejected."""
        fusion = RGBDFusion(min_valid_depth_m=0.5, max_valid_depth_m=5.0)
        det = _create_detection(cx=200, cy=200, w=100, h=100)

        # Construct frame with noise, extreme depths, NaNs, and Infs
        depth_m = np.full((480, 640), np.nan, dtype=np.float32)
        # Put some valid pixels (2.5m) mixed with invalid pixels
        depth_m[175:225, 175:225] = 2.5
        depth_m[175:185, 175:225] = 0.05  # below min_valid_depth_m (0.5m)
        depth_m[185:195, 175:225] = 12.0  # above max_valid_depth_m (5.0m)
        depth_m[195:200, 175:225] = np.inf

        results = fusion.fuse([det], depth_m)
        assert len(results) == 1
        fd = results[0]
        assert fd.distance_valid is True
        assert fd.distance_m is not None
        assert abs(fd.distance_m - 2.5) < 0.05
        # Valid depth percentage should exclude NaNs, Infs, and out-of-range pixels
        assert 0.0 < fd.valid_depth_percent < 100.0

    def test_multiple_detected_persons(self) -> None:
        """Requirement 10: Correctly associates multiple persons with distinct IDs and depths."""
        fusion = RGBDFusion()
        # Person 1 at (150, 200), Person 2 at (450, 300)
        p1 = _create_detection(cx=150, cy=200, w=60, h=120, conf=0.85)
        p2 = _create_detection(cx=450, cy=300, w=70, h=140, conf=0.91)

        depth_m = np.zeros((480, 640), dtype=np.float32)
        # Person 1 area has depth 1.8m
        depth_m[140:260, 120:180] = 1.80
        # Person 2 area has depth 5.25m
        depth_m[230:370, 415:485] = 5.25

        results = fusion.fuse([p1, p2], depth_m)
        assert len(results) == 2

        # Person 1
        assert results[0].detection_id == 1
        assert results[0].distance_valid is True
        assert abs(results[0].distance_m - 1.80) < 0.05
        assert results[0].center_px == (150, 200)

        # Person 2
        assert results[1].detection_id == 2
        assert results[1].distance_valid is True
        assert abs(results[1].distance_m - 5.25) < 0.05
        assert results[1].center_px == (450, 300)

    def test_bounding_box_boundary_handling(self) -> None:
        """Requirement 10: Handles bboxes extending beyond frame boundaries without crash."""
        fusion = RGBDFusion()
        depth_m = np.full((480, 640), 3.0, dtype=np.float32)

        # Bbox protruding out of left and top: [-30, -50, 100, 120]
        det_top_left = Detection(
            class_id=0,
            class_name="person",
            confidence=0.80,
            bbox=(-30, -50, 100, 120),
            timestamp=time.time(),
        )

        # Bbox protruding out of right and bottom: [600, 400, 750, 550]
        det_bottom_right = Detection(
            class_id=0,
            class_name="person",
            confidence=0.82,
            bbox=(600, 400, 750, 550),
            timestamp=time.time(),
        )

        # Bbox completely outside frame: [800, 800, 900, 900]
        det_outside = Detection(
            class_id=0,
            class_name="person",
            confidence=0.75,
            bbox=(800, 800, 900, 900),
            timestamp=time.time(),
        )

        results = fusion.fuse([det_top_left, det_bottom_right, det_outside], depth_m)
        assert len(results) == 3

        # Clamped bboxes inside image still acquire valid depth
        assert results[0].distance_valid is True
        assert results[1].distance_valid is True
        # Completely outside bbox returns invalid
        assert results[2].distance_valid is False
        assert results[2].depth_method == "invalid"

    def test_median_distance_estimation(self) -> None:
        """Requirement 10: Median filter rejects background and noise outliers."""
        fusion = RGBDFusion(median_roi_fraction=0.6, min_valid_depth_percent=5.0)
        det = _create_detection(cx=320, cy=240, w=100, h=100)

        # Central ROI for 100x100 bbox with 0.6 fraction is:
        # margin = 100 * (1 - 0.6) / 2 = 20 -> 60x60 region [210:270, 290:350]
        depth_m = np.zeros((480, 640), dtype=np.float32)

        # In central region, fill mostly 2.5m, with 20% background outliers (8.5m)
        roi_patch = np.full((60, 60), 2.5, dtype=np.float32)
        roi_patch[:12, :] = 8.5  # 20% background outliers
        depth_m[210:270, 290:350] = roi_patch

        results = fusion.fuse([det], depth_m)

        assert len(results) == 1
        fd = results[0]
        assert fd.distance_valid is True
        # True median of the patch is 2.5m, unaffected by the 8.5m outlier
        assert abs(fd.distance_m - 2.5) < 0.05
        assert fd.valid_depth_percent == 100.0

    def test_missing_or_unreliable_depth(self) -> None:
        """Requirement 10: Insufficient valid depth pixels flagged as unreliable."""
        fusion = RGBDFusion(min_valid_depth_percent=20.0)
        det = _create_detection(cx=320, cy=240, w=100, h=100)

        # Sparse frame with only ~2% non-zero pixels
        depth_m = np.zeros((480, 640), dtype=np.float32)
        depth_m[240, 320] = 3.0  # single valid pixel

        results = fusion.fuse([det], depth_m)
        assert len(results) == 1
        fd = results[0]

        # Valid depth percentage < 20% threshold -> unreliable
        assert fd.distance_valid is False
        assert fd.depth_method == "unreliable"
        assert fd.valid_depth_percent < 20.0

    def test_raw_uint16_depth_conversion(self) -> None:
        """Verifies uint16 mm depth array (RealSense native) is converted to metres."""
        fusion = RGBDFusion()
        det = _create_detection(cx=320, cy=240, w=80, h=100)

        # 2850 mm in uint16
        depth_mm = np.full((480, 640), 2850, dtype=np.uint16)
        results = fusion.fuse([det], depth_mm)

        assert len(results) == 1
        fd = results[0]
        assert fd.distance_valid is True
        # 2850 mm -> 2.85 m
        assert abs(fd.distance_m - 2.85) < 0.01

    def test_draw_and_save_annotated_frame(self, tmp_path: Path) -> None:
        """Requirement 6 & 7: Draw and save visual verification frame."""
        fusion = RGBDFusion()
        det = _create_detection(cx=320, cy=240, w=100, h=150, conf=0.95)
        depth_m = np.full((480, 640), 3.12, dtype=np.float32)
        fused = fusion.fuse([det], depth_m)

        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = fusion.draw_fused_detections(rgb_frame, fused, draw_center=True)

        assert annotated.shape == (480, 640, 3)
        # Verify drawing modified the frame
        assert np.any(annotated > 0)

        out_path = tmp_path / "test_visual_verification.jpg"
        saved = fusion.save_annotated_frame(rgb_frame, out_path, fused_detections=fused)
        assert saved.is_file()
        assert saved.stat().st_size > 0
