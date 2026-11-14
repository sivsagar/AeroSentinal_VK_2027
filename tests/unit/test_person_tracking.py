"""
test_person_tracking.py — Unit tests for Multi-Person Tracking and RGB-D Distance Association.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from aerosentinel.perception.inference_engine import Detection
from aerosentinel.perception.person_tracker import (
    PersonTracker,
    TrackedPerson,
    compute_iou,
    compute_normalized_center_dist,
)
from aerosentinel.perception.rgbd_fusion import FusedDetection


def _create_fused_detection(
    cx: int,
    cy: int,
    w: int = 80,
    h: int = 160,
    distance_m: float = 3.0,
    distance_valid: bool = True,
    conf: float = 0.90,
    depth_method: str = "median",
    valid_depth_pct: float = 95.0,
    det_id: int = 1,
) -> FusedDetection:
    """Helper to create a FusedDetection with explicit geometry and depth."""
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2
    det = Detection(
        class_id=0,
        class_name="person",
        confidence=conf,
        bbox=(x1, y1, x2, y2),
        timestamp=time.monotonic(),
    )
    return FusedDetection(
        detection=det,
        distance_m=distance_m,
        distance_valid=distance_valid,
        depth_method=depth_method,
        center_px=(cx, cy),
        valid_depth_percent=valid_depth_pct,
        detection_id=det_id,
        position_valid=False,
    )


class TestGeometryHelpers:
    """Tests bounding box IoU and normalized center distance helpers."""

    def test_compute_iou_identical(self) -> None:
        box = (100, 100, 200, 200)
        assert abs(compute_iou(box, box) - 1.0) < 1e-5

    def test_compute_iou_disjoint(self) -> None:
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert compute_iou(box1, box2) == 0.0

    def test_compute_iou_partial_overlap(self) -> None:
        box1 = (0, 0, 100, 100)      # area 10,000
        box2 = (50, 0, 150, 100)     # area 10,000, inter 5000, union 15000
        iou = compute_iou(box1, box2)
        assert abs(iou - (5000.0 / 15000.0)) < 1e-4

    def test_compute_normalized_center_dist(self) -> None:
        box1 = (100, 100, 200, 200)  # center (150, 150), scale 100x100
        box2 = (130, 140, 230, 240)  # center (180, 190), dx=30, dy=40 -> dist=50 -> norm=0.5
        norm_dist = compute_normalized_center_dist(box1, box2)
        assert abs(norm_dist - 0.5) < 1e-3


class TestPersonTrackerLifecycle:
    """Test suite covering Multi-Person Tracking requirements."""

    def test_track_multiple_people_across_frames(self) -> None:
        """Requirement 1 & 2: Track multiple people across successive frames with persistent IDs."""
        tracker = PersonTracker(iou_threshold=0.3, min_hits=1)

        # Frame 1: Three persons detected at distinct locations
        f1_p1 = _create_fused_detection(cx=150, cy=200, distance_m=2.20)
        f1_p2 = _create_fused_detection(cx=350, cy=220, distance_m=3.50)
        f1_p3 = _create_fused_detection(cx=520, cy=250, distance_m=4.80)

        tracks_f1 = tracker.update([f1_p1, f1_p2, f1_p3], timestamp=1.0)
        assert len(tracks_f1) == 3

        ids_f1 = {t.track_id for t in tracks_f1}
        assert len(ids_f1) == 3  # All distinct IDs

        # Frame 2: Persons move slightly
        f2_p1 = _create_fused_detection(cx=155, cy=202, distance_m=2.22)
        f2_p2 = _create_fused_detection(cx=352, cy=224, distance_m=3.48)
        f2_p3 = _create_fused_detection(cx=518, cy=248, distance_m=4.82)

        tracks_f2 = tracker.update([f2_p1, f2_p2, f2_p3], timestamp=1.3)
        assert len(tracks_f2) == 3

        ids_f2 = {t.track_id for t in tracks_f2}
        assert ids_f1 == ids_f2  # Persistent IDs preserved!

        # Check track details
        t1 = next(t for t in tracks_f2 if t.center_px == (155, 202))
        assert t1.hits == 2
        assert t1.age == 2
        assert t1.time_since_update == 0
        assert t1.state == "confirmed"
        assert abs(t1.distance_m - 2.22) < 0.05

    def test_missed_detections_and_coasting(self) -> None:
        """Requirement 4 & 5: Handle temporary detection loss without immediate track deletion."""
        tracker = PersonTracker(max_lost_frames=5, min_hits=1)

        # Frame 1: Person detected
        p = _create_fused_detection(cx=200, cy=200, distance_m=2.50)
        tracks_f1 = tracker.update([p], timestamp=1.0)
        assert len(tracks_f1) == 1
        tid = tracks_f1[0].track_id
        assert tracks_f1[0].state == "confirmed"

        # Frame 2: Missed detection (empty detection list)
        tracks_f2 = tracker.update([], timestamp=1.3)
        assert len(tracks_f2) == 1  # Track NOT deleted!
        t = tracks_f2[0]
        assert t.track_id == tid
        assert t.state == "lost"
        assert t.time_since_update == 1
        assert t.age == 2
        assert t.distance_m == 2.50  # Distance preserved

        # Frame 3: Another missed frame
        tracks_f3 = tracker.update([], timestamp=1.6)
        assert len(tracks_f3) == 1
        assert tracks_f3[0].time_since_update == 2
        assert tracks_f3[0].state == "lost"

    def test_reacquisition_after_temporary_loss(self) -> None:
        """Requirement 4 & 9: Reacquire track with original ID after temporary detection loss."""
        tracker = PersonTracker(max_lost_frames=5, min_hits=1)

        # Frame 1: Person detected (Track #1 created)
        p1 = _create_fused_detection(cx=250, cy=250, distance_m=3.0)
        t_f1 = tracker.update([p1], timestamp=1.0)
        original_id = t_f1[0].track_id

        # Frames 2 & 3: Person missed (occlusion)
        tracker.update([], timestamp=1.3)
        t_f3 = tracker.update([], timestamp=1.6)
        assert t_f3[0].state == "lost"
        assert t_f3[0].time_since_update == 2

        # Frame 4: Person reappears near last known location
        p_reacquired = _create_fused_detection(cx=255, cy=252, distance_m=3.05)
        t_f4 = tracker.update([p_reacquired], timestamp=1.9)

        assert len(t_f4) == 1
        reacquired_track = t_f4[0]
        assert reacquired_track.track_id == original_id  # Preserved original ID!
        assert reacquired_track.state == "confirmed"
        assert reacquired_track.time_since_update == 0
        assert reacquired_track.hits == 2

    def test_track_expiration(self) -> None:
        """Requirement 9: Tracks expire and are cleaned up after max_lost_frames."""
        tracker = PersonTracker(max_lost_frames=3, min_hits=1)

        # Frame 1: Person detected
        p = _create_fused_detection(cx=300, cy=300)
        tracker.update([p], timestamp=1.0)
        assert len(tracker.tracks) == 1

        # Miss 3 frames (max_lost_frames = 3)
        tracker.update([], timestamp=1.3)
        tracker.update([], timestamp=1.6)
        tracker.update([], timestamp=1.9)
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].state == "lost"

        # Frame 5: Exceeds max_lost_frames (time_since_update reaches 4) -> expired
        tracks = tracker.update([], timestamp=2.2)
        assert len(tracks) == 0  # Expired and deleted!

    def test_prevent_duplicate_tracking_ids(self) -> None:
        """Requirement 6: Verify track IDs are unique, monotonic, and deduplicated."""
        tracker = PersonTracker(dedup_iou_threshold=0.80)

        # 3 distinct detections
        dets = [
            _create_fused_detection(cx=100, cy=100),
            _create_fused_detection(cx=300, cy=300),
            _create_fused_detection(cx=500, cy=500),
        ]
        tracks = tracker.update(dets)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids))
        assert ids == [1, 2, 3]

        # Duplicate detection (almost identical bounding box) should be deduplicated
        dup1 = _create_fused_detection(cx=100, cy=100, w=80, h=160)
        dup2 = _create_fused_detection(cx=102, cy=101, w=80, h=160)
        dedup_tracks = tracker.update([dup1, dup2])
        # Only one track should be retained for this spatial location
        dup_locations = [t for t in dedup_tracks if abs(t.center_px[0] - 100) < 20]
        assert len(dup_locations) == 1

    def test_rgbd_distance_ema_smoothing(self) -> None:
        """Requirement 3 & 5: Associate distance and smooth via Exponential Moving Average."""
        tracker = PersonTracker(distance_ema_alpha=0.6)

        # Frame 1: distance 3.0m
        p1 = _create_fused_detection(cx=200, cy=200, distance_m=3.00)
        t1 = tracker.update([p1], timestamp=1.0)[0]
        assert abs(t1.distance_m - 3.00) < 0.01

        # Frame 2: sensor reading jumps slightly to 3.40m
        # EMA: 0.6 * 3.40 + 0.4 * 3.00 = 2.04 + 1.20 = 3.24m
        p2 = _create_fused_detection(cx=200, cy=200, distance_m=3.40)
        t2 = tracker.update([p2], timestamp=1.3)[0]
        assert abs(t2.distance_m - 3.24) < 0.02
        assert t2.distance_valid is True

        # Frame 3: sensor reading invalid
        p3 = _create_fused_detection(cx=200, cy=200, distance_m=None, distance_valid=False)
        t3 = tracker.update([p3], timestamp=1.6)[0]
        # Distance validity becomes False, but previous distance retained as best estimate
        assert t3.distance_valid is False
        assert t3.distance_m is not None

    def test_export_json_structured_data(self, tmp_path: Path) -> None:
        """Requirement 8: Export structured tracking payload conforming to JSON schema."""
        tracker = PersonTracker()
        p1 = _create_fused_detection(cx=150, cy=200, distance_m=2.75, conf=0.93)
        p2 = _create_fused_detection(cx=450, cy=250, distance_m=4.10, conf=0.87)

        tracks = tracker.update([p1, p2], timestamp=10.5)

        out_json_file = tmp_path / "tracks_export.json"
        json_str = tracker.export_json(tracks, filepath=out_json_file)

        assert out_json_file.is_file()
        payload = json.loads(json_str)

        assert "timestamp" in payload
        assert payload["frame_index"] == 1
        assert payload["active_tracks_count"] == 2
        assert len(payload["tracks"]) == 2

        t1 = payload["tracks"][0]
        required_keys = {
            "track_id", "class_name", "state", "confidence", "bbox",
            "center_coordinates", "estimated_distance_m", "distance_valid",
            "valid_depth_percent", "depth_method", "age_frames", "hits",
            "time_since_update_frames", "first_seen_timestamp", "last_seen_timestamp",
        }
        assert required_keys.issubset(t1.keys())
        assert t1["track_id"] == 1
        assert t1["center_coordinates"] == [150, 200]
        assert t1["estimated_distance_m"] == 2.75

    def test_draw_tracks_annotations(self) -> None:
        """Requirement 7: Draw persistent tracking IDs and distances on frame."""
        tracker = PersonTracker()
        p = _create_fused_detection(cx=200, cy=200, distance_m=3.15, conf=0.91)
        tracks = tracker.update([p])

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = tracker.draw_tracks(dummy_frame, tracks, draw_trajectory=True)

        assert annotated.shape == (480, 640, 3)
        assert np.any(annotated > 0)  # Drawing modified pixels!

