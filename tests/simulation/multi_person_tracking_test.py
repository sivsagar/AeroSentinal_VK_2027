"""
tests/simulation/multi_person_tracking_test.py

Multi-person tracking lifecycle and RGB-D metadata regression tests.

Covers:
  1.  Valid distance with nonzero valid-depth percentage.
  2.  Invalid depth with zero valid-depth percentage.
  3.  Stale (last-known) distance preserved after invalid depth frame.
  4.  Correct depth metadata propagation into every TrackedPerson.
  5.  Track age_frames across multiple updates (hit AND missed frames).
  6.  First-seen and last-seen timestamps.
  7.  Track creation and confirmation.
  8.  Temporary detection loss.
  9.  Reacquisition using the original track ID.
  10. Track expiration.
  11. Two-person tracking without ID duplication.
  12. Accurate JSON lifecycle reporting.

Additionally:
  - 60-second simulated lifecycle report.
  - Depth dropout does not produce distance_valid=True.
  - Sparse depth below threshold is invalid.
  - EMA distance does not persist as valid after invalid frame.
  - FusedDetection invariant tests.
  - RGBDFusion valid_depth_percent regression tests.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "simulation"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _det(cx: int = 320, cy: int = 240, half: int = 50, conf: float = 0.88) -> "Detection":
    from aerosentinel.perception.inference_engine import Detection
    return Detection(
        class_id=0,
        class_name="person",
        confidence=conf,
        bbox=(cx - half, cy - half, cx + half, cy + half),
        timestamp=time.time(),
    )


def _depth_valid(h: int = 480, w: int = 640, dist_m: float = 4.0) -> np.ndarray:
    """Full-frame valid depth at dist_m metres."""
    return np.full((h, w), dist_m, dtype=np.float32)


def _depth_zero(h: int = 480, w: int = 640) -> np.ndarray:
    """Full-frame zero (invalid) depth."""
    return np.zeros((h, w), dtype=np.float32)


def _depth_sparse(h: int = 480, w: int = 640, dist_m: float = 3.0, pct: float = 2.0) -> np.ndarray:
    """Depth frame with only `pct` percent valid pixels (below min_valid_depth_pct)."""
    arr = np.zeros((h, w), dtype=np.float32)
    n_valid = int(h * w * pct / 100.0)
    if n_valid > 0:
        indices = np.random.default_rng(99).choice(h * w, n_valid, replace=False)
        arr.ravel()[indices] = dist_m
    return arr


def _make_fd(
    dist_m: Optional[float],
    valid: bool,
    pct: float,
    method: str = "center",
    cx: int = 320,
    cy: int = 240,
    half: int = 50,
) -> "FusedDetection":
    from aerosentinel.perception.inference_engine import Detection
    from aerosentinel.perception.rgbd_fusion import FusedDetection
    det = Detection(0, "person", 0.9, (cx - half, cy - half, cx + half, cy + half), time.time())
    return FusedDetection(
        detection=det,
        distance_m=dist_m,
        distance_valid=valid,
        depth_method=method,
        valid_depth_percent=pct,
    )


def _tracker_and_fusion(
    min_hits: int = 3,
    max_missed: int = 5,
    max_expired_age_s: float = 20.0,
    ema_alpha: float = 0.7,
    min_valid_depth_pct: float = 5.0,
):
    from aerosentinel.perception.person_tracker import PersonTracker
    from aerosentinel.perception.rgbd_fusion import RGBDFusion
    tracker = PersonTracker(
        iou_threshold=0.3,
        min_hits=min_hits,
        max_missed=max_missed,
        max_expired_age_s=max_expired_age_s,
        ema_alpha=ema_alpha,
    )
    fusion = RGBDFusion(
        min_valid_depth_m=0.1,
        max_valid_depth_m=10.0,
        min_valid_depth_pct=min_valid_depth_pct,
    )
    return tracker, fusion


# ---------------------------------------------------------------------------
# 60-second simulation
# ---------------------------------------------------------------------------

class TestMultiPersonTracking60s:
    """
    Simulates a ~60-second tracking session at 10 Hz (600 frames).
    Person A is visible for all 600 frames at ~4.0 m.
    Person B appears at frame 100, disappears at frame 300, reappears at frame 450.
    Sensor dropout (zero depth) occurs frames 200–249.
    """

    @pytest.fixture(autouse=True)
    def _reset_tracker_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_60s_lifecycle_report(self):
        from aerosentinel.perception.person_tracker import TrackState, LifecycleEvent
        from aerosentinel.perception.rgbd_fusion import RGBDFusion

        tracker, fusion = _tracker_and_fusion(min_hits=3, max_missed=5, max_expired_age_s=20.0)

        FPS = 10
        TOTAL_FRAMES = 600
        frame_duration = 1.0 / FPS
        t0 = time.time()
        all_confirmed: List[dict] = []

        for frame_idx in range(TOTAL_FRAMES):
            frame_time = frame_idx * frame_duration
            t_sim = t0 + frame_time

            dets = [_det(cx=320, cy=240)]  # Person A always visible
            person_b_visible = (100 <= frame_idx < 300) or (frame_idx >= 450)
            if person_b_visible:
                dets.append(_det(cx=160, cy=240, half=40))

            depth_m = _depth_zero() if 200 <= frame_idx < 250 else _depth_valid(dist_m=4.0 + 0.001 * frame_idx)

            fused = fusion.fuse(dets, depth_m)
            confirmed = tracker.update(fused, ts=t_sim)
            all_confirmed.extend(c.to_dict() for c in confirmed)

        report = tracker.lifecycle_summary()
        report["session_duration_s"] = round(TOTAL_FRAMES * frame_duration, 2)
        report["frames_simulated"] = TOTAL_FRAMES
        report["fps_simulated"] = FPS

        # Lifecycle counts
        assert report["total_tracks_created"] >= 2
        assert report["total_tracks_confirmed"] >= 2
        assert report["total_tracks_lost"] >= 1, "Person B should have been lost"
        assert report["total_tracks_reacquired"] >= 1, "Person B should reappear"
        assert report["total_update_calls"] == TOTAL_FRAMES
        assert report["peak_concurrent_tracks"] >= 2

        # Monotonic track IDs
        track_ids = [int(k) for k in report["tracks"].keys()]
        assert track_ids == sorted(track_ids)
        assert all(tid >= 1 for tid in track_ids)

        # Timestamps span full session for long-running tracks
        for tid, tdata in report["tracks"].items():
            if tdata["state"] in ("confirmed", "expired", "lost"):
                if tdata["hit_count"] > 100:
                    age_s = tdata["age_seconds"]
                    assert age_s >= 5.0, (
                        f"Track {tid}: {tdata['hit_count']} hits but age_seconds={age_s:.2f}"
                    )

        # age_frames >= hit_count for every track (age_frames includes missed frames too)
        for tid, tdata in report["tracks"].items():
            assert tdata["age_frames"] >= tdata["hit_count"], (
                f"Track {tid}: age_frames={tdata['age_frames']} < hit_count={tdata['hit_count']}"
            )
            # For Person A (600 hits), age_frames should equal hit_count (always matched)
            if tdata["hit_count"] == 600:
                assert tdata["age_frames"] == 600, \
                    f"Track {tid}: always-matched track should have age_frames==600"

        # JSON report includes required depth stats
        assert "total_depth_valid_detections" in report
        assert "total_depth_invalid_detections" in report
        assert "peak_concurrent_tracks" in report

        print(f"\n60-second tracking report:\n{json.dumps(report, indent=2, default=str)}")

    def test_depth_dropout_does_not_make_valid(self):
        """During sensor dropout, distance_valid must be False."""
        tracker, fusion = _tracker_and_fusion()

        for _ in range(10):
            dets = [_det(cx=320, cy=240)]
            fused = fusion.fuse(dets, _depth_valid(dist_m=5.0))
            tracker.update(fused)

        for _ in range(3):
            dets = [_det(cx=320, cy=240)]
            fused = fusion.fuse(dets, _depth_zero())
            confirmed = tracker.update(fused)
            for track in confirmed:
                assert track.distance_valid is False
                assert track.estimated_distance_m is None
                assert track.valid_depth_percent == 0.0 or track.depth_method == "invalid"

    def test_sparse_depth_below_threshold_is_invalid(self):
        """Coverage < 5% threshold → distance_valid=False."""
        tracker, fusion = _tracker_and_fusion()
        depth_sparse = _depth_sparse(dist_m=4.0, pct=1.5)

        for _ in range(10):
            dets = [_det(cx=320, cy=240)]
            fused = fusion.fuse(dets, depth_sparse)
            confirmed = tracker.update(fused)
            for track in confirmed:
                assert track.distance_valid is False

    def test_valid_depth_reported_correctly(self):
        """Full valid depth → distance_valid=True, nonzero coverage."""
        tracker, fusion = _tracker_and_fusion()

        confirmed = []
        for _ in range(10):
            dets = [_det(cx=320, cy=240)]
            fused = fusion.fuse(dets, _depth_valid(dist_m=3.5))
            confirmed = tracker.update(fused)

        assert len(confirmed) >= 1
        for track in confirmed:
            assert track.distance_valid is True
            assert track.estimated_distance_m is not None
            assert math.isfinite(track.estimated_distance_m)
            assert track.estimated_distance_m > 0
            assert track.valid_depth_percent > 0.0

    def test_valid_depth_percent_never_zero_when_distance_valid_true(self):
        """Invariant: distance_valid=True implies valid_depth_percent > 0."""
        tracker, fusion = _tracker_and_fusion()
        rng = np.random.default_rng(42)

        for frame_idx in range(300):
            dets = [_det()]
            depth = _depth_valid(dist_m=rng.uniform(1.0, 8.0)) if rng.random() < 0.5 else _depth_zero()
            fused = fusion.fuse(dets, depth)
            confirmed = tracker.update(fused)
            for track in confirmed:
                if track.distance_valid:
                    assert track.valid_depth_percent > 0.0
                    assert track.estimated_distance_m is not None
                    assert math.isfinite(track.estimated_distance_m)
                    assert track.estimated_distance_m > 0.0


# ---------------------------------------------------------------------------
# Requirement 1: Valid distance with nonzero valid-depth percentage
# ---------------------------------------------------------------------------

class TestReq1_ValidDistanceNonzeroDepth:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_valid_distance_has_nonzero_coverage(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        fd = _make_fd(dist_m=4.5, valid=True, pct=78.0, method="center")
        track.update_from_fused(fd)

        assert track.distance_valid is True
        assert track.valid_depth_percent == pytest.approx(78.0, abs=0.01)
        assert track.estimated_distance_m == pytest.approx(4.5, abs=0.1)
        assert track.depth_method == "center"

    def test_fused_detection_valid_distance_nonzero_pct(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(detection=det, distance_m=3.0, distance_valid=True,
                            depth_method="center", valid_depth_percent=92.5)
        assert fd.distance_valid is True
        assert fd.valid_depth_percent > 0.0
        d = fd.to_dict()
        assert d["valid_depth_percent"] == pytest.approx(92.5, abs=0.01)


# ---------------------------------------------------------------------------
# Requirement 2: Invalid depth with zero valid-depth percentage
# ---------------------------------------------------------------------------

class TestReq2_InvalidDepthZeroCoverage:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_zero_coverage_forces_invalid(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(detection=det, distance_m=4.0, distance_valid=True,
                            depth_method="median", valid_depth_percent=0.0)
        assert fd.distance_valid is False
        assert fd.depth_method == "invalid"

    def test_zero_coverage_propagates_to_track(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        fd = _make_fd(dist_m=None, valid=False, pct=0.0, method="invalid")
        track.update_from_fused(fd)
        assert track.distance_valid is False
        assert track.valid_depth_percent == 0.0
        assert track.estimated_distance_m is None

    def test_rgbd_fusion_zero_depth_gives_zero_percent(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import RGBDFusion
        fusion = RGBDFusion(min_valid_depth_pct=5.0)
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        result = fusion.fuse([det], np.zeros((480, 640), dtype=np.float32))
        fd = result[0]
        assert fd.valid_depth_percent == pytest.approx(0.0, abs=0.01)
        assert fd.distance_valid is False
        assert fd.depth_method == "invalid"


# ---------------------------------------------------------------------------
# Requirement 3: Stale distance preserved as last_known_distance_m
# ---------------------------------------------------------------------------

class TestReq3_StaleDistancePreserved:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_last_known_distance_preserved_after_invalid_frame(self):
        """
        After several valid frames, an invalid frame must:
        - Set distance_valid=False and estimated_distance_m=None.
        - Preserve last_known_distance_m as the previous valid reading.
        """
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1, _ema_alpha=0.7)

        for _ in range(5):
            fd = _make_fd(dist_m=4.0, valid=True, pct=90.0)
            track.update_from_fused(fd)

        last_valid = track.estimated_distance_m
        assert last_valid is not None and last_valid > 0

        fd_invalid = _make_fd(dist_m=None, valid=False, pct=0.0, method="invalid")
        track.update_from_fused(fd_invalid)

        assert track.distance_valid is False
        assert track.estimated_distance_m is None, "EMA must not persist as valid"
        assert track.last_known_distance_m is not None, "last_known_distance_m must be preserved"
        assert track.last_known_distance_m == pytest.approx(last_valid, abs=0.001)

    def test_last_known_distance_preserved_after_mark_missed(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)

        fd = _make_fd(dist_m=3.0, valid=True, pct=70.0)
        track.update_from_fused(fd)
        last_valid = track.last_known_distance_m
        assert last_valid is not None

        track.mark_missed()

        assert track.distance_valid is False
        assert track.estimated_distance_m is None
        assert track.last_known_distance_m == pytest.approx(last_valid, abs=0.001)

    def test_last_known_distance_in_to_dict(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        fd = _make_fd(dist_m=5.5, valid=True, pct=80.0)
        track.update_from_fused(fd)
        fd_inv = _make_fd(dist_m=None, valid=False, pct=0.0, method="invalid")
        track.update_from_fused(fd_inv)

        d = track.to_dict()
        assert "last_known_distance_m" in d
        assert d["last_known_distance_m"] is not None
        assert d["distance_valid"] is False
        assert d["estimated_distance_m"] is None


# ---------------------------------------------------------------------------
# Requirement 4: Correct depth metadata propagation
# ---------------------------------------------------------------------------

class TestReq4_DepthMetadataPropagation:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_valid_depth_propagated_fully(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        fd = _make_fd(dist_m=5.0, valid=True, pct=85.0, method="center")
        track.update_from_fused(fd)
        assert track.distance_valid is True
        assert track.estimated_distance_m == pytest.approx(5.0, abs=0.01)
        assert track.valid_depth_percent == pytest.approx(85.0, abs=0.01)
        assert track.depth_method == "center"

    def test_invalid_depth_does_not_persist_ema(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        for _ in range(5):
            track.update_from_fused(_make_fd(dist_m=4.0, valid=True, pct=90.0))
        assert track.distance_valid is True

        track.update_from_fused(_make_fd(dist_m=None, valid=False, pct=0.0, method="invalid"))
        assert track.distance_valid is False
        assert track.estimated_distance_m is None
        assert track.valid_depth_percent == 0.0

    def test_mark_missed_resets_current_depth(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        track.update_from_fused(_make_fd(dist_m=3.0, valid=True, pct=70.0))
        assert track.distance_valid is True
        track.mark_missed()
        assert track.distance_valid is False
        assert track.estimated_distance_m is None
        assert track.valid_depth_percent == 0.0
        assert track.depth_method == "invalid"

    def test_depth_method_propagated_correctly(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        track.update_from_fused(_make_fd(dist_m=2.5, valid=True, pct=60.0, method="median"))
        assert track.depth_method == "median"


# ---------------------------------------------------------------------------
# Requirement 5: Track age_frames across hit AND missed updates
# ---------------------------------------------------------------------------

class TestReq5_AgeFrames:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_age_frames_increments_on_hit(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        for i in range(5):
            track.update_from_fused(_make_fd(dist_m=2.0, valid=True, pct=80.0))
        assert track.age_frames == 5
        assert track.hit_count == 5

    def test_age_frames_increments_on_miss(self):
        """age_frames must grow on missed frames too; hit_count must not."""
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        track.update_from_fused(_make_fd(dist_m=2.0, valid=True, pct=80.0))
        for _ in range(3):
            track.mark_missed()
        assert track.age_frames == 4   # 1 hit + 3 missed
        assert track.hit_count == 1
        assert track.missed_frames == 3
        assert track.total_missed == 3

    def test_age_frames_via_tracker_update(self):
        """
        Run tracker for 10 frames with a detection present every frame,
        then 5 frames without. age_frames should equal 15 total.
        """
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=3, max_missed=20)
        t0 = time.time()

        for i in range(10):
            fd = _make_fd(dist_m=4.0, valid=True, pct=80.0)
            tracker.update([fd], ts=t0 + i * 0.1)

        for i in range(5):
            tracker.update([], ts=t0 + (10 + i) * 0.1)

        all_t = tracker.all_tracks()
        assert len(all_t) == 1
        track = all_t[0]
        assert track.age_frames == 15
        assert track.hit_count == 10
        assert track.missed_frames == 5
        assert track.total_missed == 5

    def test_age_frames_ge_hit_count_always(self):
        """age_frames >= hit_count is an invariant."""
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=2, max_missed=10)
        t0 = time.time()
        rng = np.random.default_rng(7)
        for i in range(50):
            has_det = rng.random() > 0.3
            fds = [_make_fd(dist_m=3.0, valid=True, pct=70.0)] if has_det else []
            tracker.update(fds, ts=t0 + i * 0.1)

        for t in tracker.all_tracks():
            assert t.age_frames >= t.hit_count, \
                f"Track {t.track_id}: age_frames={t.age_frames} < hit_count={t.hit_count}"


# ---------------------------------------------------------------------------
# Requirement 6: First-seen and last-seen timestamps
# ---------------------------------------------------------------------------

class TestReq6_Timestamps:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_timestamps_span_real_elapsed_time(self):
        from aerosentinel.perception.person_tracker import TrackedPerson
        track = TrackedPerson(track_id=1)
        track.update_from_fused(_make_fd(dist_m=4.0, valid=True, pct=80.0))
        first = track.first_seen_ts
        time.sleep(0.05)
        track.update_from_fused(_make_fd(dist_m=4.1, valid=True, pct=80.0))
        last = track.last_seen_ts
        assert last > first
        assert last - first >= 0.03

    def test_injected_timestamps_respected(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=1)
        t0 = 1_000_000.0
        fd = _make_fd(dist_m=3.0, valid=True, pct=80.0)
        tracker.update([fd], ts=t0)

        t_later = t0 + 60.0
        fd2 = _make_fd(dist_m=3.1, valid=True, pct=80.0)
        tracker.update([fd2], ts=t_later)

        track = tracker.all_tracks()[0]
        assert track.first_seen_ts == pytest.approx(t0, abs=0.001)
        assert track.last_seen_ts == pytest.approx(t_later, abs=0.001)
        assert track.track_duration_s() == pytest.approx(60.0, abs=0.01)

    def test_last_seen_ts_updates_on_hit_only(self):
        """last_seen_ts must NOT advance on a missed frame."""
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=1, max_missed=20)
        t0 = 2_000_000.0
        fd = _make_fd(dist_m=3.0, valid=True, pct=80.0)
        tracker.update([fd], ts=t0)
        track = tracker.all_tracks()[0]
        ts_after_hit = track.last_seen_ts

        for i in range(5):
            tracker.update([], ts=t0 + (i + 1) * 0.1)

        assert track.last_seen_ts == pytest.approx(ts_after_hit, abs=0.001), \
            "last_seen_ts must not advance during missed frames"


# ---------------------------------------------------------------------------
# Requirement 7: Track creation and confirmation
# ---------------------------------------------------------------------------

class TestReq7_CreationConfirmation:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_track_starts_tentative(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState
        tracker = PersonTracker(min_hits=3)
        fd = _make_fd(dist_m=4.0, valid=True, pct=80.0)
        tracker.update([fd])
        track = tracker.all_tracks()[0]
        assert track.state == TrackState.TENTATIVE

    def test_track_confirmed_after_min_hits(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState
        tracker = PersonTracker(min_hits=3)
        for _ in range(5):
            fd = _make_fd(dist_m=4.0, valid=True, pct=80.0)
            tracker.update([fd])
        confirmed = [t for t in tracker.all_tracks() if t.state == TrackState.CONFIRMED]
        assert len(confirmed) == 1
        assert tracker.total_confirmed == 1

    def test_lifecycle_created_event_recorded(self):
        from aerosentinel.perception.person_tracker import PersonTracker, LifecycleEvent
        tracker = PersonTracker(min_hits=3)
        fd = _make_fd(dist_m=4.0, valid=True, pct=80.0)
        tracker.update([fd])
        track = tracker.all_tracks()[0]
        events = [e for _, e, _ in track.lifecycle]
        assert LifecycleEvent.CREATED in events

    def test_lifecycle_confirmed_event_recorded(self):
        from aerosentinel.perception.person_tracker import PersonTracker, LifecycleEvent
        tracker = PersonTracker(min_hits=3)
        for _ in range(5):
            fd = _make_fd(dist_m=4.0, valid=True, pct=80.0)
            tracker.update([fd])
        track = tracker.all_tracks()[0]
        events = [e for _, e, _ in track.lifecycle]
        assert LifecycleEvent.CONFIRMED in events


# ---------------------------------------------------------------------------
# Requirement 8: Temporary detection loss
# ---------------------------------------------------------------------------

class TestReq8_TemporaryLoss:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_track_lost_after_max_missed(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState
        tracker = PersonTracker(min_hits=3, max_missed=3)
        for _ in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        for _ in range(4):
            tracker.update([])
        lost = [t for t in tracker.all_tracks() if t.state == TrackState.LOST]
        assert len(lost) >= 1
        assert tracker.total_lost >= 1

    def test_depth_invalid_during_loss(self):
        """When a track is LOST, its depth fields are invalid."""
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState
        tracker = PersonTracker(min_hits=3, max_missed=3)
        for _ in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        for _ in range(4):
            tracker.update([])
        for track in tracker.all_tracks():
            if track.state == TrackState.LOST:
                assert track.distance_valid is False
                assert track.estimated_distance_m is None

    def test_lost_event_in_lifecycle(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState, LifecycleEvent
        tracker = PersonTracker(min_hits=3, max_missed=3)
        for _ in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        for _ in range(4):
            tracker.update([])
        for track in tracker.all_tracks():
            if track.state == TrackState.LOST:
                events = [e for _, e, _ in track.lifecycle]
                assert LifecycleEvent.LOST in events


# ---------------------------------------------------------------------------
# Requirement 9: Reacquisition using the original track ID
# ---------------------------------------------------------------------------

class TestReq9_Reacquisition:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_same_track_id_after_reacquisition(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState
        tracker = PersonTracker(min_hits=3, max_missed=3, max_expired_age_s=60.0)
        # Confirm
        for _ in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        original_id = tracker.all_tracks()[0].track_id

        # Lose
        for _ in range(4):
            tracker.update([])

        # Reacquire — same bbox position
        for _ in range(3):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])

        all_t = tracker.all_tracks()
        # The original track should have been reacquired (same id)
        ids = [t.track_id for t in all_t]
        assert original_id in ids, "Original track ID must be preserved on reacquisition"
        assert tracker.total_reacquired >= 1

    def test_reacquired_event_in_lifecycle(self):
        from aerosentinel.perception.person_tracker import PersonTracker, LifecycleEvent
        tracker = PersonTracker(min_hits=3, max_missed=3, max_expired_age_s=60.0)
        for _ in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        for _ in range(4):
            tracker.update([])
        for _ in range(3):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])

        for track in tracker.all_tracks():
            events = [e for _, e, _ in track.lifecycle]
            if LifecycleEvent.REACQUIRED in events:
                return  # Pass: reacquisition event found
        pytest.fail("REACQUIRED lifecycle event not recorded")


# ---------------------------------------------------------------------------
# Requirement 10: Track expiration
# ---------------------------------------------------------------------------

class TestReq10_Expiration:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_track_expires_after_max_expired_age_s(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState
        tracker = PersonTracker(min_hits=3, max_missed=3, max_expired_age_s=5.0)
        t0 = 3_000_000.0

        for i in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)], ts=t0 + i * 0.1)

        # Lose the track
        for i in range(4):
            tracker.update([], ts=t0 + 0.5 + i * 0.1)

        # Advance time past max_expired_age_s
        tracker.update([], ts=t0 + 20.0)

        expired = [t for t in tracker.all_tracks() if t.state == TrackState.EXPIRED]
        assert len(expired) >= 1
        assert tracker.total_expired >= 1

    def test_expired_event_in_lifecycle(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState, LifecycleEvent
        tracker = PersonTracker(min_hits=3, max_missed=3, max_expired_age_s=5.0)
        t0 = 4_000_000.0
        for i in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)], ts=t0 + i * 0.1)
        for i in range(4):
            tracker.update([], ts=t0 + 0.5 + i * 0.1)
        tracker.update([], ts=t0 + 20.0)

        for track in tracker.all_tracks():
            if track.state == TrackState.EXPIRED:
                events = [e for _, e, _ in track.lifecycle]
                assert LifecycleEvent.EXPIRED in events
                return
        pytest.fail("EXPIRED lifecycle event not recorded")


# ---------------------------------------------------------------------------
# Requirement 11: Two-person tracking without ID duplication
# ---------------------------------------------------------------------------

class TestReq11_TwoPersonNoIDDuplicate:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_two_persons_get_distinct_ids(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=3, max_missed=10)
        t0 = time.time()

        for i in range(10):
            fd_a = _make_fd(dist_m=4.0, valid=True, pct=80.0, cx=100, half=40)
            fd_b = _make_fd(dist_m=5.0, valid=True, pct=80.0, cx=500, half=40)
            tracker.update([fd_a, fd_b], ts=t0 + i * 0.1)

        all_ids = [t.track_id for t in tracker.all_tracks()]
        assert len(all_ids) == len(set(all_ids)), "Track IDs must be unique"
        assert len(all_ids) == 2, "Should have exactly 2 distinct tracks"

    def test_track_ids_monotonically_increasing(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=2, max_missed=5)
        t0 = time.time()

        # Introduce first person, lose, introduce second, then both
        for i in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0, cx=100)], ts=t0 + i * 0.1)
        for i in range(6):
            tracker.update([], ts=t0 + (5 + i) * 0.1)
        for i in range(5):
            fd_a = _make_fd(dist_m=4.0, valid=True, pct=80.0, cx=100)
            fd_b = _make_fd(dist_m=5.0, valid=True, pct=80.0, cx=500)
            tracker.update([fd_a, fd_b], ts=t0 + (11 + i) * 0.1)

        ids = sorted(t.track_id for t in tracker.all_tracks())
        for i in range(1, len(ids)):
            assert ids[i] > ids[i - 1], "Track IDs must be strictly increasing"

    def test_peak_concurrent_tracks(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=1, max_missed=10)
        t0 = time.time()
        for i in range(5):
            fd_a = _make_fd(dist_m=4.0, valid=True, pct=80.0, cx=100)
            fd_b = _make_fd(dist_m=5.0, valid=True, pct=80.0, cx=500)
            tracker.update([fd_a, fd_b], ts=t0 + i * 0.1)

        assert tracker.peak_concurrent_tracks == 2


# ---------------------------------------------------------------------------
# Requirement 12: Accurate JSON lifecycle reporting
# ---------------------------------------------------------------------------

class TestReq12_JSONLifecycleReport:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def test_lifecycle_summary_json_serialisable(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker()
        for _ in range(10):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        summary = tracker.lifecycle_summary()
        json_str = json.dumps(summary, default=str)
        loaded = json.loads(json_str)
        assert "tracks" in loaded
        assert "total_update_calls" in loaded
        assert loaded["total_update_calls"] == 10

    def test_report_contains_required_depth_fields(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker()
        for _ in range(5):
            tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        summary = tracker.lifecycle_summary()
        assert "total_depth_valid_detections" in summary
        assert "total_depth_invalid_detections" in summary
        assert "peak_concurrent_tracks" in summary

    def test_per_track_dict_contains_all_required_fields(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=1)
        tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        track = tracker.all_tracks()[0]
        d = track.to_dict()
        required = [
            "track_id", "state", "first_seen_ts", "last_seen_ts",
            "age_seconds", "hit_count", "age_frames", "missed_frames",
            "total_missed", "estimated_distance_m", "last_known_distance_m",
            "distance_valid", "valid_depth_percent", "depth_method",
            "latest_bbox", "lifecycle_events",
        ]
        for key in required:
            assert key in d, f"Missing field in to_dict(): '{key}'"

    def test_lifecycle_history_per_track(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker(min_hits=1)
        tracker.update([_make_fd(dist_m=4.0, valid=True, pct=80.0)])
        summary = tracker.lifecycle_summary()
        for tid, tdata in summary["tracks"].items():
            assert "lifecycle_history" in tdata
            assert isinstance(tdata["lifecycle_history"], list)
            # Each event must have ts, event, note keys
            for entry in tdata["lifecycle_history"]:
                assert "ts" in entry
                assert "event" in entry
                assert "note" in entry

    def test_report_total_detections_counted(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        tracker = PersonTracker()
        for _ in range(7):
            tracker.update([
                _make_fd(dist_m=4.0, valid=True, pct=80.0, cx=100),
                _make_fd(dist_m=5.0, valid=True, pct=80.0, cx=500),
            ])
        summary = tracker.lifecycle_summary()
        assert summary["total_detections_processed"] == 14


# ---------------------------------------------------------------------------
# FusedDetection invariant tests
# ---------------------------------------------------------------------------

class TestFusedDetectionInvariant:
    def test_distance_valid_false_when_zero_coverage(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(detection=det, distance_m=4.0, distance_valid=True,
                            depth_method="median", valid_depth_percent=0.0)
        assert fd.distance_valid is False
        assert fd.depth_method == "invalid"

    def test_distance_valid_false_when_distance_none(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(detection=det, distance_m=None, distance_valid=True,
                            depth_method="center", valid_depth_percent=80.0)
        assert fd.distance_valid is False

    def test_distance_valid_false_when_distance_nan(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(detection=det, distance_m=float("nan"), distance_valid=True,
                            depth_method="center", valid_depth_percent=80.0)
        assert fd.distance_valid is False

    def test_to_dict_includes_valid_depth_percent(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(detection=det, distance_m=3.0, distance_valid=True,
                            depth_method="center", valid_depth_percent=92.5)
        d = fd.to_dict()
        assert "valid_depth_percent" in d
        assert d["valid_depth_percent"] == pytest.approx(92.5, abs=0.01)
        assert d["distance_valid"] is True


# ---------------------------------------------------------------------------
# RGBDFusion regression tests
# ---------------------------------------------------------------------------

class TestRGBDFusionValidDepthPercent:
    def test_full_zero_depth_gives_zero_percent(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import RGBDFusion
        fusion = RGBDFusion(min_valid_depth_pct=5.0)
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        result = fusion.fuse([det], np.zeros((480, 640), dtype=np.float32))
        fd = result[0]
        assert fd.valid_depth_percent == pytest.approx(0.0, abs=0.01)
        assert fd.distance_valid is False
        assert fd.depth_method == "invalid"

    def test_full_valid_depth_gives_nonzero_percent(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import RGBDFusion
        fusion = RGBDFusion(min_valid_depth_pct=5.0)
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        depth = np.full((480, 640), 4.0, dtype=np.float32)
        fd = fusion.fuse([det], depth)[0]
        assert fd.valid_depth_percent > 50.0
        assert fd.distance_valid is True

    def test_sparse_coverage_below_threshold_is_invalid(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import RGBDFusion
        fusion = RGBDFusion(min_valid_depth_pct=5.0)
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        depth = _depth_sparse(dist_m=4.0, pct=1.0)
        fd = fusion.fuse([det], depth)[0]
        assert fd.distance_valid is False

    def test_to_dict_includes_depth_method(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import RGBDFusion
        fusion = RGBDFusion()
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        depth = np.full((480, 640), 4.0, dtype=np.float32)
        d = fusion.fuse([det], depth)[0].to_dict()
        assert "depth_method" in d
        assert "valid_depth_percent" in d
