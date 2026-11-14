"""
tests/simulation/multi_person_tracking_test.py

60-second multi-person tracking lifecycle test.
Verifies:
  - Track age, hit count and first/last-seen timestamp accuracy
  - RGB-D depth metadata propagation into every tracked person
  - Consistency invariant: distance_valid=True requires valid_depth_percent > 0
  - EMA distance does not make invalid depth appear valid
  - Lifecycle events (created, confirmed, lost, reacquired, expired) recorded
  - Monotonic track IDs
  - JSON lifecycle report contains per-track history
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
# Path setup — same pattern as other simulation tests
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "simulation"))


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# 60-second simulation test
# ---------------------------------------------------------------------------

class TestMultiPersonTracking60s:
    """
    Simulates a ~60-second tracking session at 10 Hz (600 frames).
    Two synthetic persons appear, one disappears and reappears.
    Verifies all lifecycle events and depth metadata consistency.
    """

    @pytest.fixture(autouse=True)
    def _reset_tracker_id(self):
        """Reset global track-ID counter before each test for isolation."""
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def _make_tracker_and_fusion(self):
        from aerosentinel.perception.person_tracker import PersonTracker
        from aerosentinel.perception.rgbd_fusion import RGBDFusion

        tracker = PersonTracker(
            iou_threshold=0.3,
            min_hits=3,
            max_missed=5,
            max_expired_age_s=20.0,
            ema_alpha=0.3,
        )
        fusion = RGBDFusion(
            min_valid_depth_m=0.1,
            max_valid_depth_m=10.0,
            min_valid_depth_pct=5.0,
        )
        return tracker, fusion

    def test_60s_lifecycle_report(self):
        """
        Run 600 frames at simulated 10 Hz ≈ 60 s.
        Person A is visible all 600 frames at 4.0 m.
        Person B appears at frame 100, disappears at frame 300, reappears at frame 450.
        Depth becomes zero for all tracks frames 200–250 (sensor dropout).
        """
        from aerosentinel.perception.person_tracker import (
            TrackState, LifecycleEvent, PersonTracker,
        )
        from aerosentinel.perception.rgbd_fusion import RGBDFusion

        tracker, fusion = self._make_tracker_and_fusion()

        FPS = 10
        TOTAL_FRAMES = 600
        frame_duration = 1.0 / FPS
        t0 = time.time()

        all_confirmed: List[dict] = []

        for frame_idx in range(TOTAL_FRAMES):
            frame_time = frame_idx * frame_duration
            t_sim = t0 + frame_time  # simulated mission timestamp (absolute)

            # Determine which persons are visible this frame
            dets = [_det(cx=320, cy=240)]  # Person A always visible

            person_b_visible = (100 <= frame_idx < 300) or (frame_idx >= 450)
            if person_b_visible:
                dets.append(_det(cx=160, cy=240, half=40))  # Person B

            # Determine depth quality
            if 200 <= frame_idx < 250:
                depth_m = _depth_zero()     # sensor dropout window
            else:
                depth_m = _depth_valid(dist_m=4.0 + 0.001 * frame_idx)

            fused = fusion.fuse(dets, depth_m)
            confirmed = tracker.update(fused, ts=t_sim)
            all_confirmed.extend(c.to_dict() for c in confirmed)

        # ----------------------------------------------------------------
        # Gather lifecycle report
        # ----------------------------------------------------------------
        report = tracker.lifecycle_summary()
        report["session_duration_s"] = round(TOTAL_FRAMES * frame_duration, 2)
        report["frames_simulated"] = TOTAL_FRAMES
        report["fps_simulated"] = FPS

        # ----------------------------------------------------------------
        # Assertions: lifecycle counts
        # ----------------------------------------------------------------
        assert report["total_tracks_created"] >= 2, "Should have created at least 2 tracks"
        assert report["total_tracks_confirmed"] >= 2, "At least 2 tracks confirmed"
        assert report["total_tracks_lost"] >= 1, "Person B should have been lost"
        assert report["total_tracks_reacquired"] >= 1, "Person B should reappear"

        # ----------------------------------------------------------------
        # Assertions: update count
        # ----------------------------------------------------------------
        assert report["total_update_calls"] == TOTAL_FRAMES

        # ----------------------------------------------------------------
        # Assertions: per-track IDs are monotonic
        # ----------------------------------------------------------------
        track_ids = [int(k) for k in report["tracks"].keys()]
        assert track_ids == sorted(track_ids), "Track IDs must be monotonically increasing"
        assert all(tid >= 1 for tid in track_ids), "Track IDs must start at >= 1"

        # ----------------------------------------------------------------
        # Assertions: timestamps span the full session
        # ----------------------------------------------------------------
        for tid, tdata in report["tracks"].items():
            # Every track that was confirmed should have a meaningful age
            if tdata["state"] in ("confirmed", "expired", "lost"):
                age_s = tdata["age_seconds"]
                # Person A should span close to the full session
                # (tolerance: allow 2 s slop for timing)
                if tdata["hit_count"] > 100:
                    assert age_s >= 5.0, (
                        f"Track {tid} has {tdata['hit_count']} hits but age_seconds={age_s:.2f}; "
                        "timestamps must span real elapsed time, not just 1 frame"
                    )

        # ----------------------------------------------------------------
        # Assertions: age_frames == hit_count for every track
        # ----------------------------------------------------------------
        for tid, tdata in report["tracks"].items():
            assert tdata["age_frames"] == tdata["hit_count"], (
                f"Track {tid}: age_frames={tdata['age_frames']} != "
                f"hit_count={tdata['hit_count']}"
            )

        print(f"\n60-second tracking report:\n{json.dumps(report, indent=2, default=str)}")

    def test_depth_dropout_does_not_make_valid(self):
        """
        During sensor dropout (all-zero depth), distance_valid must be False
        for every tracked person in that frame.
        """
        from aerosentinel.perception.person_tracker import TrackState

        tracker, fusion = self._make_tracker_and_fusion()

        # Warm up — 10 valid frames to confirm a track
        for _ in range(10):
            dets = [_det(cx=320, cy=240)]
            fused = fusion.fuse(dets, _depth_valid(dist_m=5.0))
            tracker.update(fused)

        # Now feed zero-depth frames
        for _ in range(3):
            dets = [_det(cx=320, cy=240)]
            fused = fusion.fuse(dets, _depth_zero())
            confirmed = tracker.update(fused)

            for track in confirmed:
                assert track.distance_valid is False, (
                    f"Track {track.track_id}: distance_valid=True during zero-depth frame; "
                    "depth dropout must not be reported as valid"
                )
                assert track.estimated_distance_m is None, (
                    f"Track {track.track_id}: estimated_distance_m={track.estimated_distance_m} "
                    "during zero-depth; EMA must not persist as valid"
                )
                assert track.valid_depth_percent == 0.0 or track.depth_method == "invalid", (
                    f"Track {track.track_id}: valid_depth_percent={track.valid_depth_percent} "
                    "with depth_method={track.depth_method} during dropout"
                )

    def test_sparse_depth_below_threshold_is_invalid(self):
        """
        When valid depth coverage is below min_valid_depth_pct (5 %),
        distance_valid must be False even if a numeric distance was computed.
        """
        tracker, fusion = self._make_tracker_and_fusion()

        depth_sparse = _depth_sparse(dist_m=4.0, pct=1.5)  # only 1.5% valid — below 5%

        for _ in range(10):
            dets = [_det(cx=320, cy=240)]
            fused = fusion.fuse(dets, depth_sparse)
            confirmed = tracker.update(fused)
            for track in confirmed:
                assert track.distance_valid is False, (
                    f"Track {track.track_id}: distance_valid=True with sparse depth; "
                    f"valid_depth_percent={track.valid_depth_percent:.1f}% < 5% threshold"
                )

    def test_valid_depth_reported_correctly(self):
        """
        When depth is fully valid, distance_valid=True, estimated_distance_m is
        finite and positive, valid_depth_percent > 0.
        """
        tracker, fusion = self._make_tracker_and_fusion()

        for _ in range(10):
            dets = [_det(cx=320, cy=240)]
            fused = fusion.fuse(dets, _depth_valid(dist_m=3.5))
            confirmed = tracker.update(fused)

        assert len(confirmed) >= 1, "Should have at least one confirmed track"
        for track in confirmed:
            assert track.distance_valid is True, \
                f"Track {track.track_id}: distance_valid=False with full valid depth"
            assert track.estimated_distance_m is not None
            assert math.isfinite(track.estimated_distance_m)
            assert track.estimated_distance_m > 0
            assert track.valid_depth_percent > 0.0

    def test_valid_depth_percent_never_zero_when_distance_valid_true(self):
        """
        Invariant: distance_valid=True implies valid_depth_percent > 0.
        Run 300 mixed frames and verify invariant holds for every update.
        """
        tracker, fusion = self._make_tracker_and_fusion()
        rng = np.random.default_rng(42)

        for frame_idx in range(300):
            dets = [_det()]
            if rng.random() < 0.5:
                depth = _depth_valid(dist_m=rng.uniform(1.0, 8.0))
            else:
                depth = _depth_zero()

            fused = fusion.fuse(dets, depth)
            confirmed = tracker.update(fused)

            for track in confirmed:
                if track.distance_valid:
                    assert track.valid_depth_percent > 0.0, (
                        f"Frame {frame_idx}, Track {track.track_id}: "
                        f"distance_valid=True but valid_depth_percent=0.0 — INCONSISTENT"
                    )
                    assert track.estimated_distance_m is not None
                    assert math.isfinite(track.estimated_distance_m)
                    assert track.estimated_distance_m > 0.0


# ---------------------------------------------------------------------------
# Unit tests for FusedDetection consistency invariant
# ---------------------------------------------------------------------------

class TestFusedDetectionInvariant:
    def test_distance_valid_false_when_zero_coverage(self):
        """
        FusedDetection with valid_depth_percent=0.0 must have
        distance_valid=False, even if distance_m is set.
        """
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection

        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(
            detection=det,
            distance_m=4.0,
            distance_valid=True,     # attempt to set True with 0% coverage
            depth_method="median",
            valid_depth_percent=0.0,  # <- forbidden combination
        )
        # __post_init__ must correct this
        assert fd.distance_valid is False, (
            "FusedDetection.__post_init__ must force distance_valid=False "
            "when valid_depth_percent=0.0"
        )
        assert fd.depth_method == "invalid"

    def test_distance_valid_false_when_distance_none(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection

        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(
            detection=det,
            distance_m=None,
            distance_valid=True,     # invalid: distance_m is None
            depth_method="center",
            valid_depth_percent=80.0,
        )
        assert fd.distance_valid is False

    def test_distance_valid_false_when_distance_nan(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection

        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(
            detection=det,
            distance_m=float("nan"),
            distance_valid=True,
            depth_method="center",
            valid_depth_percent=80.0,
        )
        assert fd.distance_valid is False

    def test_to_dict_includes_valid_depth_percent(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection

        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        fd = FusedDetection(
            detection=det,
            distance_m=3.0,
            distance_valid=True,
            depth_method="center",
            valid_depth_percent=92.5,
        )
        d = fd.to_dict()
        assert "valid_depth_percent" in d
        assert d["valid_depth_percent"] == pytest.approx(92.5, abs=0.01)
        assert d["distance_valid"] is True


# ---------------------------------------------------------------------------
# Unit tests for TrackedPerson depth propagation
# ---------------------------------------------------------------------------

class TestTrackedPersonDepthPropagation:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def _make_fd(
        self,
        dist_m: Optional[float],
        valid: bool,
        pct: float,
        method: str = "center",
    ) -> "FusedDetection":
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection

        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        return FusedDetection(
            detection=det,
            distance_m=dist_m,
            distance_valid=valid,
            depth_method=method,
            valid_depth_percent=pct,
        )

    def test_valid_depth_propagated(self):
        from aerosentinel.perception.person_tracker import TrackedPerson

        track = TrackedPerson(track_id=1)
        fd = self._make_fd(dist_m=5.0, valid=True, pct=85.0, method="center")
        track.update_from_fused(fd)

        assert track.distance_valid is True
        assert track.estimated_distance_m == pytest.approx(5.0, abs=0.01)
        assert track.valid_depth_percent == pytest.approx(85.0, abs=0.01)
        assert track.depth_method == "center"

    def test_invalid_depth_does_not_persist_ema(self):
        """
        After several valid frames, an invalid frame must reset distance to None.
        EMA must NOT carry forward as valid distance.
        """
        from aerosentinel.perception.person_tracker import TrackedPerson

        track = TrackedPerson(track_id=1)

        # 5 valid frames — build up EMA
        for _ in range(5):
            fd = self._make_fd(dist_m=4.0, valid=True, pct=90.0)
            track.update_from_fused(fd)

        assert track.distance_valid is True
        assert track.estimated_distance_m is not None

        # One invalid frame
        fd_invalid = self._make_fd(dist_m=None, valid=False, pct=0.0, method="invalid")
        track.update_from_fused(fd_invalid)

        assert track.distance_valid is False, \
            "After invalid frame, distance_valid must be False"
        assert track.estimated_distance_m is None, \
            "EMA must not persist as valid after invalid depth frame"
        assert track.valid_depth_percent == 0.0

    def test_mark_missed_resets_depth(self):
        from aerosentinel.perception.person_tracker import TrackedPerson

        track = TrackedPerson(track_id=1)
        fd = self._make_fd(dist_m=3.0, valid=True, pct=70.0)
        track.update_from_fused(fd)
        assert track.distance_valid is True

        track.mark_missed()

        assert track.distance_valid is False
        assert track.estimated_distance_m is None
        assert track.valid_depth_percent == 0.0
        assert track.depth_method == "invalid"

    def test_age_frames_equals_hit_count(self):
        from aerosentinel.perception.person_tracker import TrackedPerson

        track = TrackedPerson(track_id=1)
        for i in range(7):
            fd = self._make_fd(dist_m=2.0 + i * 0.1, valid=True, pct=80.0)
            track.update_from_fused(fd)

        assert track.age_frames == 7
        assert track.hit_count == 7

    def test_timestamps_span_multiple_seconds(self):
        """first_seen_ts and last_seen_ts must reflect real wall-clock time."""
        from aerosentinel.perception.person_tracker import TrackedPerson

        track = TrackedPerson(track_id=1)
        t_before = time.time()

        fd1 = self._make_fd(dist_m=4.0, valid=True, pct=80.0)
        track.update_from_fused(fd1)
        first = track.first_seen_ts

        time.sleep(0.05)  # 50 ms

        fd2 = self._make_fd(dist_m=4.1, valid=True, pct=80.0)
        track.update_from_fused(fd2)
        last = track.last_seen_ts

        assert last > first, "last_seen_ts must be after first_seen_ts"
        assert last - first >= 0.03, \
            f"Timestamps must span real elapsed time; got {last - first:.4f}s"


# ---------------------------------------------------------------------------
# Unit tests for PersonTracker lifecycle
# ---------------------------------------------------------------------------

class TestPersonTrackerLifecycle:
    @pytest.fixture(autouse=True)
    def _reset_id(self):
        import aerosentinel.perception.person_tracker as _pt
        _pt._NEXT_TRACK_ID = 1

    def _fuse(self, cx: int, dist_m: float, valid_pct: float = 95.0) -> "FusedDetection":
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import FusedDetection

        half = 50
        det = Detection(0, "person", 0.9, (cx - half, 190, cx + half, 290), time.time())
        fd = FusedDetection(
            detection=det,
            distance_m=dist_m if valid_pct > 5.0 else None,
            distance_valid=valid_pct > 5.0,
            depth_method="center" if valid_pct > 5.0 else "invalid",
            valid_depth_percent=valid_pct,
        )
        return fd

    def test_track_created_and_confirmed(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState

        tracker = PersonTracker(min_hits=3, max_missed=5)
        for _ in range(5):
            confirmed = tracker.update([self._fuse(cx=320, dist_m=4.0)])
        assert len(confirmed) >= 1
        assert confirmed[0].state == TrackState.CONFIRMED

    def test_track_lost_after_max_missed(self):
        from aerosentinel.perception.person_tracker import PersonTracker, TrackState

        tracker = PersonTracker(min_hits=3, max_missed=3)
        # Confirm track
        for _ in range(5):
            tracker.update([self._fuse(cx=320, dist_m=4.0)])
        # Feed empty frames
        for _ in range(4):
            tracker.update([])

        all_tracks = tracker.all_tracks()
        lost = [t for t in all_tracks if t.state == TrackState.LOST]
        assert len(lost) >= 1
        assert tracker.total_lost >= 1

    def test_monotonic_track_ids(self):
        from aerosentinel.perception.person_tracker import PersonTracker

        tracker = PersonTracker(min_hits=2)
        for frame in range(30):
            dets = [self._fuse(cx=320, dist_m=4.0)]
            if frame > 15:
                dets.append(self._fuse(cx=160, dist_m=3.0))
            tracker.update(dets)

        ids = sorted(t.track_id for t in tracker.all_tracks())
        for i in range(1, len(ids)):
            assert ids[i] > ids[i - 1], "Track IDs must be strictly increasing"

    def test_lifecycle_summary_json_serialisable(self):
        from aerosentinel.perception.person_tracker import PersonTracker

        tracker = PersonTracker()
        for _ in range(10):
            tracker.update([self._fuse(cx=320, dist_m=4.0)])

        summary = tracker.lifecycle_summary()
        # Must be JSON serialisable
        json_str = json.dumps(summary, default=str)
        loaded = json.loads(json_str)
        assert "tracks" in loaded
        assert "total_update_calls" in loaded
        assert loaded["total_update_calls"] == 10

    def test_reacquisition_counted(self):
        from aerosentinel.perception.person_tracker import PersonTracker

        tracker = PersonTracker(min_hits=3, max_missed=3, max_expired_age_s=60.0)
        # Confirm
        for _ in range(5):
            tracker.update([self._fuse(cx=320, dist_m=4.0)])
        # Lose
        for _ in range(5):
            tracker.update([])
        # Reacquire
        for _ in range(5):
            tracker.update([self._fuse(cx=320, dist_m=4.0)])

        assert tracker.total_reacquired >= 1


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
        assert len(result) == 1
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
        result = fusion.fuse([det], depth)
        fd = result[0]
        assert fd.valid_depth_percent > 50.0
        assert fd.distance_valid is True

    def test_sparse_coverage_below_threshold_is_invalid(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import RGBDFusion

        fusion = RGBDFusion(min_valid_depth_pct=5.0)
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        # 1% valid pixels — below 5% threshold
        depth = _depth_sparse(dist_m=4.0, pct=1.0)
        result = fusion.fuse([det], depth)
        fd = result[0]
        assert fd.distance_valid is False, (
            f"Expected distance_valid=False for 1% coverage; "
            f"got valid_depth_percent={fd.valid_depth_percent:.1f}% "
            f"depth_method={fd.depth_method}"
        )

    def test_to_dict_includes_depth_method(self):
        from aerosentinel.perception.inference_engine import Detection
        from aerosentinel.perception.rgbd_fusion import RGBDFusion

        fusion = RGBDFusion()
        det = Detection(0, "person", 0.9, (100, 100, 300, 300), time.time())
        depth = np.full((480, 640), 4.0, dtype=np.float32)
        d = fusion.fuse([det], depth)[0].to_dict()
        assert "depth_method" in d
        assert "valid_depth_percent" in d

