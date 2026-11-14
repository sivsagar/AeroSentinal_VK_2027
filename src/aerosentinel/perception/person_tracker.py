"""
person_tracker.py — Multi-person tracking with depth metadata for AeroSentinel.

Design rules
------------
* Monotonic, never-reused track IDs (global counter).
* age_frames increments on EVERY tracker update call (hit or miss), so it
  correctly represents the total number of frames the track has been alive,
  not just the number of matched detections.
* hit_count counts only successful detection matches.
* missed_frames counts consecutive frames without a match (resets on hit).
* first_seen_ts and last_seen_ts use wall-clock / injected timestamps so
  that elapsed time spans the actual mission duration.
* valid_depth_percent is ALWAYS propagated directly from FusedDetection —
  it reflects the CURRENT frame only and is never stale.
* estimated_distance_m is the EMA-smoothed distance; set only when the
  current frame passes depth validation.
* last_known_distance_m preserves the most recent VALID distance for
  display/logging even after depth becomes temporarily invalid.
* EMA distance is updated ONLY when distance_valid=True on the new
  measurement; a stale EMA value is NEVER reported as distance_valid=True
  when the current measurement has distance_valid=False.
* Track is CONFIRMED once hit_count >= min_hits.
* Track is LOST once missed_frames > max_missed.
* Lifecycle events (created, confirmed, lost, reacquired, expired) are
  recorded per-track and aggregated globally.

Safety
------
No flight commands are issued from this module.  It is perception-only.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .inference_engine import Detection
from .rgbd_fusion import FusedDetection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global monotonic track-ID counter
# ---------------------------------------------------------------------------
_NEXT_TRACK_ID: int = 1
_TRACK_ID_LOCK = None  # set to threading.Lock on first use


def _alloc_track_id() -> int:
    """Thread-safe monotonic track-ID allocation."""
    global _NEXT_TRACK_ID, _TRACK_ID_LOCK
    if _TRACK_ID_LOCK is None:
        import threading
        _TRACK_ID_LOCK = threading.Lock()
    with _TRACK_ID_LOCK:
        tid = _NEXT_TRACK_ID
        _NEXT_TRACK_ID += 1
    return tid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TrackState(Enum):
    TENTATIVE = "tentative"      # < min_hits detections
    CONFIRMED = "confirmed"      # >= min_hits detections
    LOST = "lost"                # missed too many frames
    EXPIRED = "expired"          # removed from active set


class LifecycleEvent(Enum):
    CREATED = "created"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REACQUIRED = "reacquired"
    EXPIRED = "expired"
    DEPTH_VALID = "depth_valid"
    DEPTH_INVALID = "depth_invalid"


# ---------------------------------------------------------------------------
# TrackedPerson
# ---------------------------------------------------------------------------

@dataclass
class TrackedPerson:
    """
    One tracked person across multiple frames.

    Depth attributes
    ----------------
    estimated_distance_m : Optional[float]
        EMA-smoothed distance — set only when current frame distance_valid=True.
        NEVER carries forward a previous valid EMA when current frame is invalid.
    last_known_distance_m : Optional[float]
        The most recent VALID distance observed for this track.
        Preserved across invalid frames so the last good reading can be
        displayed/logged without implying the current measurement is valid.
    distance_valid : bool
        Reflects the CURRENT frame's FusedDetection.distance_valid.
        False when no detection matched (missed frame) or depth is invalid.
    valid_depth_percent : float
        CURRENT frame's coverage percentage; never stale from a previous frame.
    depth_method : str
        CURRENT frame's depth_method.

    Age / hit accounting
    --------------------
    age_frames : int
        Incremented on EVERY tracker update (hit or miss) since track creation.
        Represents the total frames this track has been alive in the tracker.
    hit_count : int
        Incremented only when a detection is successfully matched to this track.
    missed_frames : int
        Consecutive frames without a matched detection. Resets to 0 on a hit.
    total_missed : int
        Cumulative missed frames since track creation (never resets).
    """

    track_id: int
    state: TrackState = TrackState.TENTATIVE

    # Timing
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)

    # Hit / miss accounting
    hit_count: int = 0           # detections matched so far
    age_frames: int = 0          # total tracker updates since creation
    missed_frames: int = 0       # consecutive missed frames
    total_missed: int = 0        # cumulative missed frames (never resets)

    # Latest detection
    latest_detection: Optional[Detection] = None
    latest_bbox: Optional[Tuple[int, int, int, int]] = None

    # Depth metadata — ALWAYS from current frame (or reset on miss)
    estimated_distance_m: Optional[float] = None
    last_known_distance_m: Optional[float] = None   # last VALID reading
    distance_valid: bool = False
    valid_depth_percent: float = 0.0
    depth_method: str = "invalid"

    # EMA state (internal; not exposed as "valid" unless current frame is valid)
    _ema_distance: Optional[float] = field(default=None, repr=False)
    _ema_alpha: float = field(default=0.7, repr=False)

    # Lifecycle history
    lifecycle: List[Tuple[float, LifecycleEvent, str]] = field(default_factory=list)

    def _record_event(self, evt: LifecycleEvent, note: str = "") -> None:
        self.lifecycle.append((time.time(), evt, note))
        logger.debug("Track %d: %s %s", self.track_id, evt.value, note)

    def update_from_fused(
        self, fd: FusedDetection, ts: Optional[float] = None
    ) -> None:
        """
        Update this track with a new FusedDetection (matched frame).

        Parameters
        ----------
        fd : FusedDetection
        ts : float or None
            Wall-clock (or simulated) timestamp for this update.
            Defaults to ``time.time()`` when None.

        Rules
        -----
        * age_frames increments by 1 (also happens in mark_missed).
        * hit_count increments by 1 (only on a detection match).
        * last_seen_ts is set to ``ts`` (or ``time.time()``).
        * missed_frames resets to 0.
        * valid_depth_percent, depth_method always taken from fd (current frame).
        * EMA distance updated ONLY when fd.distance_valid=True.
        * distance_valid reflects fd.distance_valid — never coerced to True.
        * last_known_distance_m updated only when distance_valid=True.
        """
        was_lost = self.state == TrackState.LOST

        self.age_frames += 1
        self.hit_count += 1
        self.missed_frames = 0
        self.last_seen_ts = ts if ts is not None else time.time()
        self.latest_detection = fd.detection
        self.latest_bbox = fd.detection.bbox

        # --- Always propagate current-frame depth metadata ---
        self.valid_depth_percent = fd.valid_depth_percent
        self.depth_method = fd.depth_method
        self.distance_valid = fd.distance_valid

        if fd.distance_valid and fd.distance_m is not None and math.isfinite(fd.distance_m):
            # Update EMA
            if self._ema_distance is None:
                self._ema_distance = fd.distance_m
            else:
                self._ema_distance = (
                    self._ema_alpha * fd.distance_m
                    + (1.0 - self._ema_alpha) * self._ema_distance
                )
            self.estimated_distance_m = self._ema_distance
            self.last_known_distance_m = self._ema_distance   # preserve last valid
            self._record_event(
                LifecycleEvent.DEPTH_VALID,
                f"dist={fd.distance_m:.2f}m ema={self._ema_distance:.2f}m "
                f"cov={fd.valid_depth_percent:.1f}%",
            )
        else:
            # Current depth invalid — do NOT use EMA as valid distance.
            # last_known_distance_m is preserved for display purposes.
            self.estimated_distance_m = None
            self._record_event(
                LifecycleEvent.DEPTH_INVALID,
                f"method={fd.depth_method} cov={fd.valid_depth_percent:.1f}%",
            )

        if was_lost:
            self._record_event(LifecycleEvent.REACQUIRED)

    def mark_missed(self, ts: Optional[float] = None) -> None:
        """
        Call when no detection matched this track in a frame.

        age_frames is incremented here too, so it reflects the total frames
        the track has been alive (hit frames + missed frames).
        Depth metadata is reset to invalid; last_known_distance_m is preserved.
        """
        self.age_frames += 1
        self.missed_frames += 1
        self.total_missed += 1

        # Depth becomes stale for this frame — reset current-frame fields.
        # last_known_distance_m intentionally NOT cleared (available for display).
        self.distance_valid = False
        self.estimated_distance_m = None
        self.valid_depth_percent = 0.0
        self.depth_method = "invalid"

    def track_duration_s(self, now: Optional[float] = None) -> float:
        """Wall-clock seconds from first_seen_ts to now (or last_seen_ts)."""
        end = now if now is not None else self.last_seen_ts
        return max(0.0, end - self.first_seen_ts)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "age_seconds": round(self.last_seen_ts - self.first_seen_ts, 3),
            "hit_count": self.hit_count,
            "age_frames": self.age_frames,
            "missed_frames": self.missed_frames,
            "total_missed": self.total_missed,
            "estimated_distance_m": (
                round(self.estimated_distance_m, 3)
                if self.estimated_distance_m is not None else None
            ),
            "last_known_distance_m": (
                round(self.last_known_distance_m, 3)
                if self.last_known_distance_m is not None else None
            ),
            "distance_valid": self.distance_valid,
            "valid_depth_percent": round(self.valid_depth_percent, 2),
            "depth_method": self.depth_method,
            "latest_bbox": list(self.latest_bbox) if self.latest_bbox else None,
            "lifecycle_events": len(self.lifecycle),
        }

    def lifecycle_history(self) -> List[dict]:
        return [
            {"ts": ts, "event": evt.value, "note": note}
            for ts, evt, note in self.lifecycle
        ]


# ---------------------------------------------------------------------------
# PersonTracker
# ---------------------------------------------------------------------------

class PersonTracker:
    """
    IoU-based multi-person tracker.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to associate a detection with an existing track (default 0.3).
    min_hits : int
        Number of consecutive detections required to confirm a track (default 3).
    max_missed : int
        Maximum consecutive missed frames before marking track LOST (default 15).
    max_expired_age_s : float
        Seconds after LOST before a track is fully expired and removed (default 20.0).
    ema_alpha : float
        EMA smoothing factor for distance (default 0.7).  Higher = faster response.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
        max_missed: int = 15,
        max_expired_age_s: float = 20.0,
        ema_alpha: float = 0.7,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.max_expired_age_s = max_expired_age_s
        self.ema_alpha = ema_alpha

        self._tracks: Dict[int, TrackedPerson] = {}   # track_id → TrackedPerson
        self._update_count: int = 0

        # Global lifecycle counters
        self.total_created: int = 0
        self.total_confirmed: int = 0
        self.total_lost: int = 0
        self.total_reacquired: int = 0
        self.total_expired: int = 0

        # Detection / depth statistics
        self.total_detections: int = 0
        self.total_depth_valid: int = 0
        self.total_depth_invalid: int = 0
        self.peak_concurrent_tracks: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        fused_detections: List[FusedDetection],
        ts: Optional[float] = None,
    ) -> List["TrackedPerson"]:
        """
        Update all tracks with the latest fused detections.

        Parameters
        ----------
        fused_detections : list of FusedDetection
            Output of RGBDFusion.fuse() for the current frame.
        ts : float or None
            Timestamp for this frame (simulated mission time or real wall-clock).
            Defaults to ``time.time()`` when None, ensuring real elapsed time
            is reflected in first_seen_ts / last_seen_ts.

        Returns
        -------
        list of TrackedPerson
            All currently CONFIRMED tracks.
        """
        self._update_count += 1
        now = ts if ts is not None else time.time()

        self.total_detections += len(fused_detections)
        for fd in fused_detections:
            if fd.distance_valid:
                self.total_depth_valid += 1
            else:
                self.total_depth_invalid += 1

        active_tracks = [
            t for t in self._tracks.values()
            if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)
        ]

        # ----------------------------------------------------------------
        # 1. IoU association
        # ----------------------------------------------------------------
        matched_track_ids, unmatched_det_indices, unmatched_track_ids = \
            self._associate(active_tracks, fused_detections)

        # ----------------------------------------------------------------
        # 2. Update matched tracks
        # ----------------------------------------------------------------
        for tid, det_idx in matched_track_ids:
            track = self._tracks[tid]
            fd = fused_detections[det_idx]
            was_tentative = track.state == TrackState.TENTATIVE
            was_lost = track.state == TrackState.LOST

            track.update_from_fused(fd, ts=ts)

            # State transitions
            if was_lost:
                track.state = TrackState.CONFIRMED
                self.total_reacquired += 1
                track._record_event(LifecycleEvent.REACQUIRED)
                logger.info("Track %d REACQUIRED (hits=%d)", tid, track.hit_count)
            elif was_tentative and track.hit_count >= self.min_hits:
                track.state = TrackState.CONFIRMED
                track._record_event(LifecycleEvent.CONFIRMED)
                self.total_confirmed += 1
                logger.info("Track %d CONFIRMED (hits=%d)", tid, track.hit_count)

        # ----------------------------------------------------------------
        # 3. Create new tracks for unmatched detections
        # ----------------------------------------------------------------
        for det_idx in unmatched_det_indices:
            fd = fused_detections[det_idx]
            tid = _alloc_track_id()
            track = TrackedPerson(
                track_id=tid,
                _ema_alpha=self.ema_alpha,
                first_seen_ts=now,
                last_seen_ts=now,
            )
            track._record_event(LifecycleEvent.CREATED)
            track.update_from_fused(fd, ts=ts)
            self._tracks[tid] = track
            self.total_created += 1
            logger.debug("Track %d CREATED", tid)

        # ----------------------------------------------------------------
        # 4. Mark unmatched existing tracks as missed
        #    age_frames is incremented inside mark_missed()
        # ----------------------------------------------------------------
        for tid in unmatched_track_ids:
            track = self._tracks[tid]
            track.mark_missed(ts=ts)
            if track.missed_frames > self.max_missed:
                if track.state != TrackState.LOST:
                    track.state = TrackState.LOST
                    track._record_event(LifecycleEvent.LOST)
                    self.total_lost += 1
                    logger.info("Track %d LOST (missed=%d)", tid, track.missed_frames)

        # ----------------------------------------------------------------
        # 5. Expire old LOST tracks
        # ----------------------------------------------------------------
        for tid in list(self._tracks):
            track = self._tracks[tid]
            if track.state == TrackState.LOST:
                age_since_seen = now - track.last_seen_ts
                if age_since_seen > self.max_expired_age_s:
                    track.state = TrackState.EXPIRED
                    track._record_event(LifecycleEvent.EXPIRED)
                    self.total_expired += 1
                    logger.info(
                        "Track %d EXPIRED (age_since_seen=%.1fs)", tid, age_since_seen
                    )

        # ----------------------------------------------------------------
        # 6. Update peak concurrent count
        # ----------------------------------------------------------------
        active_count = sum(
            1 for t in self._tracks.values()
            if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)
        )
        if active_count > self.peak_concurrent_tracks:
            self.peak_concurrent_tracks = active_count

        # ----------------------------------------------------------------
        # 7. Return confirmed tracks
        # ----------------------------------------------------------------
        return [t for t in self._tracks.values() if t.state == TrackState.CONFIRMED]

    def all_tracks(self) -> List[TrackedPerson]:
        """Return all tracks regardless of state."""
        return list(self._tracks.values())

    def get_track(self, track_id: int) -> Optional[TrackedPerson]:
        return self._tracks.get(track_id)

    def active_tracks(self) -> List[TrackedPerson]:
        """Return TENTATIVE + CONFIRMED tracks."""
        return [
            t for t in self._tracks.values()
            if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)
        ]

    def lost_tracks(self) -> List[TrackedPerson]:
        """Return LOST tracks."""
        return [t for t in self._tracks.values() if t.state == TrackState.LOST]

    def expired_tracks(self) -> List[TrackedPerson]:
        """Return EXPIRED tracks."""
        return [t for t in self._tracks.values() if t.state == TrackState.EXPIRED]

    def lifecycle_summary(self) -> dict:
        """Return JSON-serialisable lifecycle and per-track report."""
        all_t = list(self._tracks.values())
        confirmed_count = sum(1 for t in all_t if t.state == TrackState.CONFIRMED)
        tentative_count = sum(1 for t in all_t if t.state == TrackState.TENTATIVE)
        lost_count = sum(1 for t in all_t if t.state == TrackState.LOST)
        expired_count = sum(1 for t in all_t if t.state == TrackState.EXPIRED)

        # Depth validity stats across all track lifecycle events
        total_valid_frames = sum(
            sum(1 for _, evt, _ in t.lifecycle if evt == LifecycleEvent.DEPTH_VALID)
            for t in all_t
        )
        total_invalid_frames = sum(
            sum(1 for _, evt, _ in t.lifecycle if evt == LifecycleEvent.DEPTH_INVALID)
            for t in all_t
        )

        return {
            "total_update_calls": self._update_count,
            "total_detections_processed": self.total_detections,
            "total_depth_valid_detections": self.total_depth_valid,
            "total_depth_invalid_detections": self.total_depth_invalid,
            "total_tracks_created": self.total_created,
            "total_tracks_confirmed": self.total_confirmed,
            "total_tracks_lost": self.total_lost,
            "total_tracks_reacquired": self.total_reacquired,
            "total_tracks_expired": self.total_expired,
            "peak_concurrent_tracks": self.peak_concurrent_tracks,
            "active_track_count": confirmed_count + tentative_count,
            "confirmed_track_count": confirmed_count,
            "tentative_track_count": tentative_count,
            "lost_track_count": lost_count,
            "expired_track_count": expired_count,
            "total_track_depth_valid_frames": total_valid_frames,
            "total_track_depth_invalid_frames": total_invalid_frames,
            "tracks": {
                str(tid): {
                    **t.to_dict(),
                    "lifecycle_history": t.lifecycle_history(),
                }
                for tid, t in self._tracks.items()
            },
        }

    def reset(self) -> None:
        """Clear all tracks (keeps global ID counter monotonic)."""
        self._tracks.clear()
        self._update_count = 0
        self.total_created = 0
        self.total_confirmed = 0
        self.total_lost = 0
        self.total_reacquired = 0
        self.total_expired = 0
        self.total_detections = 0
        self.total_depth_valid = 0
        self.total_depth_invalid = 0
        self.peak_concurrent_tracks = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        if inter == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _associate(
        self,
        tracks: List[TrackedPerson],
        fused: List[FusedDetection],
    ) -> Tuple[
        List[Tuple[int, int]],   # (track_id, det_index)
        List[int],               # unmatched detection indices
        List[int],               # unmatched track IDs
    ]:
        if not tracks or not fused:
            return [], list(range(len(fused))), [t.track_id for t in tracks]

        # Build IoU cost matrix
        iou_matrix = np.zeros((len(tracks), len(fused)), dtype=np.float32)
        for ti, track in enumerate(tracks):
            bbox_t = track.latest_bbox or (0, 0, 0, 0)
            for di, fd in enumerate(fused):
                iou_matrix[ti, di] = self._iou(bbox_t, fd.detection.bbox)

        # Greedy matching (descending IoU)
        matched: List[Tuple[int, int]] = []
        matched_det_set: set = set()
        matched_track_set: set = set()

        # Sort all pairs by IoU descending
        pairs = sorted(
            ((iou_matrix[ti, di], ti, di) for ti in range(len(tracks)) for di in range(len(fused))),
            reverse=True,
        )
        for iou_val, ti, di in pairs:
            if iou_val < self.iou_threshold:
                break
            if ti in matched_track_set or di in matched_det_set:
                continue
            matched.append((tracks[ti].track_id, di))
            matched_track_set.add(ti)
            matched_det_set.add(di)

        unmatched_dets = [di for di in range(len(fused)) if di not in matched_det_set]
        unmatched_tracks = [
            tracks[ti].track_id for ti in range(len(tracks)) if ti not in matched_track_set
        ]

        return matched, unmatched_dets, unmatched_tracks
