"""
person_tracker.py — Multi-Person Tracking and RGB-D Distance Association.

Maintains persistent identities for detected persons across successive RealSense
RGB-D frames, associating bounding boxes with smoothed distance measurements,
handling temporary detection loss, and reacquiring tracks without ID switches.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .inference_engine import Detection
from .rgbd_fusion import FusedDetection

logger = logging.getLogger(__name__)

# Color palette for distinct track visualization (BGR)
_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 128),    # Spring Green
    (255, 128, 0),    # Azure Blue
    (0, 215, 255),    # Gold / Amber
    (255, 0, 128),    # Magenta / Violet
    (0, 255, 255),    # Bright Yellow
    (128, 255, 0),    # Chartreuse
    (255, 192, 203),  # Pink
    (0, 165, 255),    # Orange
    (238, 130, 238),  # Lavender
    (144, 238, 144),  # Light Green
]


# ---------------------------------------------------------------------------
# Helper Geometry Functions
# ---------------------------------------------------------------------------

def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    box_a_area = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    box_b_area = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / float(union_area)


def compute_normalized_center_dist(
    box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
) -> float:
    """Calculate Euclidean distance between bounding box centers normalized by box scale."""
    ca_x = (box_a[0] + box_a[2]) / 2.0
    ca_y = (box_a[1] + box_a[3]) / 2.0
    cb_x = (box_b[0] + box_b[2]) / 2.0
    cb_y = (box_b[1] + box_b[3]) / 2.0

    bw = max(box_a[2] - box_a[0], box_b[2] - box_b[0], 1)
    bh = max(box_a[3] - box_a[1], box_b[3] - box_b[1], 1)

    dx = (ca_x - cb_x) / float(bw)
    dy = (ca_y - cb_y) / float(bh)
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# TrackedPerson Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackedPerson:
    """Represents a persistently tracked individual with RGB-D spatial metrics."""

    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]              # (x1, y1, x2, y2) in pixels
    center_px: Tuple[int, int]                  # (cx, cy) in pixels
    confidence: float
    distance_m: Optional[float] = None          # Estimated distance in metres
    distance_valid: bool = False                # Current distance validity
    valid_depth_percent: float = 0.0            # Depth pixel coverage in ROI
    depth_method: str = "median"                # "median", "unreliable", "invalid"
    state: str = "confirmed"                    # "tentative", "confirmed", "lost", "deleted"
    age: int = 1                                # Total frames active
    hits: int = 1                               # Total detection matches
    time_since_update: int = 0                  # Consecutive frames without a detection match
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) estimated pixel motion / frame
    history: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert track state to structured dictionary format."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "state": self.state,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "center_coordinates": list(self.center_px),
            "estimated_distance_m": round(self.distance_m, 3) if self.distance_m is not None else None,
            "distance_valid": self.distance_valid,
            "valid_depth_percent": round(self.valid_depth_percent, 2),
            "depth_method": self.depth_method,
            "age_frames": self.age,
            "hits": self.hits,
            "time_since_update_frames": self.time_since_update,
            "first_seen_timestamp": round(self.first_seen_timestamp, 3),
            "last_seen_timestamp": round(self.last_seen_timestamp, 3),
        }


# ---------------------------------------------------------------------------
# PersonTracker Class
# ---------------------------------------------------------------------------

class PersonTracker:
    """
    Multi-person tracker associating detections and RealSense depth across successive frames.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU overlap to consider an existing track a match (default: 0.3).
    max_lost_frames : int
        Maximum consecutive frames a track can be lost before expiration (default: 15).
    min_hits : int
        Minimum detection hits before track is considered confirmed (default: 1).
    distance_ema_alpha : float
        Exponential moving average weight for smoothing distance measurements (default: 0.7).
    max_center_dist_norm : float
        Fallback center distance limit when IoU is 0 (default: 1.2).
    dedup_iou_threshold : float
        IoU threshold to suppress duplicate overlapping tracks (default: 0.85).
    config : dict, optional
        Optional configuration dictionary (e.g. from config/tracking.yaml).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
        min_hits: int = 1,
        distance_ema_alpha: float = 0.7,
        max_center_dist_norm: float = 1.2,
        dedup_iou_threshold: float = 0.85,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        self.iou_threshold = float(cfg.get("iou_threshold", iou_threshold))
        self.max_lost_frames = int(cfg.get("max_lost_frames", max_lost_frames))
        self.min_hits = int(cfg.get("min_hits", min_hits))
        self.distance_ema_alpha = float(cfg.get("distance_ema_alpha", distance_ema_alpha))
        self.max_center_dist_norm = float(cfg.get("max_center_dist_norm", max_center_dist_norm))
        self.dedup_iou_threshold = float(cfg.get("dedup_iou_threshold", dedup_iou_threshold))

        self._next_id: int = 1
        self._tracks: List[TrackedPerson] = []
        self._frame_count: int = 0
        self._last_timestamp: float = 0.0

    @property
    def tracks(self) -> List[TrackedPerson]:
        """Return all active tracks (confirmed, tentative, or lost)."""
        return [t for t in self._tracks if t.state != "deleted"]

    @property
    def confirmed_tracks(self) -> List[TrackedPerson]:
        """Return active tracks with at least min_hits detections."""
        return [t for t in self._tracks if t.state in ("confirmed", "lost") and t.hits >= self.min_hits]

    def reset(self) -> None:
        """Clear all active tracks and reset track ID counter."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        self._last_timestamp = 0.0

    # -----------------------------------------------------------------------
    # Main Update
    # -----------------------------------------------------------------------

    def update(
        self,
        fused_detections: List[FusedDetection],
        timestamp: Optional[float] = None,
    ) -> List[TrackedPerson]:
        """
        Update tracking state with new detections from the current frame.

        Parameters
        ----------
        fused_detections : list of FusedDetection
            Detections with associated RGB-D depth measurements.
        timestamp : float, optional
            Monotonic timestamp of current frame.

        Returns
        -------
        list of TrackedPerson
            Updated active tracks.
        """
        now = timestamp if (timestamp is not None and timestamp > 0) else time.monotonic()
        self._frame_count += 1
        self._last_timestamp = now

        active_tracks = [t for t in self._tracks if t.state != "deleted"]

        # 1. Motion prediction for existing tracks
        for track in active_tracks:
            track.age += 1
            if track.time_since_update > 0:
                track.state = "lost"
                # Velocity coasting (damped by 0.5 to prevent divergence)
                vx, vy = track.velocity
                pred_cx = int(track.center_px[0] + vx * 0.5)
                pred_cy = int(track.center_px[1] + vy * 0.5)
                w = track.bbox[2] - track.bbox[0]
                h = track.bbox[3] - track.bbox[1]
                track.bbox = (pred_cx - w // 2, pred_cy - h // 2, pred_cx + w // 2, pred_cy + h // 2)
                track.center_px = (pred_cx, pred_cy)

        # 2. Match detections with existing tracks
        matched_tracks, matched_dets = self._associate_detections(active_tracks, fused_detections)

        # 3. Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            track = active_tracks[track_idx]
            det = fused_detections[det_idx]

            # Compute pixel velocity
            new_cx, new_cy = det.center_px
            old_cx, old_cy = track.center_px
            inst_vx = float(new_cx - old_cx)
            inst_vy = float(new_cy - old_cy)
            track.velocity = (
                0.7 * inst_vx + 0.3 * track.velocity[0],
                0.7 * inst_vy + 0.3 * track.velocity[1],
            )

            # Update geometry
            track.bbox = det.detection.bbox
            track.center_px = det.center_px
            track.confidence = det.detection.confidence
            track.time_since_update = 0
            track.hits += 1
            track.state = "confirmed"
            track.last_seen_timestamp = now

            # Append position to trajectory history (max 30 points)
            track.history.append(track.center_px)
            if len(track.history) > 30:
                track.history.pop(0)

            # Update RGB-D distance with EMA smoothing
            if det.distance_valid and det.distance_m is not None:
                if track.distance_m is not None:
                    smoothed_dist = (
                        self.distance_ema_alpha * det.distance_m
                        + (1.0 - self.distance_ema_alpha) * track.distance_m
                    )
                    track.distance_m = round(smoothed_dist, 3)
                else:
                    track.distance_m = round(det.distance_m, 3)
                track.distance_valid = True
                track.depth_method = det.depth_method
                track.valid_depth_percent = det.valid_depth_percent
            else:
                # Retain last known distance as unconfirmed if measurement was momentarily lost
                track.distance_valid = False
                if det.depth_method != "median":
                    track.depth_method = det.depth_method

        # 4. Initiate new tracks for unmatched detections
        all_matched_det_indices = set(matched_dets)
        for j, det in enumerate(fused_detections):
            if j not in all_matched_det_indices:
                new_track = TrackedPerson(
                    track_id=self._next_id,
                    class_name=det.detection.class_name,
                    bbox=det.detection.bbox,
                    center_px=det.center_px,
                    confidence=det.detection.confidence,
                    distance_m=det.distance_m,
                    distance_valid=det.distance_valid,
                    valid_depth_percent=det.valid_depth_percent,
                    depth_method=det.depth_method,
                    state="confirmed" if self.min_hits <= 1 else "tentative",
                    age=1,
                    hits=1,
                    time_since_update=0,
                    first_seen_timestamp=now,
                    last_seen_timestamp=now,
                    velocity=(0.0, 0.0),
                    history=[det.center_px],
                )
                self._next_id += 1
                self._tracks.append(new_track)
                logger.debug("New track #%d initiated at %s", new_track.track_id, new_track.center_px)

        # 5. Handle unmatched tracks (missed detections / loss)
        all_matched_track_indices = set(matched_tracks)
        for i, track in enumerate(active_tracks):
            if i not in all_matched_track_indices:
                track.time_since_update += 1
                if track.time_since_update > self.max_lost_frames:
                    track.state = "deleted"
                    logger.debug("Track #%d expired after %d lost frames", track.track_id, track.time_since_update)
                else:
                    track.state = "lost"

        # 6. Deduplication and track maintenance
        self._deduplicate_tracks()
        self._tracks = [t for t in self._tracks if t.state != "deleted"]

        return self.tracks

    # -----------------------------------------------------------------------
    # Association Algorithm
    # -----------------------------------------------------------------------

    def _associate_detections(
        self, tracks: List[TrackedPerson], detections: List[FusedDetection]
    ) -> Tuple[List[int], List[int]]:
        """Greedy bipartite matching using IoU and normalized center distance."""
        if not tracks or not detections:
            return [], []

        candidates: List[Tuple[float, int, int]] = []
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = compute_iou(track.bbox, det.detection.bbox)
                if iou >= self.iou_threshold:
                    # Best match metric: 1.0 - IoU (range: 0.0 to 1.0 - threshold)
                    cost = 1.0 - iou
                    candidates.append((cost, i, j))
                else:
                    # Fallback metric for rapid displacement / low FPS
                    norm_dist = compute_normalized_center_dist(track.bbox, det.detection.bbox)
                    if norm_dist <= self.max_center_dist_norm:
                        # Secondary match metric: 1.0 + norm_dist
                        cost = 1.0 + norm_dist
                        candidates.append((cost, i, j))

        # Sort candidate matches by lowest cost (highest similarity)
        candidates.sort(key=lambda x: x[0])

        matched_tracks: List[int] = []
        matched_dets: List[int] = []
        used_tracks = set()
        used_dets = set()

        for cost, i, j in candidates:
            if i not in used_tracks and j not in used_dets:
                used_tracks.add(i)
                used_dets.add(j)
                matched_tracks.append(i)
                matched_dets.append(j)

        return matched_tracks, matched_dets

    def _deduplicate_tracks(self) -> None:
        """Suppress newer duplicate tracks sharing excessive overlap with established tracks."""
        active = [t for t in self._tracks if t.state != "deleted"]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                t1, t2 = active[i], active[j]
                if t1.state == "deleted" or t2.state == "deleted":
                    continue
                iou = compute_iou(t1.bbox, t2.bbox)
                if iou >= self.dedup_iou_threshold:
                    # Suppress track with fewer hits / younger age
                    if t1.hits >= t2.hits:
                        t2.state = "deleted"
                        logger.debug("Suppressed duplicate track #%d in favor of #%d (IoU=%.2f)", t2.track_id, t1.track_id, iou)
                    else:
                        t1.state = "deleted"
                        logger.debug("Suppressed duplicate track #%d in favor of #%d (IoU=%.2f)", t1.track_id, t2.track_id, iou)

    # -----------------------------------------------------------------------
    # Visual Annotation & Export
    # -----------------------------------------------------------------------

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: Optional[List[TrackedPerson]] = None,
        draw_trajectory: bool = True,
    ) -> np.ndarray:
        """
        Draw persistent track IDs, bounding boxes, distance, and trajectories on frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        tracks : list of TrackedPerson, optional
            List of tracks to draw (defaults to confirmed tracks).
        draw_trajectory : bool
            Whether to draw trajectory motion tails.

        Returns
        -------
        np.ndarray
            Annotated frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        display_tracks = tracks if tracks is not None else self.confirmed_tracks

        for track in display_tracks:
            color = _PALETTE[(track.track_id - 1) % len(_PALETTE)]
            x1, y1, x2, y2 = track.bbox
            x1_c, y1_c = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2_c, y2_c = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

            # Bounding box style: solid for confirmed, dashed/thinner for lost
            thickness = 2 if track.state == "confirmed" else 1
            cv2.rectangle(out, (x1_c, y1_c), (x2_c, y2_c), color, thickness)

            # Center circle
            cx, cy = track.center_px
            cx_c, cy_c = max(0, min(w - 1, cx)), max(0, min(h - 1, cy))
            cv2.circle(out, (cx_c, cy_c), 3, (0, 255, 255), -1)

            # Trajectory tail
            if draw_trajectory and len(track.history) > 1:
                pts = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

            # Label format: "ID #X Survivor (XX%) | X.XXm [state]"
            conf_pct = int(track.confidence * 100)
            if track.distance_valid and track.distance_m is not None:
                dist_str = f"{track.distance_m:.2f}m"
            elif track.distance_m is not None:
                dist_str = f"~{track.distance_m:.2f}m (unverified)"
            else:
                dist_str = "depth N/A"

            state_suffix = f"[{track.state}]" if track.state != "confirmed" else ""
            label = f"ID #{track.track_id} Survivor ({conf_pct}%) | {dist_str} {state_suffix}".strip()

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            txt_thick = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, txt_thick)
            label_y1 = max(y1_c - th - baseline - 4, 0)

            # Banner rectangle
            cv2.rectangle(
                out,
                (x1_c, label_y1),
                (x1_c + tw + 4, label_y1 + th + baseline + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                out,
                label,
                (x1_c + 2, label_y1 + th + 2),
                font,
                font_scale,
                (0, 0, 0),
                txt_thick,
                cv2.LINE_AA,
            )

        return out

    def export_json(
        self,
        tracks: Optional[List[TrackedPerson]] = None,
        filepath: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> str:
        """
        Export structured tracking data to JSON format.

        Parameters
        ----------
        tracks : list of TrackedPerson, optional
            Tracks to export (defaults to current active tracks).
        filepath : str or Path, optional
            Path to write the JSON file.
        indent : int
            Indentation level for formatting.

        Returns
        -------
        str
            JSON string.
        """
        target_tracks = tracks if tracks is not None else self.tracks
        payload = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "monotonic_time": round(self._last_timestamp, 3),
            "frame_index": self._frame_count,
            "active_tracks_count": len(target_tracks),
            "tracks": [t.to_dict() for t in target_tracks],
        }

        json_str = json.dumps(payload, indent=indent)

        if filepath is not None:
            p = Path(filepath).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Exported tracking JSON to: %s", p)

        return json_str

