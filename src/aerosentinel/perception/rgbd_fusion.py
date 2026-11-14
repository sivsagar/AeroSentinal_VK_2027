"""
rgbd_fusion.py — RGB-D fusion for AeroSentinel.

Associates person detections with depth measurements to produce
FusedDetection objects with distance and depth-coverage metadata.

Key design rules
----------------
* ``distance_valid`` is True **only** when:
  - ``distance_m`` is a finite positive number inside [min_valid_depth_m, max_valid_depth_m]
  - ``valid_depth_percent`` >= ``min_valid_depth_pct`` threshold (configurable, default 5 %)
* ``valid_depth_percent`` always reflects the **current frame**'s bbox coverage — it
  is never inherited from a previous frame.
* ``depth_method`` is one of: ``'center'``, ``'median'``, ``'invalid'``.
  ``'invalid'`` is returned whenever depth coverage or range check fails.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np

from .inference_engine import Detection

logger = logging.getLogger(__name__)

# Threshold for timestamp sync warning (seconds)
_SYNC_WARN_S = 0.100


# ---------------------------------------------------------------------------
# FusedDetection dataclass
# ---------------------------------------------------------------------------

@dataclass
class FusedDetection:
    """
    Result of fusing one Detection with the current depth frame.

    Attributes
    ----------
    detection : Detection
        The original RGB detection.
    distance_m : Optional[float]
        Estimated distance in metres; None when depth is unavailable.
    distance_valid : bool
        True **only** when distance_m is finite, positive, within range,
        AND valid_depth_percent >= threshold. Never set True when
        valid_depth_percent == 0.
    depth_method : str
        How the distance was computed: 'center', 'median', or 'invalid'.
    valid_depth_percent : float
        Fraction [0.0, 100.0] of bbox pixels with valid depth readings.
        Always reflects the current frame — never stale.
    position_valid : bool
        False until GPS/localisation is confirmed available.
    camera : str
    timestamp : str
        ISO-8601 UTC timestamp of this fusion result.
    """

    detection: Detection
    distance_m: Optional[float]
    distance_valid: bool
    depth_method: str               # 'center', 'median', 'invalid'
    valid_depth_percent: float = 0.0  # 0.0–100.0, current-frame coverage
    position_valid: bool = False      # False until GPS confirmed
    camera: str = "rgbd_fusion"
    timestamp: str = ""               # ISO-8601 UTC

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Safety invariant: distance_valid can NEVER be True when
        # valid_depth_percent is zero or distance_m is not a valid number.
        if self.distance_valid:
            if (
                self.distance_m is None
                or not math.isfinite(self.distance_m)
                or self.distance_m <= 0.0
                or self.valid_depth_percent <= 0.0
            ):
                # Force consistency
                self.distance_valid = False
                self.depth_method = "invalid"
                logger.warning(
                    "FusedDetection: forced distance_valid=False "
                    "(distance_m=%s valid_depth_percent=%.1f)",
                    self.distance_m,
                    self.valid_depth_percent,
                )

    def to_dict(self) -> dict:
        return {
            "class": self.detection.class_name,
            "label": "Potential Survivor",
            "confidence": round(self.detection.confidence, 4),
            "distance_m": round(self.distance_m, 3) if self.distance_m is not None else None,
            "distance_valid": self.distance_valid,
            "depth_method": self.depth_method,
            "valid_depth_percent": round(self.valid_depth_percent, 2),
            "camera": self.camera,
            "timestamp": self.timestamp,
            "position_valid": self.position_valid,
            "note": "Requires human verification",
        }


# ---------------------------------------------------------------------------
# RGBDFusion
# ---------------------------------------------------------------------------

class RGBDFusion:
    """
    Associate RGB detections with depth measurements.

    Parameters
    ----------
    min_valid_depth_m : float
        Depths below this are noise/invalid (default 0.1 m).
    max_valid_depth_m : float
        Depths above this are out-of-range (default 10.0 m).
    median_roi_fraction : float
        Fraction of the bbox interior used for median depth (default 0.5).
    min_valid_depth_pct : float
        Minimum percentage of valid depth pixels inside the bbox ROI required
        before declaring distance_valid=True (default 5.0 %).
    """

    def __init__(
        self,
        min_valid_depth_m: float = 0.1,
        max_valid_depth_m: float = 10.0,
        median_roi_fraction: float = 0.5,
        min_valid_depth_pct: float = 5.0,
    ) -> None:
        self.min_valid_depth_m = min_valid_depth_m
        self.max_valid_depth_m = max_valid_depth_m
        self.median_roi_fraction = median_roi_fraction
        self.min_valid_depth_pct = min_valid_depth_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        detections: List[Detection],
        depth_meters: np.ndarray,
        rgb_timestamp: Optional[float] = None,
        depth_timestamp: Optional[float] = None,
    ) -> List[FusedDetection]:
        """
        Fuse detections with a float32 depth frame.

        Parameters
        ----------
        detections : list of Detection
        depth_meters : np.ndarray
            float32, shape (H, W) or (H, W, 1); values in metres; 0 = invalid.
        rgb_timestamp : float or None
        depth_timestamp : float or None

        Returns
        -------
        list of FusedDetection — one per input detection.
        """
        self._check_sync(rgb_timestamp, depth_timestamp)

        # Squeeze single-channel
        depth_2d = depth_meters.squeeze() if depth_meters.ndim == 3 else depth_meters
        if depth_2d.ndim != 2:
            logger.error("RGBDFusion.fuse: unexpected depth shape %s", depth_meters.shape)
            depth_2d = np.zeros((480, 640), dtype=np.float32)

        fused: List[FusedDetection] = []
        for det in detections:
            fd = self._fuse_one(det, depth_2d)
            fused.append(fd)
        return fused

    def draw_fused_detections(
        self, frame: np.ndarray, fused_detections: List[FusedDetection]
    ) -> np.ndarray:
        """Draw bounding boxes with distance labels on a BGR frame."""
        out = frame.copy()
        for fd in fused_detections:
            det = fd.detection
            x1, y1, x2, y2 = det.bbox

            h, w = out.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            colour = (0, 255, 128)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            pct = int(det.confidence * 100)
            if fd.distance_valid and fd.distance_m is not None:
                label = (
                    f"Potential Survivor ({pct}%) | "
                    f"{fd.distance_m:.2f}m [{fd.depth_method}] "
                    f"cov={fd.valid_depth_percent:.0f}%"
                )
            else:
                label = f"Potential Survivor ({pct}%) | depth N/A"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y1 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(
                out,
                (x1, label_y1),
                (x1 + tw + 4, label_y1 + th + baseline + 4),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                out, label, (x1 + 2, label_y1 + th + 2),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_sync(
        self, rgb_ts: Optional[float], depth_ts: Optional[float]
    ) -> None:
        if rgb_ts is not None and depth_ts is not None:
            delta = abs(rgb_ts - depth_ts)
            if delta > _SYNC_WARN_S:
                logger.warning(
                    "RGB-D timestamp gap %.1f ms exceeds %.0f ms threshold.",
                    delta * 1000,
                    _SYNC_WARN_S * 1000,
                )

    def _bbox_depth_coverage(
        self, depth_m: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> tuple[float, np.ndarray]:
        """
        Return (valid_depth_percent, valid_values_array) for the bbox region.

        valid_depth_percent is in [0.0, 100.0].
        """
        h, w = depth_m.shape
        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(w, x2), min(h, y2)

        if rx2 <= rx1 or ry2 <= ry1:
            return 0.0, np.array([], dtype=np.float32)

        patch = depth_m[ry1:ry2, rx1:rx2]
        total = patch.size
        if total == 0:
            return 0.0, np.array([], dtype=np.float32)

        valid_mask = (patch >= self.min_valid_depth_m) & (patch <= self.max_valid_depth_m)
        valid_vals = patch[valid_mask]
        pct = 100.0 * valid_vals.size / total
        return pct, valid_vals

    def _fuse_one(self, det: Detection, depth_m: np.ndarray) -> FusedDetection:
        x1, y1, x2, y2 = det.bbox
        h, w = depth_m.shape

        # ----------------------------------------------------------------
        # 1. Compute bbox depth coverage (always, for every detection)
        # ----------------------------------------------------------------
        valid_pct, valid_vals = self._bbox_depth_coverage(depth_m, x1, y1, x2, y2)

        # ----------------------------------------------------------------
        # 2. Coverage gate — if too few valid pixels, return invalid now
        # ----------------------------------------------------------------
        if valid_pct < self.min_valid_depth_pct or valid_vals.size == 0:
            return FusedDetection(
                detection=det,
                distance_m=None,
                distance_valid=False,
                depth_method="invalid",
                valid_depth_percent=valid_pct,
            )

        # ----------------------------------------------------------------
        # 3. Try centre pixel first (must be within valid range)
        # ----------------------------------------------------------------
        cx = max(0, min(w - 1, (x1 + x2) // 2))
        cy = max(0, min(h - 1, (y1 + y2) // 2))
        center_d = float(depth_m[cy, cx])

        if self._is_valid_depth(center_d):
            return FusedDetection(
                detection=det,
                distance_m=center_d,
                distance_valid=True,
                depth_method="center",
                valid_depth_percent=valid_pct,
            )

        # ----------------------------------------------------------------
        # 4. Fall back: median of valid pixels in shrunk ROI
        # ----------------------------------------------------------------
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        margin_x = int(bw * (1.0 - self.median_roi_fraction) / 2)
        margin_y = int(bh * (1.0 - self.median_roi_fraction) / 2)
        rx1 = max(0, x1 + margin_x)
        ry1 = max(0, y1 + margin_y)
        rx2 = min(w, x2 - margin_x)
        ry2 = min(h, y2 - margin_y)

        if rx2 > rx1 and ry2 > ry1:
            inner = depth_m[ry1:ry2, rx1:rx2]
            inner_valid = inner[
                (inner >= self.min_valid_depth_m) & (inner <= self.max_valid_depth_m)
            ]
            if inner_valid.size > 0:
                median_d = float(np.median(inner_valid))
                if self._is_valid_depth(median_d):
                    return FusedDetection(
                        detection=det,
                        distance_m=median_d,
                        distance_valid=True,
                        depth_method="median",
                        valid_depth_percent=valid_pct,
                    )

        # ----------------------------------------------------------------
        # 5. All attempts failed — invalid
        # ----------------------------------------------------------------
        return FusedDetection(
            detection=det,
            distance_m=None,
            distance_valid=False,
            depth_method="invalid",
            valid_depth_percent=valid_pct,
        )

    def _is_valid_depth(self, d: float) -> bool:
        return (
            math.isfinite(d)
            and self.min_valid_depth_m <= d <= self.max_valid_depth_m
        )
