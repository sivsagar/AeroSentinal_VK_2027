"""
rgbd_fusion.py — RGB-D fusion for AeroSentinel.

Associates person detections with depth measurements to produce
FusedDetection objects with distance information.
"""

from __future__ import annotations

import logging
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
    detection: Detection
    distance_m: Optional[float]
    distance_valid: bool
    depth_method: str               # 'center', 'median', 'invalid'
    position_valid: bool = False    # False until GPS/localisation available
    camera: str = "rgbd_fusion"
    timestamp: str = ""             # ISO-8601

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> dict:
        return {
            "class": self.detection.class_name,
            "label": "Potential Survivor",
            "confidence": round(self.detection.confidence, 4),
            "distance_m": round(self.distance_m, 3) if self.distance_m is not None else None,
            "distance_valid": self.distance_valid,
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
        Depths below this are considered noise/invalid (default 0.1 m).
    max_valid_depth_m : float
        Depths above this are considered out-of-range (default 10.0 m).
    median_roi_fraction : float
        Fraction of the bbox interior to use for median depth (default 0.5).
    """

    def __init__(
        self,
        min_valid_depth_m: float = 0.1,
        max_valid_depth_m: float = 10.0,
        median_roi_fraction: float = 0.5,
    ) -> None:
        self.min_valid_depth_m = min_valid_depth_m
        self.max_valid_depth_m = max_valid_depth_m
        self.median_roi_fraction = median_roi_fraction

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
            float32 frame, metres; 0 = invalid.
        rgb_timestamp : float or None
        depth_timestamp : float or None

        Returns
        -------
        list of FusedDetection
        """
        self._check_sync(rgb_timestamp, depth_timestamp)

        fused: List[FusedDetection] = []
        for det in detections:
            fd = self._fuse_one(det, depth_meters)
            fused.append(fd)
        return fused

    def draw_fused_detections(
        self, frame: np.ndarray, fused_detections: List[FusedDetection]
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, confidence and distance on a BGR frame.

        Returns annotated copy.
        """
        out = frame.copy()
        for fd in fused_detections:
            det = fd.detection
            x1, y1, x2, y2 = det.bbox

            # Clamp to frame
            h, w = out.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            colour = (0, 255, 128)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            pct = int(det.confidence * 100)
            if fd.distance_valid and fd.distance_m is not None:
                label = f"Potential Survivor ({pct}%) | {fd.distance_m:.2f}m [{fd.depth_method}]"
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

    def _fuse_one(self, det: Detection, depth_m: np.ndarray) -> FusedDetection:
        x1, y1, x2, y2 = det.bbox
        h, w = depth_m.shape[:2]

        # Try centre depth first
        cx = max(0, min(w - 1, (x1 + x2) // 2))
        cy = max(0, min(h - 1, (y1 + y2) // 2))
        center_d = float(depth_m[cy, cx])

        if self._is_valid_depth(center_d):
            return FusedDetection(
                detection=det,
                distance_m=center_d,
                distance_valid=True,
                depth_method="center",
            )

        # Fall back to median within bbox interior
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w - 1, x2)
        y2c = min(h - 1, y2)

        bw, bh = x2c - x1c, y2c - y1c
        if bw > 0 and bh > 0:
            # Shrink ROI to median_roi_fraction
            margin_x = int(bw * (1 - self.median_roi_fraction) / 2)
            margin_y = int(bh * (1 - self.median_roi_fraction) / 2)
            rx1 = x1c + margin_x
            ry1 = y1c + margin_y
            rx2 = x2c - margin_x
            ry2 = y2c - margin_y
            if rx2 > rx1 and ry2 > ry1:
                patch = depth_m[ry1:ry2, rx1:rx2]
                valid_vals = patch[patch > 0]
                if len(valid_vals) > 0:
                    median_d = float(np.median(valid_vals))
                    if self._is_valid_depth(median_d):
                        return FusedDetection(
                            detection=det,
                            distance_m=median_d,
                            distance_valid=True,
                            depth_method="median",
                        )

        # No valid depth found
        return FusedDetection(
            detection=det,
            distance_m=None,
            distance_valid=False,
            depth_method="invalid",
        )

    def _is_valid_depth(self, d: float) -> bool:
        return self.min_valid_depth_m <= d <= self.max_valid_depth_m
