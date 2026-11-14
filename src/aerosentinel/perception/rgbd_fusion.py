"""
rgbd_fusion.py — RGB-D fusion for AeroSentinel.

Associates person detections with depth measurements to produce
FusedDetection objects with distance information.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Detection associated with depth measurement and spatial quality metrics."""

    detection: Detection
    distance_m: Optional[float]
    distance_valid: bool
    depth_method: str = "median"                    # 'median', 'center', 'unreliable', 'invalid'
    center_px: Tuple[int, int] = (0, 0)             # (cx, cy) pixel coordinates
    valid_depth_percent: float = 0.0                # Percentage of valid depth pixels in ROI
    detection_id: int = 1                           # Sequential identifier for current frame
    position_valid: bool = False                    # False until GPS/localisation available
    camera: str = "rgbd_fusion"
    timestamp: str = ""                             # ISO-8601 timestamp

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to structured dictionary format."""
        return {
            # Standard AeroSentinel detection fields (for mission recording & tests)
            "class": self.detection.class_name,
            "label": "Potential Survivor",
            "confidence": round(self.detection.confidence, 4),
            "distance_m": round(self.distance_m, 3) if self.distance_m is not None else None,
            "distance_valid": self.distance_valid,
            "camera": self.camera,
            "timestamp": self.timestamp,
            "position_valid": self.position_valid,
            "note": "Requires human verification",
            # Milestone explicit structured fields
            "detection_id": self.detection_id,
            "class_name": self.detection.class_name,
            "confidence_score": round(self.detection.confidence, 4),
            "bbox": list(self.detection.bbox),
            "center_coordinates": list(self.center_px),
            "estimated_distance_m": round(self.distance_m, 3) if self.distance_m is not None else None,
            "valid_depth_percent": round(self.valid_depth_percent, 2),
            "depth_measurement_validity": self.distance_valid,
            "depth_method": self.depth_method,
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
    min_valid_depth_percent : float
        Minimum valid-pixel percentage in ROI to consider depth reliable (default 5.0%).
    config : dict, optional
        Optional configuration dictionary (e.g. from config/fusion.yaml).
    """

    def __init__(
        self,
        min_valid_depth_m: float = 0.1,
        max_valid_depth_m: float = 10.0,
        median_roi_fraction: float = 0.5,
        min_valid_depth_percent: float = 5.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        self.min_valid_depth_m = float(cfg.get("min_valid_depth_m", min_valid_depth_m))
        self.max_valid_depth_m = float(cfg.get("max_valid_depth_m", max_valid_depth_m))
        self.median_roi_fraction = float(cfg.get("median_roi_fraction", median_roi_fraction))
        self.min_valid_depth_percent = float(cfg.get("min_valid_depth_percent", min_valid_depth_percent))
        self.sync_tolerance_s = float(cfg.get("sync_tolerance_ms", _SYNC_WARN_S * 1000.0)) / 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        detections: List[Detection],
        depth_frame: np.ndarray,
        rgb_timestamp: Optional[float] = None,
        depth_timestamp: Optional[float] = None,
    ) -> List[FusedDetection]:
        """
        Fuse detections with an aligned depth frame.

        Parameters
        ----------
        detections : list of Detection
        depth_frame : np.ndarray
            Either uint16 millimetres (raw) or float32 metres; 0 = invalid.
        rgb_timestamp : float, optional
        depth_timestamp : float, optional

        Returns
        -------
        list of FusedDetection
        """
        self._check_sync(rgb_timestamp, depth_timestamp)

        # Convert uint16 mm to float32 metres if necessary
        if depth_frame.dtype == np.uint16:
            depth_m = depth_frame.astype(np.float32) * 0.001
        else:
            depth_m = depth_frame.astype(np.float32)

        ts_str = ""
        if rgb_timestamp is not None and rgb_timestamp > 0:
            try:
                ts_str = datetime.fromtimestamp(rgb_timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, OSError):
                ts_str = ""
        elif depth_timestamp is not None and depth_timestamp > 0:
            try:
                ts_str = datetime.fromtimestamp(depth_timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, OSError):
                ts_str = ""

        fused: List[FusedDetection] = []
        for i, det in enumerate(detections):
            fd = self._fuse_one(
                det=det,
                depth_m=depth_m,
                detection_id=i + 1,
                timestamp_str=ts_str or None,
            )
            fused.append(fd)
        return fused

    def draw_fused_detections(
        self,
        frame: np.ndarray,
        fused_detections: List[FusedDetection],
        draw_center: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, center coordinates, confidence and distance on a BGR frame.

        Returns annotated copy.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        for fd in fused_detections:
            det = fd.detection
            x1, y1, x2, y2 = det.bbox

            # Clamp to frame
            x1_c, y1_c = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2_c, y2_c = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

            # Colour coding by measurement validity:
            # Green = valid depth
            # Yellow/Orange = unreliable depth (insufficient valid pixels)
            # Red/Orange = invalid depth (missing or out of range)
            if fd.distance_valid:
                colour = (0, 255, 128)  # Bright Green
            elif fd.depth_method == "unreliable":
                colour = (0, 200, 255)  # Orange-Yellow
            else:
                colour = (0, 100, 255)  # Orange-Red

            cv2.rectangle(out, (x1_c, y1_c), (x2_c, y2_c), colour, 2)

            # Draw center point
            if draw_center:
                cx, cy = fd.center_px
                cx_c, cy_c = max(0, min(w - 1, cx)), max(0, min(h - 1, cy))
                cv2.circle(out, (cx_c, cy_c), 4, (0, 255, 255), -1)

            # Format label
            pct = int(det.confidence * 100)
            id_prefix = f"#{fd.detection_id} " if fd.detection_id > 0 else ""
            if fd.distance_valid and fd.distance_m is not None:
                label = f"{id_prefix}Potential Survivor ({pct}%) | {fd.distance_m:.2f}m [{fd.valid_depth_percent:.0f}% depth]"
            elif fd.depth_method == "unreliable" and fd.distance_m is not None:
                label = f"{id_prefix}Potential Survivor ({pct}%) | ~{fd.distance_m:.2f}m [unreliable]"
            else:
                label = f"{id_prefix}Potential Survivor ({pct}%) | depth N/A"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y1 = max(y1_c - th - baseline - 4, 0)
            cv2.rectangle(
                out,
                (x1_c, label_y1),
                (x1_c + tw + 4, label_y1 + th + baseline + 4),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                out,
                label,
                (x1_c + 2, label_y1 + th + 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
        return out

    def save_annotated_frame(
        self,
        frame: np.ndarray,
        output_path: Union[str, Path],
        fused_detections: Optional[List[FusedDetection]] = None,
    ) -> Path:
        """Save annotated visual output frame to disk for visual verification."""
        out_frame = self.draw_fused_detections(frame, fused_detections) if fused_detections else frame
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), out_frame)
        logger.info("Saved annotated RGB-D frame to: %s", p)
        return p

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_sync(
        self, rgb_ts: Optional[float], depth_ts: Optional[float]
    ) -> None:
        if rgb_ts is not None and depth_ts is not None:
            delta = abs(rgb_ts - depth_ts)
            if delta > self.sync_tolerance_s:
                logger.warning(
                    "RGB-D timestamp gap %.1f ms exceeds %.0f ms threshold.",
                    delta * 1000.0,
                    self.sync_tolerance_s * 1000.0,
                )

    def _fuse_one(
        self,
        det: Detection,
        depth_m: np.ndarray,
        detection_id: int = 1,
        timestamp_str: Optional[str] = None,
    ) -> FusedDetection:
        h, w = depth_m.shape[:2]
        x1, y1, x2, y2 = det.bbox

        # 1. Calculate centre coordinates
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        # 2. Bounding box boundary handling & clamping
        x1_c = max(0, min(w, x1))
        y1_c = max(0, min(h, y1))
        x2_c = max(0, min(w, x2))
        y2_c = max(0, min(h, y2))

        # Degenerate or completely out-of-frame bbox
        if x2_c <= x1_c or y2_c <= y1_c:
            return FusedDetection(
                detection=det,
                distance_m=None,
                distance_valid=False,
                depth_method="invalid",
                center_px=(cx, cy),
                valid_depth_percent=0.0,
                detection_id=detection_id,
                timestamp=timestamp_str or "",
            )

        # 3. Extract depth ROI for central region
        bw = x2_c - x1_c
        bh = y2_c - y1_c
        margin_x = int(round(bw * (1.0 - self.median_roi_fraction) / 2.0))
        margin_y = int(round(bh * (1.0 - self.median_roi_fraction) / 2.0))
        rx1 = max(0, x1_c + margin_x)
        ry1 = max(0, y1_c + margin_y)
        rx2 = min(w, x2_c - margin_x)
        ry2 = min(h, y2_c - margin_y)

        # Ensure ROI has at least 1 pixel
        if rx2 <= rx1 or ry2 <= ry1:
            cx_clamped = max(0, min(w - 1, cx))
            cy_clamped = max(0, min(h - 1, cy))
            rx1, rx2 = max(0, cx_clamped - 1), min(w, cx_clamped + 2)
            ry1, ry2 = max(0, cy_clamped - 1), min(h, cy_clamped + 2)

        roi = depth_m[ry1:ry2, rx1:rx2]
        total_pixels = int(roi.size)

        # 4. Reject zero, out-of-range, NaN, and Inf depth pixels
        valid_mask = (
            (roi > 0)
            & (roi >= self.min_valid_depth_m)
            & (roi <= self.max_valid_depth_m)
            & ~np.isnan(roi)
            & ~np.isinf(roi)
        )
        valid_vals = roi[valid_mask]
        valid_count = int(len(valid_vals))
        valid_pct = (valid_count / float(total_pixels) * 100.0) if total_pixels > 0 else 0.0

        # 5. Median distance estimation & validity check
        if valid_count > 0:
            median_d = float(np.median(valid_vals))
            if valid_pct >= self.min_valid_depth_percent:
                return FusedDetection(
                    detection=det,
                    distance_m=round(median_d, 3),
                    distance_valid=True,
                    depth_method="median",
                    center_px=(cx, cy),
                    valid_depth_percent=round(valid_pct, 2),
                    detection_id=detection_id,
                    timestamp=timestamp_str or "",
                )
            else:
                # Valid depth percentage below reliability threshold
                return FusedDetection(
                    detection=det,
                    distance_m=round(median_d, 3),
                    distance_valid=False,
                    depth_method="unreliable",
                    center_px=(cx, cy),
                    valid_depth_percent=round(valid_pct, 2),
                    detection_id=detection_id,
                    timestamp=timestamp_str or "",
                )

        # No valid depth found in ROI
        return FusedDetection(
            detection=det,
            distance_m=None,
            distance_valid=False,
            depth_method="invalid",
            center_px=(cx, cy),
            valid_depth_percent=0.0,
            detection_id=detection_id,
            timestamp=timestamp_str or "",
        )

    def _is_valid_depth(self, d: float) -> bool:
        return self.min_valid_depth_m <= d <= self.max_valid_depth_m
