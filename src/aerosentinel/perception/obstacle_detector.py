"""
obstacle_detector.py — Real-Time Depth-Based Obstacle Proximity Assessment System.

Analyses aligned RealSense D435i depth images across configurable spatial sectors
(Left, Centre, Right or 3×3 grid), computes robust percentiles and near-range occupancy,
applies temporal filtering and transition hysteresis, associates tracked persons,
and classifies proximity severity (CLEAR, CAUTION, WARNING, CRITICAL, UNKNOWN).

IMPORTANT SAFETY NOTICE:
This is a perception and advisory subsystem, NOT an autonomous flight-control system.
All thresholds are provisional bench values and MUST be calibrated against drone
dimensions, mounting positions, flight dynamics, and stopping distance before flight.
All coordinates remain in the RealSense camera-relative frame (the cameras are mounted
backwards relative to drone forward flight).
DOES NOT output flight or motor commands.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR color palette for proximity levels
_PROXIMITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "CLEAR": (0, 200, 0),       # Green
    "CAUTION": (0, 220, 255),   # Yellow
    "WARNING": (0, 140, 255),   # Orange
    "CRITICAL": (0, 0, 240),    # Red
    "UNKNOWN": (128, 128, 128), # Grey
}


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class ObstacleStatus(Enum):
    """Proximity classification levels."""

    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    # Backwards compatibility alias for CLEAR
    SAFE = "CLEAR"


# Alias for explicit domain naming
ProximityLevel = ObstacleStatus


@dataclass
class RegionAnalysis:
    """Detailed spatial and depth metrics for an individual image region."""

    region_name: str
    min_distance_m: Optional[float] = None
    mean_distance_m: Optional[float] = None
    obstacle_density: float = 0.0          # fraction [0, 1] of valid pixels
    status: ObstacleStatus = ObstacleStatus.CLEAR

    # Enhanced Proximity Assessment Metrics
    region_id: str = ""
    coordinates: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    valid_pixel_count: int = 0
    total_pixel_count: int = 0
    valid_depth_percent: float = 0.0                       # 0.0 - 100.0%
    median_distance_m: Optional[float] = None
    percentile_10_distance_m: Optional[float] = None
    nearest_reliable_distance_m: Optional[float] = None    # 10th percentile
    near_range_occupancy_percent: float = 0.0              # % of valid pixels <= caution dist
    confidence: float = 0.0                               # 0.0 - 1.0 reliability score
    is_valid: bool = False                                # whether measurement is valid
    human_associated: bool = False                        # overlaps a tracked person
    human_track_ids: List[int] = field(default_factory=list)
    warning_duration_s: float = 0.0                       # duration in current warning state
    last_valid_distance_m: Optional[float] = None          # preserved across dropouts

    def __post_init__(self) -> None:
        if not self.region_id:
            self.region_id = self.region_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert region metrics to a structured dictionary."""
        return {
            "region_id": self.region_id,
            "region_name": self.region_name,
            "coordinates": list(self.coordinates),
            "status": self.status.name,
            "is_valid": self.is_valid,
            "valid_pixel_count": self.valid_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "valid_depth_percent": round(self.valid_depth_percent, 2),
            "median_distance_m": round(self.median_distance_m, 3) if self.median_distance_m is not None else None,
            "percentile_10_distance_m": round(self.percentile_10_distance_m, 3) if self.percentile_10_distance_m is not None else None,
            "nearest_reliable_distance_m": round(self.nearest_reliable_distance_m, 3) if self.nearest_reliable_distance_m is not None else None,
            "min_distance_m": round(self.min_distance_m, 3) if self.min_distance_m is not None else None,
            "mean_distance_m": round(self.mean_distance_m, 3) if self.mean_distance_m is not None else None,
            "obstacle_density": round(self.obstacle_density, 4),
            "near_range_occupancy_percent": round(self.near_range_occupancy_percent, 2),
            "confidence": round(self.confidence, 3),
            "human_associated": self.human_associated,
            "human_track_ids": list(self.human_track_ids),
            "warning_duration_s": round(self.warning_duration_s, 2),
            "last_valid_distance_m": round(self.last_valid_distance_m, 3) if self.last_valid_distance_m is not None else None,
        }


@dataclass
class ObstacleReport:
    """Comprehensive proximity assessment report across all spatial regions."""

    timestamp: float
    left: RegionAnalysis
    center: RegionAnalysis
    right: RegionAnalysis
    overall_status: ObstacleStatus
    closest_obstacle_m: Optional[float] = None
    closest_obstacle_x: Optional[int] = None
    closest_obstacle_y: Optional[int] = None
    warning_active: bool = False

    # Enhanced Proximity Assessment Fields
    grid_layout: str = "1x3"
    regions: Dict[str, RegionAnalysis] = field(default_factory=dict)
    closest_reliable_candidate: Optional[Dict[str, Any]] = None
    caution_active: bool = False
    critical_active: bool = False
    human_occupied_regions: List[str] = field(default_factory=list)
    total_valid_coverage_percent: float = 0.0
    provisional_bench_notice: str = "PROVISIONAL BENCH VALUES — NOT FOR OPERATIONAL FLIGHT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire obstacle report to structured dictionary."""
        iso_ts = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return {
            "timestamp": iso_ts,
            "timestamp_unix": round(self.timestamp, 4),
            "grid_layout": self.grid_layout,
            "overall_status": self.overall_status.name,
            "warning_active": self.warning_active,
            "caution_active": self.caution_active,
            "critical_active": self.critical_active,
            "total_valid_coverage_percent": round(self.total_valid_coverage_percent, 2),
            "closest_obstacle_m": round(self.closest_obstacle_m, 3) if self.closest_obstacle_m is not None else None,
            "closest_obstacle_x": self.closest_obstacle_x,
            "closest_obstacle_y": self.closest_obstacle_y,
            "closest_reliable_candidate": self.closest_reliable_candidate,
            "human_occupied_regions": list(self.human_occupied_regions),
            "safety_notice": self.provisional_bench_notice,
            "left": self.left.to_dict(),
            "center": self.center.to_dict(),
            "right": self.right.to_dict(),
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class _RegionTemporalState:
    """Internal tracker for temporal filtering and state transition hysteresis."""

    smoothed_distance: Optional[float] = None
    current_status: ObstacleStatus = ObstacleStatus.CLEAR
    candidate_status: ObstacleStatus = ObstacleStatus.CLEAR
    candidate_count: int = 0
    state_entered_time: float = 0.0
    consecutive_dropouts: int = 0
    last_valid_distance: Optional[float] = None
    last_valid_time: float = 0.0


# ---------------------------------------------------------------------------
# ObstacleDetector
# ---------------------------------------------------------------------------

class ObstacleDetector:
    """
    Real-Time Depth-Based Obstacle Proximity Assessment System.

    Divides RealSense depth frames into spatial sectors, performs robust
    statistical depth analysis, rejects noisy outliers, applies temporal
    hysteresis and smoothing, and correlates proximity hazards with tracked humans.
    """

    def __init__(
        self,
        left_pct: float = 0.33,
        right_pct: float = 0.33,
        roi_top_pct: float = 0.15,
        roi_bottom_pct: float = 0.10,
        warning_distance_m: float = 2.0,
        critical_distance_m: float = 1.0,
        caution_distance_m: float = 3.5,
        min_density_for_warning: float = 0.05,
        min_valid_coverage_pct: float = 10.0,
        near_range_occupancy_threshold_pct: float = 15.0,
        min_range_m: float = 0.25,
        max_range_m: float = 8.0,
        grid_layout: str = "1x3",
        temporal_smoothing_alpha: float = 0.30,
        hysteresis_escalate_frames: int = 1,
        hysteresis_deescalate_frames: int = 3,
        stale_timeout_s: float = 2.0,
        max_dropout_frames: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}

        # Grid and ROI configuration
        self.grid_layout: str = cfg.get("grid_layout", grid_layout)

        roi_cfg = cfg.get("roi", {})
        self.roi_top_pct: float = float(roi_cfg.get("top_pct", cfg.get("roi_top_pct", roi_top_pct)))
        self.roi_bottom_pct: float = float(roi_cfg.get("bottom_pct", cfg.get("roi_bottom_pct", roi_bottom_pct)))
        self.roi_left_pct: float = float(roi_cfg.get("left_pct", 0.0))
        self.roi_right_pct: float = float(roi_cfg.get("right_pct", 0.0))

        # Sectors for 1x3 layout
        sec_cfg = cfg.get("sectors_1x3", {})
        self.left_pct: float = float(sec_cfg.get("left_pct", cfg.get("left_pct", left_pct)))
        self.right_pct: float = float(sec_cfg.get("right_pct", cfg.get("right_pct", right_pct)))

        # Distance thresholds (Provisional Bench Values)
        dist_cfg = cfg.get("distances", {})
        self.critical_distance_m: float = float(dist_cfg.get("critical_distance_m", cfg.get("critical_distance_m", critical_distance_m)))
        self.warning_distance_m: float = float(dist_cfg.get("warning_distance_m", cfg.get("warning_distance_m", warning_distance_m)))
        self.caution_distance_m: float = float(dist_cfg.get("caution_distance_m", cfg.get("caution_distance_m", caution_distance_m)))

        # Sensor physical range limits
        range_cfg = cfg.get("range_limits", {})
        self.min_range_m: float = float(range_cfg.get("min_range_m", cfg.get("min_range_m", min_range_m)))
        self.max_range_m: float = float(range_cfg.get("max_range_m", cfg.get("max_range_m", max_range_m)))

        # Validity & quality criteria
        qual_cfg = cfg.get("quality", {})
        self.min_valid_coverage_pct: float = float(qual_cfg.get("min_valid_coverage_pct", cfg.get("min_valid_coverage_pct", min_valid_coverage_pct)))
        self.near_range_occupancy_threshold_pct: float = float(qual_cfg.get("near_range_occupancy_threshold_pct", cfg.get("near_range_occupancy_threshold_pct", near_range_occupancy_threshold_pct)))
        self.reliable_depth_percentile: float = float(qual_cfg.get("reliable_depth_percentile", 10.0))
        self.min_density_for_warning: float = float(cfg.get("min_density_for_warning", min_density_for_warning))

        # Temporal stability & hysteresis
        temp_cfg = cfg.get("temporal", {})
        self.temporal_smoothing_alpha: float = float(temp_cfg.get("smoothing_alpha", cfg.get("temporal_smoothing_alpha", temporal_smoothing_alpha)))
        self.hysteresis_escalate_frames: int = int(temp_cfg.get("hysteresis_escalate_frames", cfg.get("hysteresis_escalate_frames", hysteresis_escalate_frames)))
        self.hysteresis_deescalate_frames: int = int(temp_cfg.get("hysteresis_deescalate_frames", cfg.get("hysteresis_deescalate_frames", hysteresis_deescalate_frames)))
        self.stale_timeout_s: float = float(temp_cfg.get("stale_timeout_s", cfg.get("stale_timeout_s", stale_timeout_s)))
        self.max_dropout_frames: int = int(temp_cfg.get("max_dropout_frames", cfg.get("max_dropout_frames", max_dropout_frames)))

        # Temporal state storage per region
        self._temporal_state: Dict[str, _RegionTemporalState] = {}

    def reset_temporal_state(self) -> None:
        """Clear all temporal smoothing and hysteresis history."""
        self._temporal_state.clear()

    # ------------------------------------------------------------------
    # Public Analysis API
    # ------------------------------------------------------------------

    def analyze(
        self,
        depth_meters: np.ndarray,
        tracks: Optional[List[Any]] = None,
        timestamp: Optional[float] = None,
    ) -> ObstacleReport:
        """
        Analyse depth image, perform spatial grid statistics, and classify proximity.

        Parameters
        ----------
        depth_meters : np.ndarray
            Depth frame. If dtype is uint16, interpreted as millimetres and converted.
            If float32, interpreted as metres (0 = invalid/missing).
        tracks : list, optional
            List of active TrackedPerson or FusedDetection objects for human association.
        timestamp : float, optional
            Current frame timestamp in seconds (default time.time()).

        Returns
        -------
        ObstacleReport
        """
        now = timestamp if timestamp is not None else time.time()

        # Normalise input array to float32 metres
        if depth_meters.dtype == np.uint16:
            depth_m = depth_meters.astype(np.float32) / 1000.0
        else:
            depth_m = depth_meters.astype(np.float32, copy=False)

        h, w = depth_m.shape[:2]

        # Calculate Region of Interest (ROI) boundaries
        top = int(h * self.roi_top_pct)
        bottom = max(top + 1, h - int(h * self.roi_bottom_pct))
        roi_left = int(w * self.roi_left_pct)
        roi_right = max(roi_left + 1, w - int(w * self.roi_right_pct))

        # Partition spatial regions
        region_defs: Dict[str, Tuple[int, int, int, int]] = {}
        if self.grid_layout == "3x3":
            # 3 rows × 3 columns
            row_h = (bottom - top) // 3
            col_w = (roi_right - roi_left) // 3
            row_names = ["top", "mid", "bottom"]
            col_names = ["left", "center", "right"]
            for r_idx, r_name in enumerate(row_names):
                y1 = top + r_idx * row_h
                y2 = bottom if r_idx == 2 else y1 + row_h
                for c_idx, c_name in enumerate(col_names):
                    x1 = roi_left + c_idx * col_w
                    x2 = roi_right if c_idx == 2 else x1 + col_w
                    region_defs[f"{r_name}_{c_name}"] = (x1, y1, x2, y2)
        else:
            # Standard 1x3 layout: Left, Center, Right
            left_end = int(w * self.left_pct)
            right_start = w - int(w * self.right_pct)
            region_defs["left"] = (0, top, left_end, bottom)
            region_defs["center"] = (left_end, top, right_start, bottom)
            region_defs["right"] = (right_start, top, w, bottom)

        # Process each region
        regions: Dict[str, RegionAnalysis] = {}
        total_valid_pixels_roi = 0
        total_pixels_roi = 0

        for r_name, (x1, y1, x2, y2) in region_defs.items():
            r_patch = depth_m[y1:y2, x1:x2]
            r_analysis = self._analyze_region_patch(
                name=r_name,
                patch=r_patch,
                coords=(x1, y1, x2, y2),
                tracks=tracks,
                now=now,
            )
            regions[r_name] = r_analysis
            total_valid_pixels_roi += r_analysis.valid_pixel_count
            total_pixels_roi += r_analysis.total_pixel_count

        total_cov_pct = (total_valid_pixels_roi / total_pixels_roi * 100.0) if total_pixels_roi > 0 else 0.0

        # Backwards compatible direct sector access
        if self.grid_layout == "3x3":
            left_analysis = regions.get("mid_left", list(regions.values())[0])
            center_analysis = regions.get("mid_center", list(regions.values())[1])
            right_analysis = regions.get("mid_right", list(regions.values())[2])
        else:
            left_analysis = regions["left"]
            center_analysis = regions["center"]
            right_analysis = regions["right"]

        # Aggregate overall status (worst of all regions)
        overall_status = self._aggregate_status(*(r.status for r in regions.values()))

        # Find closest reliable proximity candidate across ROI
        full_roi = depth_m[top:bottom, roi_left:roi_right]
        closest_m, closest_x, closest_y, cand_info = self._find_closest_reliable_candidate(
            roi=full_roi,
            top_offset=top,
            left_offset=roi_left,
            regions=regions,
        )

        warning_active = overall_status in (ObstacleStatus.WARNING, ObstacleStatus.CRITICAL)
        caution_active = overall_status == ObstacleStatus.CAUTION
        critical_active = overall_status == ObstacleStatus.CRITICAL

        human_regions = [r.region_id for r in regions.values() if r.human_associated]

        return ObstacleReport(
            timestamp=now,
            left=left_analysis,
            center=center_analysis,
            right=right_analysis,
            overall_status=overall_status,
            closest_obstacle_m=closest_m,
            closest_obstacle_x=closest_x,
            closest_obstacle_y=closest_y,
            warning_active=warning_active,
            grid_layout=self.grid_layout,
            regions=regions,
            closest_reliable_candidate=cand_info,
            caution_active=caution_active,
            critical_active=critical_active,
            human_occupied_regions=human_regions,
            total_valid_coverage_percent=total_cov_pct,
        )

    # ------------------------------------------------------------------
    # Region Analysis & Filtering Internals
    # ------------------------------------------------------------------

    def _analyze_region_patch(
        self,
        name: str,
        patch: np.ndarray,
        coords: Tuple[int, int, int, int],
        tracks: Optional[List[Any]],
        now: float,
    ) -> RegionAnalysis:
        x1, y1, x2, y2 = coords
        total_pixels = int(patch.size)

        # Check human association with this region
        human_associated, human_ids = self._check_human_association(coords, tracks)

        # Get or create temporal state
        if name not in self._temporal_state:
            self._temporal_state[name] = _RegionTemporalState(
                current_status=ObstacleStatus.CLEAR,
                state_entered_time=now,
            )
        temp_state = self._temporal_state[name]

        if total_pixels == 0:
            return RegionAnalysis(
                region_name=name,
                region_id=name,
                coordinates=coords,
                status=ObstacleStatus.UNKNOWN,
            )

        # Strict depth validation filter:
        # 1. > 0 (reject zero depth / missing return)
        # 2. isfinite (reject NaN and inf)
        # 3. >= min_range_m (reject near-field sensor noise)
        # 4. <= max_range_m (reject out-of-range returns)
        valid_mask = (
            (patch > 0)
            & np.isfinite(patch)
            & (patch >= self.min_range_m)
            & (patch <= self.max_range_m)
        )
        valid_count = int(np.count_nonzero(valid_mask))
        valid_depth_pct = (valid_count / total_pixels) * 100.0
        density = valid_count / total_pixels

        # Insufficient coverage check
        is_sufficient_coverage = valid_depth_pct >= self.min_valid_coverage_pct

        if not is_sufficient_coverage:
            raw_status = ObstacleStatus.UNKNOWN
            min_d = mean_d = median_d = p10_d = nearest_reliable_d = None
            near_occupancy_pct = 0.0
            confidence = 0.0
            is_valid_measurement = False
        else:
            is_valid_measurement = True
            valid_vals = patch[valid_mask]
            min_d = float(valid_vals.min())
            mean_d = float(valid_vals.mean())
            median_d = float(np.median(valid_vals))
            p10_d = float(np.percentile(valid_vals, 10))
            nearest_reliable_d = float(np.percentile(valid_vals, self.reliable_depth_percentile))

            # Near-range occupancy: percentage of valid pixels within caution/warning zone
            near_pixels = np.count_nonzero(valid_vals <= self.caution_distance_m)
            near_occupancy_pct = float(near_pixels / valid_count * 100.0)

            # Measurement confidence calculation
            cov_score = min(1.0, valid_depth_pct / 50.0)
            depth_std = float(np.std(valid_vals))
            consistency_score = max(0.1, 1.0 - min(1.0, depth_std / 3.0))
            confidence = round(float(cov_score * consistency_score), 3)

            # Raw proximity classification based on reliable depth
            raw_status = self._classify_raw(nearest_reliable_d, near_occupancy_pct, density)

        # Apply temporal smoothing and transition hysteresis
        filtered_status, smoothed_dist, warning_duration = self._apply_temporal_filter(
            temp_state=temp_state,
            raw_status=raw_status,
            raw_distance=nearest_reliable_d,
            is_valid=is_valid_measurement,
            now=now,
        )

        return RegionAnalysis(
            region_name=name,
            region_id=name,
            coordinates=coords,
            status=filtered_status,
            is_valid=is_valid_measurement,
            min_distance_m=min_d,
            mean_distance_m=mean_d,
            median_distance_m=median_d,
            percentile_10_distance_m=p10_d,
            nearest_reliable_distance_m=nearest_reliable_d,
            valid_pixel_count=valid_count,
            total_pixel_count=total_pixels,
            valid_depth_percent=valid_depth_pct,
            obstacle_density=density,
            near_range_occupancy_percent=near_occupancy_pct,
            confidence=confidence,
            human_associated=human_associated,
            human_track_ids=human_ids,
            warning_duration_s=warning_duration,
            last_valid_distance_m=temp_state.last_valid_distance,
        )

    def _classify_raw(
        self, nearest_reliable_m: float, near_occupancy_pct: float, density: float
    ) -> ObstacleStatus:
        """Classify proximity severity from reliable depth and density."""
        if density < self.min_density_for_warning:
            return ObstacleStatus.CLEAR

        # Near-range occupancy confirms real physical objects vs occasional edge scatter
        is_sustained_return = (
            near_occupancy_pct >= self.near_range_occupancy_threshold_pct
            or density >= 0.50
        )

        if nearest_reliable_m <= self.critical_distance_m and is_sustained_return:
            return ObstacleStatus.CRITICAL
        if nearest_reliable_m <= self.warning_distance_m and is_sustained_return:
            return ObstacleStatus.WARNING
        if nearest_reliable_m <= self.caution_distance_m:
            return ObstacleStatus.CAUTION
        return ObstacleStatus.CLEAR

    def _apply_temporal_filter(
        self,
        temp_state: _RegionTemporalState,
        raw_status: ObstacleStatus,
        raw_distance: Optional[float],
        is_valid: bool,
        now: float,
    ) -> Tuple[ObstacleStatus, Optional[float], float]:
        """Apply EMA smoothing, state transition hysteresis, and dropout handling."""
        severity_rank = {
            ObstacleStatus.CRITICAL: 4,
            ObstacleStatus.WARNING: 3,
            ObstacleStatus.CAUTION: 2,
            ObstacleStatus.CLEAR: 1,
            ObstacleStatus.UNKNOWN: 0,
        }

        if is_valid and raw_distance is not None:
            temp_state.consecutive_dropouts = 0
            temp_state.last_valid_distance = raw_distance
            temp_state.last_valid_time = now

            # Distance EMA smoothing
            if temp_state.smoothed_distance is None:
                temp_state.smoothed_distance = raw_distance
            else:
                alpha = self.temporal_smoothing_alpha
                temp_state.smoothed_distance = (
                    alpha * raw_distance + (1.0 - alpha) * temp_state.smoothed_distance
                )

            curr_rank = severity_rank.get(temp_state.current_status, 1)
            raw_rank = severity_rank.get(raw_status, 1)

            if raw_status == temp_state.current_status:
                temp_state.candidate_status = raw_status
                temp_state.candidate_count = 0
            elif raw_rank > curr_rank:
                # Escalation
                if raw_status == ObstacleStatus.CRITICAL or self.hysteresis_escalate_frames <= 1:
                    temp_state.current_status = raw_status
                    temp_state.candidate_status = raw_status
                    temp_state.candidate_count = 0
                    temp_state.state_entered_time = now
                else:
                    if temp_state.candidate_status == raw_status:
                        temp_state.candidate_count += 1
                    else:
                        temp_state.candidate_status = raw_status
                        temp_state.candidate_count = 1

                    if temp_state.candidate_count >= self.hysteresis_escalate_frames:
                        temp_state.current_status = raw_status
                        temp_state.state_entered_time = now
                        temp_state.candidate_count = 0
            else:
                # De-escalation: requires sustained confirmation
                if temp_state.candidate_status == raw_status:
                    temp_state.candidate_count += 1
                else:
                    temp_state.candidate_status = raw_status
                    temp_state.candidate_count = 1

                if temp_state.candidate_count >= self.hysteresis_deescalate_frames:
                    temp_state.current_status = raw_status
                    temp_state.state_entered_time = now
                    temp_state.candidate_count = 0

            # Calculate active warning duration
            if temp_state.current_status in (ObstacleStatus.CAUTION, ObstacleStatus.WARNING, ObstacleStatus.CRITICAL):
                duration = max(0.0, now - temp_state.state_entered_time)
            else:
                duration = 0.0

            return temp_state.current_status, temp_state.smoothed_distance, duration

        else:
            # Depth dropout / invalid measurement
            temp_state.consecutive_dropouts += 1
            time_since_valid = (now - temp_state.last_valid_time) if temp_state.last_valid_time > 0 else 999.0

            if temp_state.consecutive_dropouts <= self.max_dropout_frames and time_since_valid < self.stale_timeout_s:
                # Tolerated temporary dropout: retain previous state and distance
                duration = max(0.0, now - temp_state.state_entered_time) if temp_state.current_status in (
                    ObstacleStatus.CAUTION, ObstacleStatus.WARNING, ObstacleStatus.CRITICAL
                ) else 0.0
                return temp_state.current_status, temp_state.smoothed_distance, duration
            else:
                # Stale dropout exceeded: transition to UNKNOWN, clear warning duration
                temp_state.current_status = ObstacleStatus.UNKNOWN
                temp_state.smoothed_distance = None
                return ObstacleStatus.UNKNOWN, None, 0.0

    def _check_human_association(
        self,
        coords: Tuple[int, int, int, int],
        tracks: Optional[List[Any]],
    ) -> Tuple[bool, List[int]]:
        """Check if any tracked person overlaps the region coordinates."""
        if not tracks:
            return False, []

        x1, y1, x2, y2 = coords
        human_ids: List[int] = []

        for trk in tracks:
            track_id = getattr(trk, "track_id", getattr(trk, "detection_id", None))
            bbox = getattr(trk, "bbox", None)
            center = getattr(trk, "center_px", None)

            if isinstance(trk, dict):
                track_id = trk.get("track_id", trk.get("detection_id", None))
                bbox = trk.get("bbox")
                center = trk.get("center_px", trk.get("center_coordinates"))

            if center is not None:
                cx, cy = center
                if x1 <= cx < x2 and y1 <= cy < y2:
                    if track_id is not None and track_id not in human_ids:
                        human_ids.append(track_id)
                    continue

            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                inter_x1 = max(x1, bx1)
                inter_y1 = max(y1, by1)
                inter_x2 = min(x2, bx2)
                inter_y2 = min(y2, by2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    box_area = max(1, (bx2 - bx1) * (by2 - by1))
                    if inter_area / box_area > 0.15:
                        if track_id is not None and track_id not in human_ids:
                            human_ids.append(track_id)

        return (len(human_ids) > 0), human_ids

    def _find_closest_reliable_candidate(
        self,
        roi: np.ndarray,
        top_offset: int,
        left_offset: int,
        regions: Dict[str, RegionAnalysis],
    ) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[Dict[str, Any]]]:
        """Find the closest depth return in the ROI and map to its containing region."""
        valid_mask = (
            (roi > 0)
            & np.isfinite(roi)
            & (roi >= self.min_range_m)
            & (roi <= self.max_range_m)
        )
        if not np.any(valid_mask):
            return None, None, None, None

        min_val = float(roi[valid_mask].min())
        indices = np.argwhere((roi == min_val) & valid_mask)
        if len(indices) == 0:
            return None, None, None, None

        y_roi, x_roi = indices[0]
        cand_y = int(y_roi + top_offset)
        cand_x = int(x_roi + left_offset)

        # Identify which region contains this candidate
        cand_region = "center"
        for r_name, r_analysis in regions.items():
            rx1, ry1, rx2, ry2 = r_analysis.coordinates
            if rx1 <= cand_x < rx2 and ry1 <= cand_y < ry2:
                cand_region = r_name
                break

        cand_info = {
            "distance_m": round(min_val, 3),
            "pixel_x": cand_x,
            "pixel_y": cand_y,
            "region": cand_region,
        }
        return min_val, cand_x, cand_y, cand_info

    # ------------------------------------------------------------------
    # Backwards Compatibility API
    # ------------------------------------------------------------------

    def _classify(self, min_distance_m: float, density: float) -> ObstacleStatus:
        """Legacy helper for tests."""
        return self._classify_raw(min_distance_m, 100.0 if density > 0 else 0.0, density)

    @staticmethod
    def _aggregate_status(*statuses: ObstacleStatus) -> ObstacleStatus:
        """Aggregate multiple regional statuses into the overall worst status."""
        priority = {
            ObstacleStatus.CRITICAL: 4,
            ObstacleStatus.WARNING: 3,
            ObstacleStatus.CAUTION: 2,
            ObstacleStatus.CLEAR: 1,
            ObstacleStatus.UNKNOWN: 0,
        }
        return max(statuses, key=lambda s: priority.get(s, 0))

    def _find_closest(
        self, roi: np.ndarray, top_offset: int = 0
    ) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Legacy find closest for tests."""
        valid_mask = (roi > 0) & np.isfinite(roi) & (roi >= self.min_range_m) & (roi <= self.max_range_m)
        if not np.any(valid_mask):
            return None, None, None

        min_val = float(roi[valid_mask].min())
        indices = np.argwhere((roi == min_val) & valid_mask)
        if len(indices) == 0:
            return None, None, None

        y_roi, x = indices[0]
        y = y_roi + top_offset
        return min_val, int(x), int(y)

    # ------------------------------------------------------------------
    # Visualisation Methods
    # ------------------------------------------------------------------

    def draw_proximity_overlay(
        self,
        rgb_frame: np.ndarray,
        report: ObstacleReport,
        alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Draw spatial region divisions, distance measurements, status labels,
        human badges, and closest candidate reticles on RealSense RGB frame.

        Parameters
        ----------
        rgb_frame : np.ndarray
            RealSense RGB frame (BGR format).
        report : ObstacleReport
            Obstacle proximity report.
        alpha : float
            Semi-transparent overlay opacity (default 0.35).

        Returns
        -------
        np.ndarray
            Annotated BGR frame.
        """
        annotated = rgb_frame.copy()
        h, w = annotated.shape[:2]
        overlay = annotated.copy()

        # 1. Fill regions with semi-transparent status colors
        for r_name, r_info in report.regions.items():
            x1, y1, x2, y2 = r_info.coordinates
            color = _PROXIMITY_COLORS.get(r_info.status.name, (128, 128, 128))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

        # 2. Draw crisp region boundaries, HUD text, and status tags
        for r_name, r_info in report.regions.items():
            x1, y1, x2, y2 = r_info.coordinates
            color = _PROXIMITY_COLORS.get(r_info.status.name, (128, 128, 128))

            # Region bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Header badge
            hdr_text = f"[{r_info.region_id.upper()}] {r_info.status.name}"
            cv2.putText(
                annotated,
                hdr_text,
                (x1 + 10, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Distance information
            if r_info.is_valid and r_info.median_distance_m is not None:
                dist_text = f"Med: {r_info.median_distance_m:.2f}m | P10: {r_info.percentile_10_distance_m:.2f}m"
            elif r_info.last_valid_distance_m is not None:
                dist_text = f"Last: {r_info.last_valid_distance_m:.2f}m (Stale)"
            else:
                dist_text = "Dist: N/A (Insufficient Data)"

            cv2.putText(
                annotated,
                dist_text,
                (x1 + 10, y1 + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            # Coverage & confidence
            cov_text = f"Cov: {r_info.valid_depth_percent:.1f}% | Conf: {r_info.confidence:.2f}"
            cv2.putText(
                annotated,
                cov_text,
                (x1 + 10, y1 + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Human association badge
            if r_info.human_associated:
                tracks_str = ",".join(str(tid) for tid in r_info.human_track_ids[:3])
                human_text = f"HUMAN: #{tracks_str}"
                cv2.rectangle(annotated, (x1 + 8, y1 + 76), (x1 + 180, y1 + 96), (200, 0, 200), -1)
                cv2.putText(
                    annotated,
                    human_text,
                    (x1 + 12, y1 + 91),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Draw closest reliable proximity candidate reticle
        if report.closest_reliable_candidate:
            cx = report.closest_reliable_candidate["pixel_x"]
            cy = report.closest_reliable_candidate["pixel_y"]
            cdist = report.closest_reliable_candidate["distance_m"]
            creg = report.closest_reliable_candidate["region"]

            # Reticle / crosshair
            cand_color = (0, 0, 255) if report.critical_active else ((0, 140, 255) if report.warning_active else (0, 255, 255))
            cv2.circle(annotated, (cx, cy), 16, cand_color, 2)
            cv2.circle(annotated, (cx, cy), 4, cand_color, -1)
            cv2.line(annotated, (cx - 24, cy), (cx + 24, cy), cand_color, 1)
            cv2.line(annotated, (cx, cy - 24), (cx, cy + 24), cand_color, 1)

            reticle_label = f"CLOSEST: {cdist:.2f}m ({creg.upper()})"
            cv2.rectangle(annotated, (cx + 18, cy - 14), (cx + 18 + len(reticle_label) * 8, cy + 8), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                reticle_label,
                (cx + 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 4. Top Status & Advisory Banner
        cv2.rectangle(annotated, (0, 0), (w, 32), (20, 20, 20), -1)
        ovr_color = _PROXIMITY_COLORS.get(report.overall_status.name, (200, 200, 200))

        closest_str = f"{report.closest_obstacle_m:.2f}m" if report.closest_obstacle_m is not None else "N/A"
        banner_text = (
            f"AeroSentinel Proximity Advisory | Status: {report.overall_status.name} | "
            f"Closest: {closest_str} | BENCH TEST VALUES ONLY"
        )
        cv2.putText(
            annotated,
            banner_text,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            ovr_color,
            2,
            cv2.LINE_AA,
        )

        return annotated

    def create_proximity_depth_visualization(
        self,
        depth_meters: np.ndarray,
        report: ObstacleReport,
    ) -> np.ndarray:
        """
        Generate colorized depth image highlighting near-range regions and spatial grid.

        Parameters
        ----------
        depth_meters : np.ndarray
            Depth frame in metres.
        report : ObstacleReport
            Obstacle report.

        Returns
        -------
        np.ndarray
            Colorized BGR depth visualization.
        """
        if depth_meters.dtype == np.uint16:
            depth_m = depth_meters.astype(np.float32) / 1000.0
        else:
            depth_m = depth_meters.astype(np.float32, copy=False)

        h, w = depth_m.shape[:2]

        # Valid depth range normalization [min_range_m, max_range_m] -> [0, 255]
        valid_mask = (depth_m >= self.min_range_m) & (depth_m <= self.max_range_m)
        depth_norm = np.zeros((h, w), dtype=np.uint8)

        if np.any(valid_mask):
            scaled = (depth_m[valid_mask] - self.min_range_m) / max(0.1, (self.max_range_m - self.min_range_m))
            scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
            depth_norm[valid_mask] = scaled

        # Color map (COLORMAP_TURBO provides high contrast for near vs far)
        color_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        # Invalid pixels colored dark slate
        color_depth[~valid_mask] = (40, 40, 40)

        # Draw grid outlines and status overlays
        for r_name, r_info in report.regions.items():
            x1, y1, x2, y2 = r_info.coordinates
            color = _PROXIMITY_COLORS.get(r_info.status.name, (128, 128, 128))
            cv2.rectangle(color_depth, (x1, y1), (x2, y2), color, 2)

            label = f"{r_info.region_id.upper()}: {r_info.status.name}"
            cv2.putText(
                color_depth,
                label,
                (x1 + 10, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return color_depth

    @staticmethod
    def save_annotated_frame(filepath: Union[str, Path], frame: np.ndarray) -> bool:
        """Save an annotated image to disk, ensuring directory existence."""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            return cv2.imwrite(str(path), frame)
        except Exception as exc:
            logger.warning("Failed to save annotated frame to %s: %s", filepath, exc)
            return False
