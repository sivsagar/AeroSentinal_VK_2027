"""
obstacle_detector.py — Obstacle detection from depth data for AeroSentinel.

Divides the camera FOV into Left / Centre / Right regions, computes
per-region statistics, and classifies obstacle severity.

DOES NOT output flight commands — data only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class ObstacleStatus(Enum):
    SAFE = auto()
    WARNING = auto()
    CRITICAL = auto()
    UNKNOWN = auto()


@dataclass
class RegionAnalysis:
    region_name: str
    min_distance_m: Optional[float]
    mean_distance_m: Optional[float]
    obstacle_density: float          # fraction [0, 1] of valid pixels
    status: ObstacleStatus


@dataclass
class ObstacleReport:
    timestamp: float
    left: RegionAnalysis
    center: RegionAnalysis
    right: RegionAnalysis
    overall_status: ObstacleStatus
    closest_obstacle_m: Optional[float]
    closest_obstacle_x: Optional[int]
    closest_obstacle_y: Optional[int]
    warning_active: bool


# ---------------------------------------------------------------------------
# ObstacleDetector
# ---------------------------------------------------------------------------

class ObstacleDetector:
    """
    Obstacle detection from float32 depth-in-metres arrays.

    Parameters
    ----------
    left_pct : float
        Fraction of frame width assigned to the left region (default 0.33).
    right_pct : float
        Fraction of frame width assigned to the right region (default 0.33).
        Centre gets the remainder.
    roi_top_pct : float
        Fraction of frame height to skip from the top (default 0.2).
    roi_bottom_pct : float
        Fraction of frame height to skip from the bottom (default 0.1).
    warning_distance_m : float
        Below this distance → WARNING (default 2.0 m).
    critical_distance_m : float
        Below this distance → CRITICAL (default 0.8 m).
    min_density_for_warning : float
        Minimum obstacle_density to consider an obstacle real (default 0.05).
    """

    def __init__(
        self,
        left_pct: float = 0.33,
        right_pct: float = 0.33,
        roi_top_pct: float = 0.2,
        roi_bottom_pct: float = 0.1,
        warning_distance_m: float = 2.0,
        critical_distance_m: float = 0.8,
        min_density_for_warning: float = 0.05,
    ) -> None:
        self.left_pct = left_pct
        self.right_pct = right_pct
        self.roi_top_pct = roi_top_pct
        self.roi_bottom_pct = roi_bottom_pct
        self.warning_distance_m = warning_distance_m
        self.critical_distance_m = critical_distance_m
        self.min_density_for_warning = min_density_for_warning

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, depth_meters: np.ndarray) -> ObstacleReport:
        """
        Analyse a float32 depth frame (metres; 0 = invalid).

        Returns
        -------
        ObstacleReport
        """
        h, w = depth_meters.shape[:2]

        # Apply ROI
        top = int(h * self.roi_top_pct)
        bottom = h - int(h * self.roi_bottom_pct)
        roi = depth_meters[top:bottom, :]

        # Region boundaries
        left_end = int(w * self.left_pct)
        right_start = w - int(w * self.right_pct)

        left_region = roi[:, :left_end]
        center_region = roi[:, left_end:right_start]
        right_region = roi[:, right_start:]

        left_analysis = self._analyze_region("left", left_region)
        center_analysis = self._analyze_region("center", center_region)
        right_analysis = self._analyze_region("right", right_region)

        # Overall status = worst of three
        overall = self._aggregate_status(
            left_analysis.status, center_analysis.status, right_analysis.status
        )

        # Find closest obstacle in full ROI
        closest_m, closest_x, closest_y = self._find_closest(roi, top_offset=top)

        return ObstacleReport(
            timestamp=time.time(),
            left=left_analysis,
            center=center_analysis,
            right=right_analysis,
            overall_status=overall,
            closest_obstacle_m=closest_m,
            closest_obstacle_x=closest_x,
            closest_obstacle_y=closest_y,
            warning_active=overall in (ObstacleStatus.WARNING, ObstacleStatus.CRITICAL),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyze_region(self, name: str, region: np.ndarray) -> RegionAnalysis:
        valid_mask = region > 0
        total = region.size
        valid_count = int(np.count_nonzero(valid_mask))

        if total == 0:
            return RegionAnalysis(
                region_name=name,
                min_distance_m=None,
                mean_distance_m=None,
                obstacle_density=0.0,
                status=ObstacleStatus.UNKNOWN,
            )

        density = valid_count / total

        if valid_count == 0:
            return RegionAnalysis(
                region_name=name,
                min_distance_m=None,
                mean_distance_m=None,
                obstacle_density=density,
                status=ObstacleStatus.UNKNOWN,
            )

        valid_vals = region[valid_mask]
        min_d = float(valid_vals.min())
        mean_d = float(valid_vals.mean())

        status = self._classify(min_d, density)

        return RegionAnalysis(
            region_name=name,
            min_distance_m=min_d,
            mean_distance_m=mean_d,
            obstacle_density=density,
            status=status,
        )

    def _classify(self, min_distance_m: float, density: float) -> ObstacleStatus:
        if density < self.min_density_for_warning:
            return ObstacleStatus.SAFE
        if min_distance_m <= self.critical_distance_m:
            return ObstacleStatus.CRITICAL
        if min_distance_m <= self.warning_distance_m:
            return ObstacleStatus.WARNING
        return ObstacleStatus.SAFE

    @staticmethod
    def _aggregate_status(*statuses: ObstacleStatus) -> ObstacleStatus:
        priority = {
            ObstacleStatus.CRITICAL: 3,
            ObstacleStatus.WARNING: 2,
            ObstacleStatus.SAFE: 1,
            ObstacleStatus.UNKNOWN: 0,
        }
        return max(statuses, key=lambda s: priority.get(s, 0))

    def _find_closest(
        self, roi: np.ndarray, top_offset: int = 0
    ) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        valid_mask = roi > 0
        if not np.any(valid_mask):
            return None, None, None

        min_val = float(roi[valid_mask].min())
        # Find first occurrence of min
        indices = np.argwhere((roi == min_val) & valid_mask)
        if len(indices) == 0:
            return None, None, None

        y_roi, x = indices[0]
        y = y_roi + top_offset
        return min_val, int(x), int(y)
