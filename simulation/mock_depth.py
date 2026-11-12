"""
mock_depth.py — Synthetic depth frame generator and depth scenarios for AeroSentinel.

Provides
--------
* ``generate_depth_frame`` — creates a uint16 depth image with gradient + obstacles + noise
* ``generate_imu_data``    — returns IMUData with sinusoidal motion
* ``DepthScenario``        — enum of preset obstacle configurations
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.aerosentinel.cameras.camera_manager import IMUData


# ---------------------------------------------------------------------------
# DepthScenario
# ---------------------------------------------------------------------------

class DepthScenario(enum.Enum):
    """Preset obstacle configurations for synthetic depth frames."""

    CLEAR = "clear"
    OBSTACLE_LEFT = "obstacle_left"
    OBSTACLE_CENTER = "obstacle_center"
    OBSTACLE_RIGHT = "obstacle_right"
    MULTI_OBSTACLE = "multi_obstacle"


# Obstacle = (centre_x_frac, centre_y_frac, radius_frac, distance_mm)
_SCENARIO_OBSTACLES: dict = {
    DepthScenario.CLEAR: [],
    DepthScenario.OBSTACLE_LEFT: [(0.2, 0.5, 0.15, 800)],
    DepthScenario.OBSTACLE_CENTER: [(0.5, 0.5, 0.2, 600)],
    DepthScenario.OBSTACLE_RIGHT: [(0.8, 0.5, 0.15, 900)],
    DepthScenario.MULTI_OBSTACLE: [
        (0.2, 0.4, 0.12, 700),
        (0.5, 0.6, 0.18, 500),
        (0.75, 0.3, 0.10, 1100),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_depth_frame(
    width: int,
    height: int,
    obstacles: Optional[List[Tuple[float, float, float, int]]] = None,
    scenario: Optional[DepthScenario] = None,
    noise_sigma: float = 15.0,
    background_near_mm: int = 2000,
    background_far_mm: int = 4000,
) -> np.ndarray:
    """
    Generate a synthetic depth frame (uint16, millimetres).

    Parameters
    ----------
    width, height : int
        Frame dimensions in pixels.
    obstacles : list of (cx_frac, cy_frac, radius_frac, distance_mm), optional
        Each obstacle is a filled circle drawn at fractional image coordinates
        with radius as a fraction of image width.
    scenario : DepthScenario, optional
        Convenience preset; overridden by explicit ``obstacles`` argument.
    noise_sigma : float
        Standard deviation of Gaussian depth noise (mm).
    background_near_mm : int
        Depth of the top row of the background gradient.
    background_far_mm : int
        Depth of the bottom row of the background gradient.

    Returns
    -------
    np.ndarray
        uint16 array of shape (height, width).
    """
    # Background: horizontal gradient (top = near, bottom = far)
    gradient = np.linspace(background_near_mm, background_far_mm, height, dtype=np.float32)
    frame = np.tile(gradient[:, np.newaxis], (1, width))

    # Gaussian noise
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, (height, width)).astype(np.float32)
        frame += noise

    # Determine obstacle list
    obs_list = obstacles
    if obs_list is None and scenario is not None:
        obs_list = _SCENARIO_OBSTACLES.get(scenario, [])
    if obs_list is None:
        obs_list = []

    # Draw obstacles as filled circles (Gaussian depth profile around centre)
    yy, xx = np.mgrid[0:height, 0:width]
    for cx_frac, cy_frac, r_frac, dist_mm in obs_list:
        cx = int(cx_frac * width)
        cy = int(cy_frac * height)
        r = max(1, int(r_frac * width))
        sigma_px = r / 2.5
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist_sq <= r ** 2
        # Gaussian depth taper: centre closest, edges transition to background
        blend = np.exp(-dist_sq / (2 * sigma_px ** 2))
        frame[mask] = (
            dist_mm * blend[mask] + frame[mask] * (1 - blend[mask])
        )

    # Clip and cast to uint16
    frame = np.clip(frame, 0, 65535).astype(np.uint16)
    return frame


def generate_imu_data(t: float) -> IMUData:
    """
    Generate synthetic IMUData with sinusoidal motion at time *t*.

    Parameters
    ----------
    t : float
        Elapsed time in seconds (from an arbitrary epoch).

    Returns
    -------
    IMUData
        Simulated sensor reading with gravity on Z axis.
    """
    return IMUData(
        timestamp=time.monotonic(),
        accel_x=0.08 * math.sin(t * 0.7),
        accel_y=0.06 * math.cos(t * 1.1),
        accel_z=9.81 + 0.04 * math.sin(t * 2.0),   # gravity + vibration
        gyro_x=0.012 * math.sin(t * 0.4),
        gyro_y=0.010 * math.cos(t * 0.6),
        gyro_z=0.006 * math.sin(t * 0.3),
        is_valid=True,
    )
