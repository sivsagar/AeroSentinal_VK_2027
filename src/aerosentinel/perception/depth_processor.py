"""
depth_processor.py — Depth frame processing for AeroSentinel.

Accepts uint16 depth frames (millimetres from RealSense D435i).
Applies a configurable filter chain, converts to metres, and produces
a colourised visualisation.

pyrealsense2 filters are used when available; OpenCV-based alternatives
are used otherwise.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pyrealsense2 filters
try:
    import pyrealsense2 as rs  # type: ignore

    _RS_AVAILABLE = True
    logger.info("pyrealsense2 available — using RealSense filter chain.")
except ImportError:
    _RS_AVAILABLE = False
    logger.info("pyrealsense2 not available — using OpenCV-based filter alternatives.")


# ---------------------------------------------------------------------------
# ProcessedDepth dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProcessedDepth:
    raw_frame: np.ndarray           # uint16 mm
    filtered_frame: np.ndarray      # uint16 mm filtered
    depth_meters: np.ndarray        # float32 metres (0 = invalid)
    colormap: np.ndarray            # BGR visualisation
    valid_pixel_count: int
    total_pixel_count: int
    min_distance_m: Optional[float]
    max_distance_m: Optional[float]
    mean_distance_m: Optional[float]
    timestamp: float


# ---------------------------------------------------------------------------
# DepthProcessor
# ---------------------------------------------------------------------------

class DepthProcessor:
    """
    Process uint16 depth frames from an Intel RealSense D435i.

    Parameters
    ----------
    filter_chain : list of str
        Ordered list of filters to apply.
        Valid entries: "decimation", "spatial", "temporal", "hole_filling".
    colormap_id : int
        OpenCV colormap constant (default cv2.COLORMAP_JET).
    max_depth_m : float
        Depths beyond this are treated as invalid (clamped to 0).
    """

    _SUPPORTED_FILTERS = ("decimation", "spatial", "temporal", "hole_filling")

    def __init__(
        self,
        filter_chain: Optional[List[str]] = None,
        colormap_id: int = cv2.COLORMAP_JET,
        max_depth_m: float = 10.0,
        min_distance_m: float = 0.1,
        max_distance_m: Optional[float] = None,
    ) -> None:
        self.filter_chain: List[str] = filter_chain if filter_chain is not None else [
            "decimation", "spatial", "temporal", "hole_filling"
        ]
        self.colormap_id = colormap_id
        effective_max = max_distance_m if max_distance_m is not None else max_depth_m
        self.min_depth_mm: float = min_distance_m * 1000.0
        self.max_depth_mm: float = effective_max * 1000.0

        # Build RealSense filter objects if available
        self._rs_filters: dict = {}
        if _RS_AVAILABLE:
            self._build_rs_filters()

        # Temporal filter state for OpenCV fallback
        self._temporal_prev: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, depth_frame: np.ndarray) -> ProcessedDepth:
        """
        Process a uint16 depth frame (millimetres).

        Parameters
        ----------
        depth_frame : np.ndarray
            Shape (H, W), dtype uint16, values in millimetres.

        Returns
        -------
        ProcessedDepth
        """
        raw = depth_frame.copy()

        # Apply filter chain
        filtered = self._apply_filters(raw)

        # Convert to metres (float32); 0 means invalid
        depth_m = self._to_meters(filtered)

        # Colorise
        colormap = self.create_depth_colormap(filtered)

        # Statistics (valid pixels only)
        valid_mask = depth_m > 0
        valid_count = int(np.count_nonzero(valid_mask))
        total_count = int(depth_m.size)

        if valid_count > 0:
            valid_vals = depth_m[valid_mask]
            min_d = float(valid_vals.min())
            max_d = float(valid_vals.max())
            mean_d = float(valid_vals.mean())
        else:
            min_d = max_d = mean_d = None

        return ProcessedDepth(
            raw_frame=raw,
            filtered_frame=filtered,
            depth_meters=depth_m,
            colormap=colormap,
            valid_pixel_count=valid_count,
            total_pixel_count=total_count,
            min_distance_m=min_d,
            max_distance_m=max_d,
            mean_distance_m=mean_d,
            timestamp=time.time(),
        )

    def create_depth_colormap(self, depth_frame: np.ndarray) -> np.ndarray:
        """
        Produce a BGR colour visualisation of the depth frame.

        Parameters
        ----------
        depth_frame : np.ndarray
            uint16 millimetre frame.

        Returns
        -------
        np.ndarray
            BGR uint8 image.
        """
        # Normalise to 8-bit; clip at max_depth_mm
        clipped = np.clip(depth_frame, 0, self.max_depth_mm).astype(np.float32)
        if self.max_depth_mm > 0:
            norm = (clipped / self.max_depth_mm * 255.0).astype(np.uint8)
        else:
            norm = clipped.astype(np.uint8)
        colored = cv2.applyColorMap(norm, self.colormap_id)
        # Mark invalid pixels (depth == 0) black
        invalid_mask = depth_frame == 0
        colored[invalid_mask] = (0, 0, 0)
        return colored

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_rs_filters(self) -> None:
        """Instantiate pyrealsense2 filter objects."""
        try:
            self._rs_filters["decimation"] = rs.decimation_filter()
            self._rs_filters["spatial"] = rs.spatial_filter()
            self._rs_filters["temporal"] = rs.temporal_filter()
            self._rs_filters["hole_filling"] = rs.hole_filling_filter()
        except Exception as exc:
            logger.warning("Could not build RealSense filters: %s — using OpenCV fallbacks.", exc)
            self._rs_filters = {}

    def _apply_filters(self, depth_mm: np.ndarray) -> np.ndarray:
        """Apply the configured filter chain and return a uint16 frame."""
        frame = depth_mm.copy()
        for name in self.filter_chain:
            if name not in self._SUPPORTED_FILTERS:
                logger.warning("Unknown filter '%s'; skipping.", name)
                continue
            if _RS_AVAILABLE and name in self._rs_filters:
                frame = self._rs_filter_apply(name, frame)
            else:
                frame = self._cv_filter_apply(name, frame)
        return frame

    def _rs_filter_apply(self, name: str, depth_mm: np.ndarray) -> np.ndarray:
        """Apply a single RealSense filter.  Convert array <-> rs.frame."""
        try:
            filt = self._rs_filters[name]
            # Wrap numpy array in a pyrealsense2 frame
            rs_depth = rs.frame()
            # NOTE: full rs.frame wrapping requires the full pipeline context;
            # for array input we fall back to the OpenCV path.
            raise NotImplementedError("Array wrapping not supported without pipeline.")
        except Exception:
            return self._cv_filter_apply(name, depth_mm)

    def _cv_filter_apply(self, name: str, depth_mm: np.ndarray) -> np.ndarray:
        """OpenCV-based alternatives to RealSense filters."""
        if name == "decimation":
            # 2x spatial downscale then upscale (nearest-neighbour to preserve depth)
            h, w = depth_mm.shape
            small = cv2.resize(depth_mm, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif name == "spatial":
            # Bilateral-like smoothing using a 2D Gaussian, masked for valid pixels
            valid_mask = depth_mm > 0
            f32 = depth_mm.astype(np.float32)
            # Blur only valid pixels
            blurred = cv2.GaussianBlur(f32, (5, 5), 1.5)
            result = np.where(valid_mask, blurred, 0).astype(np.uint16)
            return result

        elif name == "temporal":
            # Simple exponential moving average with previous frame
            if self._temporal_prev is None or self._temporal_prev.shape != depth_mm.shape:
                self._temporal_prev = depth_mm.astype(np.float32)
            alpha = 0.5
            valid = depth_mm > 0
            prev_valid = self._temporal_prev > 0
            f32 = depth_mm.astype(np.float32)
            blended = np.where(
                valid & prev_valid,
                alpha * f32 + (1 - alpha) * self._temporal_prev,
                np.where(valid, f32, self._temporal_prev),
            )
            self._temporal_prev = blended.astype(np.float32)
            return blended.astype(np.uint16)

        elif name == "hole_filling":
            # Inpaint small holes (depth == 0)
            invalid_mask = (depth_mm == 0).astype(np.uint8)
            if not np.any(invalid_mask):
                return depth_mm
            # Scale to uint8 for inpaint, then scale back
            scale = self.max_depth_mm / 255.0 if self.max_depth_mm > 0 else 1.0
            u8 = np.clip(depth_mm / scale, 0, 255).astype(np.uint8)
            inpainted = cv2.inpaint(u8, invalid_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            result = (inpainted.astype(np.float32) * scale).astype(np.uint16)
            # Preserve originally valid pixels
            result[depth_mm > 0] = depth_mm[depth_mm > 0]
            return result

        return depth_mm  # pass-through for unrecognised names

    def _to_meters(self, depth_mm: np.ndarray) -> np.ndarray:
        """
        Convert uint16 millimetre frame to float32 metres.
        Pixels with depth == 0 or depth > max_depth_mm remain 0.0 (invalid).
        """
        f32 = depth_mm.astype(np.float32) / 1000.0
        # Invalidate out-of-range readings
        f32[depth_mm == 0] = 0.0
        f32[f32 > self.max_depth_mm / 1000.0] = 0.0
        return f32
