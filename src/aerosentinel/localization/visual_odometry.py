"""
AeroSentinel Visual Odometry Placeholder
==========================================
Visual odometry integration placeholder.

Requires feature-matching or optical-flow implementation.
Feasibility on RPi 4 depends on resolution and frame rate.
See docs/architecture.md for context.

Implementation status: NOT IMPLEMENTED
--------------------------------------
Visual odometry is on the roadmap but has not been implemented yet.
Any attempt to instantiate ``VisualOdometry`` will raise ``NotImplementedError``
to signal clearly that it is not ready for use.

Planned approach (see ``outline()``):
- ORB (Oriented FAST and Rotated BRIEF) feature detection and matching
  between consecutive frames to estimate camera motion.
- Alternatively: Lucas-Kanade sparse optical flow for lower compute.
- Pose integration to produce a local-frame position estimate.
- Feasibility gate: 320x240 resolution, ~5-15 FPS on RPi 4 without hardware
  acceleration. Accuracy TBD — requires real-world evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aerosentinel.localization.localization_interface import LocalizationSource


class VisualOdometry(LocalizationSource):
    """
    Visual odometry integration placeholder.

    Requires feature-matching or optical-flow implementation.
    Feasibility on RPi 4 depends on resolution and frame rate.
    See docs/architecture.md.

    **NOT IMPLEMENTED** — raises ``NotImplementedError`` on all data methods.
    Use ``outline()`` (class method) to inspect the planned implementation approach.
    Use ``NullLocalizationSource`` or ``MAVLinkLocalizationSource`` instead.
    """

    _NOT_IMPL_MSG = (
        "VisualOdometry is not yet implemented. "
        "It is included as a placeholder to show where the integration point "
        "will live. See docs/architecture.md and VisualOdometry.outline() for "
        "the planned implementation approach. "
        "Use NullLocalizationSource or MAVLinkLocalizationSource instead."
    )

    def __init__(self) -> None:
        raise NotImplementedError(self._NOT_IMPL_MSG)

    def get_position(self) -> Dict[str, Optional[float]]:
        raise NotImplementedError(self._NOT_IMPL_MSG)

    def get_orientation(self) -> Dict[str, Optional[float]]:
        raise NotImplementedError(self._NOT_IMPL_MSG)

    def get_velocity(self) -> Dict[str, Optional[float]]:
        raise NotImplementedError(self._NOT_IMPL_MSG)

    def get_confidence(self) -> float:
        raise NotImplementedError(self._NOT_IMPL_MSG)

    def get_status(self) -> Dict[str, Any]:
        raise NotImplementedError(self._NOT_IMPL_MSG)

    @classmethod
    def outline(cls) -> Dict[str, Any]:
        """
        Return a dict describing the planned VO implementation approach.

        Call without instantiation::

            from aerosentinel.localization.visual_odometry import VisualOdometry
            plan = VisualOdometry.outline()
        """
        return {
            "method": "monocular_visual_odometry",
            "algorithms": {
                "primary": {
                    "name": "ORB feature matching",
                    "description": (
                        "Detect ORB keypoints in each frame, match against "
                        "previous frame using BFMatcher (Hamming distance), "
                        "estimate Essential Matrix via RANSAC, decompose into "
                        "R and t, integrate to build trajectory."
                    ),
                    "opencv_api": [
                        "cv2.ORB_create()",
                        "cv2.BFMatcher(cv2.NORM_HAMMING)",
                        "cv2.findEssentialMat()",
                        "cv2.recoverPose()",
                    ],
                },
                "alternative": {
                    "name": "Lucas-Kanade sparse optical flow",
                    "description": (
                        "Track a sparse set of Shi-Tomasi corners across frames "
                        "using cv2.calcOpticalFlowPyrLK(). Faster than ORB on "
                        "RPi 4 but less robust to large rotations."
                    ),
                    "opencv_api": [
                        "cv2.goodFeaturesToTrack()",
                        "cv2.calcOpticalFlowPyrLK()",
                    ],
                },
            },
            "rpi4_feasibility": {
                "assessed": False,
                "notes": (
                    "At 320x240, ORB detection + matching estimated ~40-80 ms on "
                    "RPi 4 (Cortex-A72). Lucas-Kanade is lighter (~10-20 ms). "
                    "Real-world benchmarking required before enabling."
                ),
                "recommended_resolution": "320x240",
                "max_expected_fps_orb": "12-15 (estimated, unverified)",
                "max_expected_fps_lk": "20-30 (estimated, unverified)",
            },
            "dependencies": [
                "opencv-python-headless>=4.8.0",
                "numpy>=1.24.0",
                "picamera2 (for frame acquisition)",
            ],
            "open_questions": [
                "Drift accumulation over long missions — IMU fusion needed.",
                "Scale ambiguity with monocular — RealSense depth can resolve.",
                "Thread safety of frame buffer access.",
                "Integration with LocalizationManager fallback chain.",
            ],
            "references": [
                "docs/architecture.md",
                "HARDWARE_COMPATIBILITY.md",
            ],
        }
