"""
AeroSentinel SLAM Adapter Placeholder
=======================================
SLAM integration placeholder.

Compatible engines: ORB-SLAM3, RTAB-Map.
Computational feasibility on RPi 4 must be evaluated before enabling.
See HARDWARE_COMPATIBILITY.md.

Implementation status: NOT IMPLEMENTED
--------------------------------------
Full SLAM on RPi 4 is computationally demanding. This module holds the
integration point for when a viable SLAM solution is identified and validated.

``is_feasible_on_rpi4()`` currently returns ``False`` (pending evaluation).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aerosentinel.localization.localization_interface import LocalizationSource


class SLAMAdapter(LocalizationSource):
    """
    SLAM integration placeholder.

    Compatible engines: ORB-SLAM3, RTAB-Map.
    Computational feasibility on RPi 4 must be evaluated before enabling.
    See HARDWARE_COMPATIBILITY.md.

    **NOT IMPLEMENTED** — raises ``NotImplementedError`` on all data methods.
    Call ``is_feasible_on_rpi4()`` (class method) for the current assessment.
    """

    _NOT_IMPL_MSG = (
        "SLAMAdapter is not yet implemented. "
        "Full SLAM (ORB-SLAM3, RTAB-Map) is computationally demanding and "
        "has not been validated on RPi 4B. "
        "is_feasible_on_rpi4() returns False pending evaluation. "
        "See HARDWARE_COMPATIBILITY.md for context."
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
    def is_feasible_on_rpi4(cls) -> bool:
        """
        Return whether full SLAM has been validated as feasible on RPi 4B.

        Returns False — feasibility evaluation is pending.

        Call without instantiation::

            from aerosentinel.localization.slam_adapter import SLAMAdapter
            if SLAMAdapter.is_feasible_on_rpi4():
                # enable SLAM source
                pass
        """
        return False  # pending evaluation — see HARDWARE_COMPATIBILITY.md

    @classmethod
    def candidate_engines(cls) -> Dict[str, Dict[str, Any]]:
        """Return info about candidate SLAM engines and their RPi 4 outlook."""
        return {
            "ORB-SLAM3": {
                "url": "https://github.com/UZ-SLAMLab/ORB_SLAM3",
                "type": "monocular / stereo / RGB-D",
                "rpi4_notes": (
                    "Requires C++14 build from source. No arm64 binary available. "
                    "High CPU load; likely < 5 FPS at 320x240 on RPi 4. "
                    "RAM usage ~300-500 MB for map. Needs evaluation."
                ),
                "python_binding": "None official; pybind11 wrappers exist but unmaintained.",
            },
            "RTAB-Map": {
                "url": "https://github.com/introlab/rtabmap",
                "type": "RGB-D / stereo / lidar",
                "rpi4_notes": (
                    "Available via ROS2 or standalone. Heavier than ORB-SLAM3. "
                    "RealSense D435i depth + RGB would be a natural input. "
                    "Needs profiling on RPi 4."
                ),
                "python_binding": "rtabmap_ros (requires ROS2 install).",
            },
            "OpenVSLAM": {
                "url": "https://github.com/OpenVSLAM-Community/openvslam",
                "type": "monocular / stereo / RGB-D",
                "rpi4_notes": (
                    "Community fork; lighter than ORB-SLAM3. "
                    "Python bindings available. Needs benchmarking."
                ),
                "python_binding": "stellarXR/openvslam Python bindings.",
            },
        }
