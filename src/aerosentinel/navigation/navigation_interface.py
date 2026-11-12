"""
AeroSentinel Navigation Command Interface
=========================================
SAFETY ARCHITECTURE MANDATE:
Perception-only and read-only telemetry mode initially.
NO automatic arming, takeoff, landing, motor control, or obstacle-avoidance
flight maneuvers may be commanded directly from the companion computer.
Flight controller (e.g. ArduPilot / PX4) retains exclusive authority over
flight stabilization and failsafes.

All future navigation commands are defined here as abstract interfaces with
mandatory safety verification requirements.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class NavigationCommandError(Exception):
    """Raised when an invalid or unverified navigation command is attempted."""


class BaseNavigationInterface(ABC):
    """
    Abstract interface for future autonomous UAV navigation commands.

    Every concrete implementation must require explicit safety validation,
    arm-state verification, and emergency-stop checks before issuing any
    write command to the flight controller.
    """

    @abstractmethod
    def is_navigation_enabled(self) -> bool:
        """Return True if autonomous navigation write-mode is safely authorized."""
        pass

    @abstractmethod
    def goto_position(
        self, lat: float, lon: float, alt_m: float, yaw_deg: Optional[float] = None
    ) -> None:
        """
        Command vehicle to navigate to target WGS84 coordinate.

        Parameters
        ----------
        lat : float
            Target latitude in degrees.
        lon : float
            Target longitude in degrees.
        alt_m : float
            Target altitude above ground in meters.
        yaw_deg : Optional[float]
            Target vehicle heading in degrees.
        """
        pass

    @abstractmethod
    def set_velocity(self, vx_ms: float, vy_ms: float, vz_ms: float) -> None:
        """
        Command vehicle velocity in body or NED frame.

        Parameters
        ----------
        vx_ms, vy_ms, vz_ms : float
            Velocities in meters per second.
        """
        pass

    @abstractmethod
    def return_to_home(self) -> None:
        """Command vehicle to trigger Return-To-Launch (RTL)."""
        pass


class SafeNavigationInterface(BaseNavigationInterface):
    """
    Initial production implementation of navigation interface: Read-Only Safety Enforcer.

    Blocks all active write commands, ensuring companion computer never inadvertently
    overrides flight controller stabilization or failsafes.
    """

    def __init__(self, allow_autonomous_writes: bool = False) -> None:
        self._allow_writes = allow_autonomous_writes
        logger.info(
            "SafeNavigationInterface initialized. Autonomous writes enabled: %s",
            self._allow_writes,
        )

    def is_navigation_enabled(self) -> bool:
        return self._allow_writes

    def get_recommended_action(self, obstacle_report: Any) -> Dict[str, Any]:
        """
        Provide advisory decision support based on obstacle detection reports.
        DOES NOT execute flight maneuvers. Advisory data for ground station or pilot.
        """
        if obstacle_report is None:
            return {"action": "HOLD", "reason": "No obstacle data available"}

        status = getattr(obstacle_report, "overall_status", None)
        status_name = getattr(status, "name", str(status))

        if status_name == "CRITICAL":
            closest = getattr(obstacle_report, "closest_obstacle_m", 0.0)
            return {
                "action": "HOVER_WARN",
                "advisory": "Critical near-field obstacle detected. Manual pilot intervention advised.",
                "closest_distance_m": closest,
            }
        elif status_name == "WARNING":
            return {
                "action": "SLOW_FORWARD",
                "advisory": "Obstacle in warning zone. Reduce speed.",
            }
        return {
            "action": "PROCEED",
            "advisory": "Forward corridor clear within safety margins.",
        }

    def goto_position(
        self, lat: float, lon: float, alt_m: float, yaw_deg: Optional[float] = None
    ) -> None:
        if not self._allow_writes:
            raise NotImplementedError(
                "SAFETY LOCK: Autonomous goto_position is disabled in perception-only mode. "
                "Must be validated in SITL simulation and authorized by flight operator."
            )

    def set_velocity(self, vx_ms: float, vy_ms: float, vz_ms: float) -> None:
        if not self._allow_writes:
            raise NotImplementedError(
                "SAFETY LOCK: Autonomous velocity commands disabled in perception-only mode."
            )

    def return_to_home(self) -> None:
        if not self._allow_writes:
            raise NotImplementedError(
                "SAFETY LOCK: Direct RTH from companion computer disabled. "
                "Flight controller RC failsafe retains authority."
            )

