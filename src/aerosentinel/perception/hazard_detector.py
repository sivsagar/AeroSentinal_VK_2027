"""
hazard_detector.py — Hazard assessment for AeroSentinel.

Combines outputs from PersonDetector and ObstacleDetector to produce
a list of HazardEvents, all initially requiring human verification.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from .inference_engine import Detection
from .obstacle_detector import ObstacleReport, ObstacleStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HazardCategory(Enum):
    PERSON_DETECTED = "PERSON_DETECTED"
    OBSTACLE_CLOSE = "OBSTACLE_CLOSE"
    OBSTACLE_CRITICAL = "OBSTACLE_CRITICAL"
    VEHICLE = "VEHICLE"
    DEBRIS = "DEBRIS"


class VerificationStatus(Enum):
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"


class LocalizationStatus(Enum):
    UNAVAILABLE = "UNAVAILABLE"
    GPS_PENDING = "GPS_PENDING"
    LOCALIZED = "LOCALIZED"


# ---------------------------------------------------------------------------
# HazardEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class HazardEvent:
    id: str                            # uuid4 string
    timestamp: float
    category: HazardCategory
    confidence: float
    distance_m: Optional[float]
    image_path: Optional[str]
    localization_status: LocalizationStatus
    verification_status: VerificationStatus

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "confidence": self.confidence,
            "distance_m": self.distance_m,
            "image_path": self.image_path,
            "localization_status": self.localization_status.value,
            "verification_status": self.verification_status.value,
        }


# ---------------------------------------------------------------------------
# HazardDetector
# ---------------------------------------------------------------------------

class HazardDetector:
    """
    Assess hazards from person detections and obstacle reports.

    Parameters
    ----------
    person_conf_threshold : float
        Minimum confidence to raise a PERSON_DETECTED event.
    vehicle_class_ids : list of int
        COCO class IDs treated as vehicles (default: car=2, truck=7, bus=5).
    debris_class_ids : list of int
        Class IDs treated as debris/obstructions.
    """

    def __init__(
        self,
        person_conf_threshold: float = 0.4,
        vehicle_class_ids: Optional[List[int]] = None,
        debris_class_ids: Optional[List[int]] = None,
    ) -> None:
        self.person_conf_threshold = person_conf_threshold
        self.vehicle_class_ids: List[int] = vehicle_class_ids or [2, 5, 7]
        self.debris_class_ids: List[int] = debris_class_ids or []

        # Callback registry: category -> list of callables
        self._callbacks: Dict[HazardCategory, List[Callable[[HazardEvent], None]]] = {
            cat: [] for cat in HazardCategory
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(
        self, category: HazardCategory, fn: Callable[[HazardEvent], None]
    ) -> None:
        """Register a callback fired for every new HazardEvent of given category."""
        self._callbacks[category].append(fn)

    def assess(
        self,
        detections: List[Detection],
        obstacle_report: Optional[ObstacleReport] = None,
    ) -> List[HazardEvent]:
        """
        Assess hazards from detection + obstacle data.

        Parameters
        ----------
        detections : list of Detection
            Output from PersonDetector (or any InferenceEngine).
        obstacle_report : ObstacleReport or None
            Output from ObstacleDetector.

        Returns
        -------
        list of HazardEvent
            All events have verification_status = PENDING_HUMAN_REVIEW.
        """
        events: List[HazardEvent] = []
        now = time.time()

        # --- Person detections ---
        for det in detections:
            if det.class_id == 0 and det.confidence >= self.person_conf_threshold:
                evt = HazardEvent(
                    id=str(uuid.uuid4()),
                    timestamp=now,
                    category=HazardCategory.PERSON_DETECTED,
                    confidence=det.confidence,
                    distance_m=None,
                    image_path=None,
                    localization_status=LocalizationStatus.UNAVAILABLE,
                    verification_status=VerificationStatus.PENDING_HUMAN_REVIEW,
                )
                events.append(evt)
                self._fire_callbacks(evt)

            elif det.class_id in self.vehicle_class_ids:
                evt = HazardEvent(
                    id=str(uuid.uuid4()),
                    timestamp=now,
                    category=HazardCategory.VEHICLE,
                    confidence=det.confidence,
                    distance_m=None,
                    image_path=None,
                    localization_status=LocalizationStatus.UNAVAILABLE,
                    verification_status=VerificationStatus.PENDING_HUMAN_REVIEW,
                )
                events.append(evt)
                self._fire_callbacks(evt)

            elif det.class_id in self.debris_class_ids:
                evt = HazardEvent(
                    id=str(uuid.uuid4()),
                    timestamp=now,
                    category=HazardCategory.DEBRIS,
                    confidence=det.confidence,
                    distance_m=None,
                    image_path=None,
                    localization_status=LocalizationStatus.UNAVAILABLE,
                    verification_status=VerificationStatus.PENDING_HUMAN_REVIEW,
                )
                events.append(evt)
                self._fire_callbacks(evt)

        # --- Obstacle events ---
        if obstacle_report is not None:
            if obstacle_report.overall_status == ObstacleStatus.CRITICAL:
                evt = HazardEvent(
                    id=str(uuid.uuid4()),
                    timestamp=now,
                    category=HazardCategory.OBSTACLE_CRITICAL,
                    confidence=1.0,
                    distance_m=obstacle_report.closest_obstacle_m,
                    image_path=None,
                    localization_status=LocalizationStatus.UNAVAILABLE,
                    verification_status=VerificationStatus.PENDING_HUMAN_REVIEW,
                )
                events.append(evt)
                self._fire_callbacks(evt)

            elif obstacle_report.overall_status == ObstacleStatus.WARNING:
                evt = HazardEvent(
                    id=str(uuid.uuid4()),
                    timestamp=now,
                    category=HazardCategory.OBSTACLE_CLOSE,
                    confidence=1.0,
                    distance_m=obstacle_report.closest_obstacle_m,
                    image_path=None,
                    localization_status=LocalizationStatus.UNAVAILABLE,
                    verification_status=VerificationStatus.PENDING_HUMAN_REVIEW,
                )
                events.append(evt)
                self._fire_callbacks(evt)

        # Log all events
        for evt in events:
            logger.info(
                "HazardEvent id=%s category=%s confidence=%.2f dist=%s",
                evt.id,
                evt.category.value,
                evt.confidence,
                f"{evt.distance_m:.2f}m" if evt.distance_m is not None else "N/A",
            )

        return events

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fire_callbacks(self, event: HazardEvent) -> None:
        for fn in self._callbacks.get(event.category, []):
            try:
                fn(event)
            except Exception as exc:
                logger.error(
                    "Callback error for category %s: %s", event.category, exc
                )
