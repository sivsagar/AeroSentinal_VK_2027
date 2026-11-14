"""
AeroSentinel Perception Package.

AI-powered detection, depth processing, obstacle detection,
RGB-D fusion, and hazard assessment.
"""

from .inference_engine import (
    Detection,
    InferenceEngine,
    InferenceEngineFactory,
    MockInferenceEngine,
    ONNXInferenceEngine,
)
from .detector import PersonDetector
from .depth_processor import DepthProcessor, ProcessedDepth
from .obstacle_detector import (
    ObstacleDetector,
    ObstacleReport,
    ObstacleStatus,
    ProximityLevel,
    RegionAnalysis,
)
from .rgbd_fusion import FusedDetection, RGBDFusion
from .person_tracker import PersonTracker, TrackedPerson
from .hazard_detector import (
    HazardCategory,
    HazardDetector,
    HazardEvent,
    LocalizationStatus,
    VerificationStatus,
)

__all__ = [
    "Detection",
    "InferenceEngine",
    "InferenceEngineFactory",
    "MockInferenceEngine",
    "ONNXInferenceEngine",
    "PersonDetector",
    "DepthProcessor",
    "ProcessedDepth",
    "ObstacleDetector",
    "ObstacleReport",
    "ObstacleStatus",
    "ProximityLevel",
    "RegionAnalysis",
    "FusedDetection",
    "RGBDFusion",
    "PersonTracker",
    "TrackedPerson",
    "HazardCategory",
    "HazardDetector",
    "HazardEvent",
    "LocalizationStatus",
    "VerificationStatus",
]
