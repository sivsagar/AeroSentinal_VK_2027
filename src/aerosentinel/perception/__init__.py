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
    RegionAnalysis,
)
from .rgbd_fusion import FusedDetection, RGBDFusion
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
    "RegionAnalysis",
    "FusedDetection",
    "RGBDFusion",
    "HazardCategory",
    "HazardDetector",
    "HazardEvent",
    "LocalizationStatus",
    "VerificationStatus",
]
