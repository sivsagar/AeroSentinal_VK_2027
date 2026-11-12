"""
inference_engine.py — Abstracted inference backend for AeroSentinel.

Supports:
  - ONNXInferenceEngine: CPU-only onnxruntime (RPi 4 compatible)
  - MockInferenceEngine: synthetic detections for testing
  - InferenceEngineFactory: config-driven construction
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    timestamp: float
    inference_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class InferenceEngine(ABC):
    """Abstract inference backend."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a BGR frame and return detections."""

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the engine (optional dummy inference passes)."""


# ---------------------------------------------------------------------------
# ONNX engine
# ---------------------------------------------------------------------------

class ONNXInferenceEngine(InferenceEngine):
    """
    CPU-only ONNX Runtime inference engine.

    Parameters
    ----------
    model_path : str
        Path to the .onnx model file.
    input_resolution : Tuple[int, int]
        (width, height) expected by the model, default (320, 320).
    class_names : List[str]
        COCO or custom class label list.
    conf_threshold : float
        Minimum confidence to retain a detection.
    iou_threshold : float
        IoU threshold for NMS.
    """

    def __init__(
        self,
        model_path: str,
        input_resolution: Tuple[int, int] = (320, 320),
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_path = model_path
        self.input_resolution = input_resolution  # (W, H)
        self.class_names: List[str] = class_names or [str(i) for i in range(80)]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self._session: Any = None  # onnxruntime.InferenceSession
        self._input_name: str = ""
        self._output_names: List[str] = []

        self.inference_latency_ms: float = 0.0
        self._frame_count: int = 0
        self._fps_start: float = time.monotonic()
        self.inference_fps: float = 0.0

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        try:
            import onnxruntime as ort  # noqa: F401

            # CPU-only: explicitly exclude any GPU providers
            providers = ["CPUExecutionProvider"]
            logger.info("Loading ONNX model from '%s' (CPU-only)", self.model_path)
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            logger.info(
                "ONNX model loaded. Input: '%s', outputs: %s",
                self._input_name,
                self._output_names,
            )
        except ImportError:
            logger.warning(
                "onnxruntime not available — ONNXInferenceEngine cannot load. "
                "Use MockInferenceEngine for testing."
            )
            raise
        except Exception as exc:
            logger.error("Failed to load ONNX model: %s", exc)
            raise

    def warmup(self) -> None:
        if self._session is None:
            logger.warning("warmup() called before load(); skipping.")
            return
        w, h = self.input_resolution
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        logger.info("Warming up ONNX engine with dummy input %s ...", dummy.shape)
        for _ in range(3):
            self._session.run(self._output_names, {self._input_name: dummy})
        logger.info("Warmup complete.")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        if self._session is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        t0 = time.monotonic()
        blob = self._preprocess(frame)
        raw_outputs = self._session.run(self._output_names, {self._input_name: blob})
        detections = self._postprocess(raw_outputs, frame.shape)
        t1 = time.monotonic()

        self.inference_latency_ms = (t1 - t0) * 1000.0
        self._frame_count += 1
        elapsed = t1 - self._fps_start
        if elapsed >= 1.0:
            self.inference_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start = t1

        for d in detections:
            d.inference_latency_ms = self.inference_latency_ms
        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalise to [0,1] and convert to NCHW float32."""
        import cv2

        w, h = self.input_resolution
        resized = cv2.resize(frame, (w, h))
        # BGR -> RGB
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        # HWC -> CHW -> NCHW
        nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        return nchw

    def _postprocess(
        self, raw_outputs: List[np.ndarray], original_shape: Tuple
    ) -> List[Detection]:
        """
        Decode YOLOv8-style output and apply NMS.
        """
        import cv2

        orig_h, orig_w = original_shape[:2]
        in_w, in_h = self.input_resolution
        scale_x = orig_w / in_w
        scale_y = orig_h / in_h

        output = raw_outputs[0]

        detections: List[Detection] = []

        # Handle flat (N, 6) format: [x1, y1, x2, y2, conf, class_id]
        if output.ndim == 2 and output.shape[1] == 6:
            for row in output:
                x1, y1, x2, y2, conf, cls_id = row
                if conf < self.conf_threshold:
                    continue
                detections.append(
                    Detection(
                        class_id=int(cls_id),
                        class_name=self.class_names[int(cls_id)]
                        if int(cls_id) < len(self.class_names)
                        else str(int(cls_id)),
                        confidence=float(conf),
                        bbox=(
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(x2 * scale_x),
                            int(y2 * scale_y),
                        ),
                        timestamp=time.time(),
                    )
                )
            return self._apply_nms(detections)

        # Handle (1, 84, N) YOLOv8 format
        if output.ndim == 3:
            output = output[0]  # (84, N)
            output = output.T   # (N, 84)

        boxes, scores, class_ids = [], [], []
        for row in output:
            cx, cy, bw, bh = row[:4]
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])
            if conf < self.conf_threshold:
                continue
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold
        )
        if len(indices) == 0:
            return []

        for i in (indices.flatten() if hasattr(indices, "flatten") else indices):
            x, y, w_b, h_b = boxes[i]
            detections.append(
                Detection(
                    class_id=class_ids[i],
                    class_name=self.class_names[class_ids[i]]
                    if class_ids[i] < len(self.class_names)
                    else str(class_ids[i]),
                    confidence=scores[i],
                    bbox=(x, y, x + w_b, y + h_b),
                    timestamp=time.time(),
                )
            )
        return detections

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply NMS to detections already in x1,y1,x2,y2 format."""
        import cv2

        if not detections:
            return []
        boxes = [
            [d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]]
            for d in detections
        ]
        scores = [d.confidence for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        if len(indices) == 0:
            return []
        flat = indices.flatten() if hasattr(indices, "flatten") else indices
        return [detections[i] for i in flat]


# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------

class MockInferenceEngine(InferenceEngine):
    """
    Returns synthetic detections — no model file needed.
    Safe for unit tests and CI.
    """

    def __init__(
        self,
        synthetic_detections: Optional[List[Detection]] = None,
        input_resolution: Tuple[int, int] = (320, 320),
        latency_ms: float = 5.0,
    ) -> None:
        self.synthetic_detections = synthetic_detections
        self.input_resolution = input_resolution
        self.latency_ms = latency_ms
        self.inference_latency_ms: float = latency_ms
        self.inference_fps: float = 0.0
        self._loaded = False

    def load(self) -> None:
        self._loaded = True
        logger.info("MockInferenceEngine loaded (no model file required).")

    def warmup(self) -> None:
        logger.info("MockInferenceEngine warmup (no-op).")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        if not self._loaded:
            raise RuntimeError("Call load() before infer().")
        t = time.time()
        if self.synthetic_detections is not None:
            detections = [
                Detection(
                    class_id=d.class_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox=d.bbox,
                    timestamp=t,
                    inference_latency_ms=self.latency_ms,
                )
                for d in self.synthetic_detections
            ]
        else:
            # Default: one person at the frame centre
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            half = min(w, h) // 6
            detections = [
                Detection(
                    class_id=0,
                    class_name="person",
                    confidence=0.88,
                    bbox=(cx - half, cy - half, cx + half, cy + half),
                    timestamp=t,
                    inference_latency_ms=self.latency_ms,
                )
            ]
        return detections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class InferenceEngineFactory:
    """Create an InferenceEngine from a config dict."""

    @staticmethod
    def create(config: Dict[str, Any]) -> InferenceEngine:
        """
        Parameters (in config dict)
        ---------------------------
        engine_type : str       "onnx" | "mock"
        model_path  : str       Path to .onnx file (required for onnx)
        input_resolution : list [width, height], default [320, 320]
        class_names : list      Class label strings
        conf_threshold : float
        iou_threshold  : float
        """
        engine_type = config.get("engine_type", "mock").lower()
        input_res = tuple(config.get("input_resolution", [320, 320]))

        if engine_type == "onnx":
            model_path = config.get("model_path", "")
            try:
                engine = ONNXInferenceEngine(
                    model_path=model_path,
                    input_resolution=input_res,
                    class_names=config.get("class_names"),
                    conf_threshold=config.get("conf_threshold", 0.4),
                    iou_threshold=config.get("iou_threshold", 0.45),
                )
                engine.load()
                return engine
            except Exception as exc:
                logger.warning(
                    "ONNXInferenceEngine failed (%s); falling back to MockInferenceEngine.",
                    exc,
                )
                mock = MockInferenceEngine(input_resolution=input_res)
                mock.load()
                return mock

        # Default / explicit "mock"
        mock = MockInferenceEngine(input_resolution=input_res)
        mock.load()
        return mock
