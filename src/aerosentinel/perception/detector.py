"""
detector.py — Main AI detection pipeline for AeroSentinel.

PersonDetector:
  - Accepts BGR frames
  - Runs inference via InferenceEngine in a background thread
  - Filters to target classes (person=0 by default)
  - Applies confidence threshold
  - Draws "Potential Survivor (XX%)" bounding boxes
  - Returns annotated frame + List[Detection]
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .inference_engine import Detection, InferenceEngine, MockInferenceEngine

logger = logging.getLogger(__name__)

# Colour palette (BGR) keyed by class_id
_COLOURS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 128),   # person — bright green
}
_DEFAULT_COLOUR = (0, 200, 255)  # orange-yellow for other classes


class PersonDetector:
    """
    Main detection pipeline.

    Parameters
    ----------
    engine : InferenceEngine
        Pre-loaded inference backend.
    target_class_ids : list of int
        Class IDs to report (default: [0] — person in COCO).
    conf_threshold : float
        Minimum confidence to draw / report a detection.
    inference_fps_target : float or None
        Rate-limit: skip frames that arrive faster than this target.
        None disables rate limiting.
    warmup_frames : int
        Number of initial frames to use for engine warmup.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        target_class_ids: Optional[List[int]] = None,
        conf_threshold: float = 0.4,
        inference_fps_target: Optional[float] = 5.0,
        warmup_frames: int = 3,
    ) -> None:
        self.engine = engine
        self.target_class_ids: List[int] = target_class_ids if target_class_ids is not None else [0]
        self.conf_threshold = conf_threshold
        self.inference_fps_target = inference_fps_target
        self.warmup_frames = warmup_frames

        # Thread-safe results queue
        self._results_queue: queue.Queue[Tuple[np.ndarray, List[Detection]]] = queue.Queue(maxsize=1)
        self._worker_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        # Rate limiting
        self._last_inference_time: float = 0.0

        # Frame rate tracking
        self._detect_count: int = 0
        self._fps_start: float = time.monotonic()
        self.output_fps: float = 0.0

        # Warmup counter
        self._warmup_done: bool = False
        self._warmup_count: int = 0

        # Latest inference result (from background thread)
        self._latest_detections: List[Detection] = []
        self._latest_lock = threading.Lock()

        self._start_worker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """
        Run detection on a BGR frame.

        Returns annotated frame and list of Detections for target classes.
        The inference runs in a background thread; this method returns the
        most recent result immediately (non-blocking, may be one frame stale).
        """
        # Warmup
        if not self._warmup_done:
            self._warmup_count += 1
            if self._warmup_count >= self.warmup_frames:
                self.engine.warmup()
                self._warmup_done = True

        # Rate limiting
        if self.inference_fps_target is not None and self.inference_fps_target > 0:
            min_interval = 1.0 / self.inference_fps_target
            now = time.monotonic()
            if (now - self._last_inference_time) < min_interval:
                # Return frame with last detections drawn (no new inference)
                with self._latest_lock:
                    detections = list(self._latest_detections)
                return self._annotate(frame.copy(), detections), detections
            self._last_inference_time = now

        # Submit frame to background worker (drop if busy)
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Worker busy — use latest result

        # Collect result if available
        try:
            _, new_detections = self._results_queue.get_nowait()
            with self._latest_lock:
                self._latest_detections = new_detections
        except queue.Empty:
            pass

        with self._latest_lock:
            detections = list(self._latest_detections)

        # FPS tracking
        self._detect_count += 1
        elapsed = time.monotonic() - self._fps_start
        if elapsed >= 1.0:
            self.output_fps = self._detect_count / elapsed
            self._detect_count = 0
            self._fps_start = time.monotonic()

        annotated = self._annotate(frame.copy(), detections)
        return annotated, detections

    def stop(self) -> None:
        """Stop the background inference thread."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(
            target=self._inference_worker, daemon=True, name="inference-worker"
        )
        self._worker_thread.start()

    def _inference_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                raw_detections = self.engine.infer(frame)
            except Exception as exc:
                logger.error("Inference error: %s", exc)
                raw_detections = []

            # Filter by target class and confidence
            filtered = [
                d for d in raw_detections
                if d.class_id in self.target_class_ids and d.confidence >= self.conf_threshold
            ]

            try:
                self._results_queue.put_nowait((frame, filtered))
            except queue.Full:
                # Discard old result, put new
                try:
                    self._results_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._results_queue.put_nowait((frame, filtered))
                except queue.Full:
                    pass

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame (in-place)."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            colour = _COLOURS.get(det.class_id, _DEFAULT_COLOUR)

            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label — always "Potential Survivor" not "Person"
            pct = int(det.confidence * 100)
            label = f"Potential Survivor ({pct}%)"

            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y1 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(
                frame,
                (x1, label_y1),
                (x1 + tw + 4, label_y1 + th + baseline + 4),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y1 + th + 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

            # Confidence bar (below bbox)
            bar_h = 6
            bar_y = min(y2 + 4, h - bar_h - 1)
            bar_w = x2 - x1
            filled_w = int(bar_w * det.confidence)
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (50, 50, 50), cv2.FILLED)
            cv2.rectangle(frame, (x1, bar_y), (x1 + filled_w, bar_y + bar_h), colour, cv2.FILLED)

        return frame
