#!/usr/bin/env python3
"""
hardware_smoke_test.py — Standalone Hardware Integration Diagnostic Test.

Validates physical cameras (Raspberry Pi Camera Module 3 IMX708 and Intel
RealSense D435i) and ONNX YOLOv8n inference pipeline under real hardware mode
(mock=False) on Raspberry Pi 4 Model B (Debian 13 Trixie ARM64, Python 3.13).

Requirements verified:
  1. Loads actual camera configuration from config/camera.yaml.
  2. Initializes CameraManager in real hardware mode (mock=False).
  3. Loads models/yolov8n.onnx using ONNXInferenceEngine.
  4. Starts both physical cameras.
  5. Runs a 60-second acquisition and inference test.
  6. Collects RGB frame counts, depth frame counts, IMU availability,
     YOLO detection counts, inference latency and dropped-frame statistics.
  7. Records camera initialization errors, reconnect attempts, and timeout errors.
  8. Reports CPU temperature and RAM usage before and after the test.
  9. Safely releases all camera and inference resources (try/finally).
 10. Saves timestamped JSON and text reports under data/diagnostics/.

Safety constraints:
  - Read-only diagnostics.
  - Zero motor commands, arming commands, or MAVLink flight-control commands.
  - Never silently substitutes mock hardware.

Usage:
  python scripts/hardware_smoke_test.py [--duration 60] [--model models/yolov8n.onnx]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root and 'src' are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from aerosentinel.cameras.camera_manager import CameraManager, IMUData
from aerosentinel.perception.inference_engine import Detection, ONNXInferenceEngine
from aerosentinel.utils.config_loader import ConfigError, ConfigLoader
from aerosentinel.utils.performance import (
    read_cpu_temperature,
    read_cpu_usage_percent,
    read_ram_usage,
)

logger = logging.getLogger("hardware_smoke_test")


# ---------------------------------------------------------------------------
# Diagnostic Log Handler: intercept warnings and errors from drivers
# ---------------------------------------------------------------------------

class DiagnosticLogHandler(logging.Handler):
    """Intercepts and categorises camera/inference log records during testing."""

    def __init__(self) -> None:
        super().__init__()
        self.init_errors: List[str] = []
        self.reconnect_attempts: List[str] = []
        self.timeout_errors: List[str] = []
        self.other_errors: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        msg_lower = msg.lower()
        formatted = f"[{record.levelname}] {record.name}: {msg}"
        if any(k in msg_lower for k in ("init failed", "unavailable", "failed to initialize", "failed to start")):
            self.init_errors.append(formatted)
        elif any(k in msg_lower for k in ("reconnecting", "reconnect")):
            self.reconnect_attempts.append(formatted)
        elif any(k in msg_lower for k in ("timeout", "timed out", "wait_for_frames")):
            self.timeout_errors.append(formatted)
        elif record.levelno >= logging.ERROR:
            self.other_errors.append(formatted)

    def summary(self) -> Dict[str, Any]:
        return {
            "init_errors": list(self.init_errors),
            "reconnect_attempts": list(self.reconnect_attempts),
            "timeout_errors": list(self.timeout_errors),
            "other_errors": list(self.other_errors),
        }


# ---------------------------------------------------------------------------
# Test Runner Class
# ---------------------------------------------------------------------------

class HardwareSmokeTest:
    """Orchestrates the 60-second physical hardware integration test."""

    def __init__(
        self,
        config_dir: str = "config",
        model_path: str = "models/yolov8n.onnx",
        duration_s: float = 60.0,
        output_dir: str = "data/diagnostics",
        sync_inference: bool = False,
        poll_interval_s: float = 0.005,
    ) -> None:
        self.config_dir = Path(config_dir).resolve()
        self.model_path = Path(model_path).resolve()
        self.duration_s = float(duration_s)
        self.output_dir = Path(output_dir).resolve()
        self.sync_inference = sync_inference
        self.poll_interval_s = float(poll_interval_s)

        self._stop_event = threading.Event()
        self.log_handler = DiagnosticLogHandler()

        # Camera manager & inference engine handles
        self.camera_mgr: Optional[CameraManager] = None
        self.engine: Optional[ONNXInferenceEngine] = None

        # Configurations
        self.camera_cfg: Dict[str, Any] = {}
        self.rgb_cfg: Dict[str, Any] = {}
        self.depth_cfg: Dict[str, Any] = {}
        self.ai_cfg: Dict[str, Any] = {}

        # Raw metrics
        self.metrics: Dict[str, Any] = {
            "rgb_frames_received": 0,
            "depth_frames_received": 0,
            "imu_samples_received": 0,
            "imu_valid_samples": 0,
            "inference_passes": 0,
            "inference_skipped_frames": 0,
            "total_detections": 0,
            "class_counts": {},
            "inference_latencies_ms": [],
            "rgb_intervals_s": [],
            "depth_intervals_s": [],
            "depth_valid_pixel_pcts": [],
            "latest_imu": None,
            "start_time": 0.0,
            "end_time": 0.0,
            "actual_duration_s": 0.0,
        }

        # System states before & after
        self.system_metrics: Dict[str, Any] = {
            "ram_before": None,
            "ram_after": None,
            "cpu_temp_before_c": None,
            "cpu_temp_after_c": None,
            "cpu_percent_before": None,
            "cpu_percent_after": None,
        }

        # Status & results
        self.test_status: Dict[str, Any] = {
            "rgb_camera": {"status": "NOT_RUN", "reason": ""},
            "depth_camera": {"status": "NOT_RUN", "reason": ""},
            "imu": {"status": "NOT_RUN", "reason": ""},
            "onnx_inference": {"status": "NOT_RUN", "reason": ""},
            "overall": {"status": "NOT_RUN", "reason": ""},
        }

    # -----------------------------------------------------------------------
    # Setup & Teardown
    # -----------------------------------------------------------------------

    def setup_logging(self) -> None:
        """Attach DiagnosticLogHandler to capture driver messages."""
        root_logger = logging.getLogger()
        self.log_handler.setLevel(logging.WARNING)
        root_logger.addHandler(self.log_handler)

    def remove_logging(self) -> None:
        """Detach DiagnosticLogHandler."""
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.log_handler)

    def load_configurations(self) -> None:
        """Load camera.yaml and optional ai.yaml from config_dir."""
        logger.info("Loading configuration from %s...", self.config_dir)
        loader = ConfigLoader(config_dir=str(self.config_dir))

        camera_yaml = self.config_dir / "camera.yaml"
        if not camera_yaml.is_file():
            raise FileNotFoundError(f"Missing camera configuration file: {camera_yaml}")
        loader.load("camera.yaml")

        self.rgb_cfg = loader.get("rgb_camera", {})
        self.depth_cfg = loader.get("depth_camera", {})
        self.camera_cfg = {
            "rgb_camera": self.rgb_cfg,
            "depth_camera": self.depth_cfg,
        }

        # Load AI configuration if present
        ai_yaml = self.config_dir / "ai.yaml"
        if ai_yaml.is_file():
            try:
                loader.load("ai.yaml")
                self.ai_cfg = loader.get("detection", {})
            except ConfigError as exc:
                logger.warning("Optional ai.yaml load error: %s", exc)

    def init_onnx_engine(self) -> None:
        """Initialize and warm up ONNXInferenceEngine with models/yolov8n.onnx."""
        logger.info("Initializing ONNXInferenceEngine with model: %s", self.model_path)
        if not self.model_path.is_file():
            err_msg = f"ONNX model file not found at: '{self.model_path}'"
            self.test_status["onnx_inference"] = {"status": "FAIL", "reason": err_msg}
            raise FileNotFoundError(err_msg)

        input_res = tuple(self.ai_cfg.get("input_resolution", [320, 320]))
        conf_thresh = float(self.ai_cfg.get("confidence_threshold", 0.45))
        nms_thresh = float(self.ai_cfg.get("nms_threshold", 0.5))

        self.engine = ONNXInferenceEngine(
            model_path=str(self.model_path),
            input_resolution=input_res,
            conf_threshold=conf_thresh,
            iou_threshold=nms_thresh,
        )
        self.engine.load()
        self.engine.warmup()
        logger.info("ONNXInferenceEngine loaded and warmed up successfully.")

    def init_camera_manager(self) -> None:
        """Initialize CameraManager with real hardware mode (mock=False)."""
        logger.info("Initializing CameraManager in hardware mode (mock=False)...")
        # Explicitly enforce mock=False
        self.camera_mgr = CameraManager(config=self.camera_cfg, mock=False)

    # -----------------------------------------------------------------------
    # Main Execution
    # -----------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the complete diagnostic smoke test."""
        self.setup_logging()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting AeroSentinel Hardware Smoke Test (Duration: %.1fs)", self.duration_s)
        print("\n" + "=" * 80)
        print("  AEROSENTINEL HARDWARE INTEGRATION SMOKE TEST")
        print(f"  Duration: {self.duration_s:.1f} s | Target Model: {self.model_path.name}")
        print("=" * 80)

        # 1. Record pre-test system metrics
        self.system_metrics["cpu_temp_before_c"] = read_cpu_temperature()
        self.system_metrics["ram_before"] = read_ram_usage()
        self.system_metrics["cpu_percent_before"] = read_cpu_usage_percent(interval=0.1)

        temp_str = (
            f"{self.system_metrics['cpu_temp_before_c']:.1f}°C"
            if self.system_metrics["cpu_temp_before_c"] is not None
            else "N/A"
        )
        ram_str = (
            f"{self.system_metrics['ram_before'].get('used_mb', 0):.1f} MB / "
            f"{self.system_metrics['ram_before'].get('total_mb', 0):.1f} MB "
            f"({self.system_metrics['ram_before'].get('percent', 0):.1f}%)"
            if self.system_metrics["ram_before"]
            else "N/A"
        )
        print(f"Initial State : CPU Temp: {temp_str} | RAM: {ram_str} | CPU Load: {self.system_metrics['cpu_percent_before'] or 0.0:.1f}%")

        # 2. Initialize inference engine and cameras
        try:
            self.load_configurations()
            self.init_onnx_engine()
        except Exception as exc:
            logger.error("Inference Engine Initialization Error: %s", exc)
            self.test_status["onnx_inference"] = {"status": "FAIL", "reason": str(exc)}
            self.log_handler.init_errors.append(f"Inference Engine init error: {exc}")

        try:
            self.init_camera_manager()
        except Exception as exc:
            logger.error("CameraManager Initialization Error: %s", exc)
            self.log_handler.init_errors.append(f"CameraManager init error: {exc}")

        # 3. Start physical cameras
        if self.camera_mgr is not None:
            logger.info("Starting physical cameras...")
            self.camera_mgr.start_all()
            time.sleep(1.0)  # brief stabilization window

        # Check driver initial connection status
        cam_status = self.camera_mgr.status() if self.camera_mgr else {}
        rgb_info = cam_status.get("rgb", {})
        depth_info = cam_status.get("depth", {})
        print(f"Camera Status : RGB: {'CONNECTED' if rgb_info.get('connected') else 'DISCONNECTED/UNAVAILABLE'} | "
              f"Depth: {'CONNECTED' if depth_info.get('connected') else 'DISCONNECTED/UNAVAILABLE'}")

        # 4. Run acquisition and inference loop
        inference_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        worker_thread: Optional[threading.Thread] = None

        if not self.sync_inference and self.engine is not None:
            worker_thread = threading.Thread(
                target=self._inference_worker_loop,
                args=(inference_queue,),
                name="smoke-infer-worker",
                daemon=True,
            )
            worker_thread.start()

        try:
            self._acquisition_loop(inference_queue)
        except KeyboardInterrupt:
            print("\n[!] Keyboard interrupt received — terminating test gracefully...")
        finally:
            self._stop_event.set()
            if worker_thread is not None:
                worker_thread.join(timeout=2.0)

            # 5. Release all camera and inference resources safely
            logger.info("Releasing camera and sensor resources...")
            if self.camera_mgr is not None:
                try:
                    self.camera_mgr.stop_all()
                except Exception as exc:
                    logger.error("Error stopping CameraManager: %s", exc)

        # 6. Record post-test system metrics
        self.system_metrics["cpu_temp_after_c"] = read_cpu_temperature()
        self.system_metrics["ram_after"] = read_ram_usage()
        self.system_metrics["cpu_percent_after"] = read_cpu_usage_percent(interval=0.1)

        # 7. Evaluate pass/fail and generate report
        self.evaluate_results()
        report = self.generate_report()
        self.save_report(report)
        self.print_summary(report)

        self.remove_logging()
        return report

    # -----------------------------------------------------------------------
    # Acquisition and Inference Loop
    # -----------------------------------------------------------------------

    def _inference_worker_loop(self, infer_q: queue.Queue[np.ndarray]) -> None:
        """Background worker running ONNX inference concurrently."""
        while not self._stop_event.is_set():
            try:
                frame = infer_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.engine is None:
                continue

            try:
                t0 = time.perf_counter()
                dets = self.engine.infer(frame)
                lat_ms = (time.perf_counter() - t0) * 1000.0

                self.metrics["inference_latencies_ms"].append(lat_ms)
                self.metrics["inference_passes"] += 1
                self.metrics["total_detections"] += len(dets)
                for d in dets:
                    self.metrics["class_counts"][d.class_name] = (
                        self.metrics["class_counts"].get(d.class_name, 0) + 1
                    )
            except Exception as exc:
                logger.error("Inference execution error: %s", exc)
                self.log_handler.other_errors.append(f"Inference error: {exc}")

    def _acquisition_loop(self, infer_q: queue.Queue[np.ndarray]) -> None:
        """Poll camera manager for RGB, Depth and IMU frames."""
        start_ts = time.monotonic()
        self.metrics["start_time"] = start_ts
        last_progress_print = start_ts
        last_rgb_ts = 0.0
        last_depth_ts = 0.0
        last_rgb_frame_time = start_ts
        last_depth_frame_time = start_ts

        rgb_configured_fps = float(self.rgb_cfg.get("fps", 15))
        depth_configured_fps = float(self.depth_cfg.get("fps", 15))

        print("\n--- Running 60s Acquisition & Inference Test (Ctrl+C to abort) ---")

        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed = now - start_ts
            if elapsed >= self.duration_s:
                break

            # 1. Drain RGB frame
            if self.camera_mgr is not None:
                rgb_res = self.camera_mgr.get_rgb_frame()
                if rgb_res is not None:
                    rgb_frame, rgb_ts = rgb_res
                    self.metrics["rgb_frames_received"] += 1
                    last_rgb_frame_time = now
                    if last_rgb_ts > 0.0:
                        self.metrics["rgb_intervals_s"].append(rgb_ts - last_rgb_ts)
                    last_rgb_ts = rgb_ts

                    # Feed to inference
                    if self.engine is not None:
                        if self.sync_inference:
                            try:
                                t0 = time.perf_counter()
                                dets = self.engine.infer(rgb_frame)
                                lat_ms = (time.perf_counter() - t0) * 1000.0
                                self.metrics["inference_latencies_ms"].append(lat_ms)
                                self.metrics["inference_passes"] += 1
                                self.metrics["total_detections"] += len(dets)
                                for d in dets:
                                    self.metrics["class_counts"][d.class_name] = (
                                        self.metrics["class_counts"].get(d.class_name, 0) + 1
                                    )
                            except Exception as exc:
                                logger.error("Sync inference error: %s", exc)
                        else:
                            try:
                                infer_q.put_nowait(rgb_frame)
                            except queue.Full:
                                self.metrics["inference_skipped_frames"] += 1

            # 2. Drain Depth frame
            if self.camera_mgr is not None:
                depth_res = self.camera_mgr.get_depth_frame()
                if depth_res is not None:
                    depth_mm, color_bgr, depth_ts = depth_res
                    self.metrics["depth_frames_received"] += 1
                    last_depth_frame_time = now
                    if last_depth_ts > 0.0:
                        self.metrics["depth_intervals_s"].append(depth_ts - last_depth_ts)
                    last_depth_ts = depth_ts

                    # Sample depth quality (every 15 frames)
                    if self.metrics["depth_frames_received"] % 15 == 1:
                        total_px = depth_mm.size
                        valid_px = int(np.count_nonzero(depth_mm))
                        pct = (valid_px / float(total_px)) * 100.0 if total_px > 0 else 0.0
                        self.metrics["depth_valid_pixel_pcts"].append(pct)

            # 3. Drain IMU reading
            if self.camera_mgr is not None:
                imu_data = self.camera_mgr.get_imu_data()
                if imu_data is not None:
                    self.metrics["imu_samples_received"] += 1
                    if imu_data.is_valid:
                        self.metrics["imu_valid_samples"] += 1
                    self.metrics["latest_imu"] = {
                        "timestamp": imu_data.timestamp,
                        "accel": [imu_data.accel_x, imu_data.accel_y, imu_data.accel_z],
                        "gyro": [imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z],
                        "roll": imu_data.roll_estimate,
                        "pitch": imu_data.pitch_estimate,
                        "is_valid": imu_data.is_valid,
                    }

            # 4. Check for sensor timeout (if stream was active but stalled > 3.0s)
            if self.metrics["rgb_frames_received"] > 0 and (now - last_rgb_frame_time) > 3.0:
                self.log_handler.timeout_errors.append(
                    f"RGB stream stalled for {now - last_rgb_frame_time:.1f}s at t={elapsed:.1f}s"
                )
                last_rgb_frame_time = now  # prevent spam

            if self.metrics["depth_frames_received"] > 0 and (now - last_depth_frame_time) > 3.0:
                self.log_handler.timeout_errors.append(
                    f"Depth stream stalled for {now - last_depth_frame_time:.1f}s at t={elapsed:.1f}s"
                )
                last_depth_frame_time = now  # prevent spam

            # Periodic status print (every 5 seconds)
            if now - last_progress_print >= 5.0:
                last_progress_print = now
                rgb_fps = self.metrics["rgb_frames_received"] / max(elapsed, 0.1)
                depth_fps = self.metrics["depth_frames_received"] / max(elapsed, 0.1)
                infer_passes = self.metrics["inference_passes"]
                avg_lat = (
                    np.mean(self.metrics["inference_latencies_ms"][-10:])
                    if self.metrics["inference_latencies_ms"]
                    else 0.0
                )
                temp = read_cpu_temperature()
                temp_s = f"{temp:.1f}°C" if temp is not None else "N/A"
                print(
                    f"[{elapsed:5.1f}s / {self.duration_s:.0f}s] "
                    f"RGB: {self.metrics['rgb_frames_received']:4d} ({rgb_fps:4.1f} FPS) | "
                    f"Depth: {self.metrics['depth_frames_received']:4d} ({depth_fps:4.1f} FPS) | "
                    f"IMU: {self.metrics['imu_samples_received']:5d} | "
                    f"YOLO: {infer_passes:3d} ({avg_lat:5.1f}ms) | Temp: {temp_s}"
                )

            time.sleep(self.poll_interval_s)

        end_ts = time.monotonic()
        self.metrics["end_time"] = end_ts
        self.metrics["actual_duration_s"] = end_ts - start_ts

        # Fallback verification if cameras returned 0 frames (e.g. absent hardware)
        # Ensure ONNX model execution itself is thoroughly verified
        if self.metrics["inference_passes"] == 0 and self.engine is not None:
            try:
                dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                t0 = time.perf_counter()
                dets = self.engine.infer(dummy_frame)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                self.metrics["inference_latencies_ms"].append(lat_ms)
                self.metrics["inference_passes"] += 1
                logger.info("Executed standalone dummy inference validation (latency: %.2f ms)", lat_ms)
            except Exception as exc:
                self.log_handler.other_errors.append(f"Standalone inference validation failed: {exc}")

    # -----------------------------------------------------------------------
    # Evaluation & Reporting
    # -----------------------------------------------------------------------

    def evaluate_results(self) -> None:
        """Evaluate PASS/FAIL for each physical sensor and pipeline component."""
        elapsed = max(self.metrics["actual_duration_s"], 0.1)
        rgb_count = self.metrics["rgb_frames_received"]
        depth_count = self.metrics["depth_frames_received"]
        imu_count = self.metrics["imu_samples_received"]
        infer_count = self.metrics["inference_passes"]

        # RGB Camera evaluation
        if rgb_count > 0:
            rgb_fps = rgb_count / elapsed
            self.test_status["rgb_camera"] = {
                "status": "PASS",
                "reason": f"Acquired {rgb_count} frames at {rgb_fps:.2f} FPS",
            }
        else:
            self.test_status["rgb_camera"] = {
                "status": "FAIL",
                "reason": "Zero RGB frames acquired (camera not detected or driver failed to start)",
            }

        # Depth Camera evaluation
        if depth_count > 0:
            depth_fps = depth_count / elapsed
            avg_valid_pct = (
                float(np.mean(self.metrics["depth_valid_pixel_pcts"]))
                if self.metrics["depth_valid_pixel_pcts"]
                else 0.0
            )
            self.test_status["depth_camera"] = {
                "status": "PASS",
                "reason": f"Acquired {depth_count} depth frames at {depth_fps:.2f} FPS (avg valid: {avg_valid_pct:.1f}%)",
            }
        else:
            self.test_status["depth_camera"] = {
                "status": "FAIL",
                "reason": "Zero Depth frames acquired (RealSense not detected or driver failed to start)",
            }

        # IMU evaluation
        if imu_count > 0:
            rate_hz = imu_count / elapsed
            valid_pct = (self.metrics["imu_valid_samples"] / float(imu_count)) * 100.0
            self.test_status["imu"] = {
                "status": "PASS",
                "reason": f"Received {imu_count} IMU samples at {rate_hz:.1f} Hz ({valid_pct:.1f}% valid)",
            }
        else:
            self.test_status["imu"] = {
                "status": "FAIL",
                "reason": "Zero IMU samples received from RealSense D435i",
            }

        # ONNX Inference evaluation
        if infer_count > 0:
            avg_lat = float(np.mean(self.metrics["inference_latencies_ms"]))
            self.test_status["onnx_inference"] = {
                "status": "PASS",
                "reason": f"Completed {infer_count} inference passes (avg latency: {avg_lat:.1f} ms)",
            }
        elif self.test_status["onnx_inference"]["status"] != "FAIL":
            self.test_status["onnx_inference"] = {
                "status": "FAIL",
                "reason": "Zero inference passes executed",
            }

        # Overall Status
        all_passed = (
            self.test_status["rgb_camera"]["status"] == "PASS"
            and self.test_status["depth_camera"]["status"] == "PASS"
            and self.test_status["imu"]["status"] == "PASS"
            and self.test_status["onnx_inference"]["status"] == "PASS"
        )
        self.test_status["overall"] = {
            "status": "PASS" if all_passed else "FAIL",
            "reason": "All hardware sensors and inference pipeline passed validation"
            if all_passed
            else "One or more hardware components failed validation",
        }

    def generate_report(self) -> Dict[str, Any]:
        """Compile complete structured report dictionary."""
        elapsed = max(self.metrics["actual_duration_s"], 0.1)
        rgb_fps_target = float(self.rgb_cfg.get("fps", 15))
        depth_fps_target = float(self.depth_cfg.get("fps", 15))

        rgb_expected = int(round(rgb_fps_target * elapsed))
        rgb_received = self.metrics["rgb_frames_received"]
        rgb_dropped = max(0, rgb_expected - rgb_received)
        rgb_drop_rate = (rgb_dropped / rgb_expected * 100.0) if rgb_expected > 0 else 0.0

        depth_expected = int(round(depth_fps_target * elapsed))
        depth_received = self.metrics["depth_frames_received"]
        depth_dropped = max(0, depth_expected - depth_received)
        depth_drop_rate = (depth_dropped / depth_expected * 100.0) if depth_expected > 0 else 0.0

        lats = self.metrics["inference_latencies_ms"]
        latency_stats = {
            "count": len(lats),
            "min_ms": round(float(np.min(lats)), 2) if lats else 0.0,
            "max_ms": round(float(np.max(lats)), 2) if lats else 0.0,
            "mean_ms": round(float(np.mean(lats)), 2) if lats else 0.0,
            "median_ms": round(float(np.median(lats)), 2) if lats else 0.0,
            "p95_ms": round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
        }

        temp_before = self.system_metrics["cpu_temp_before_c"]
        temp_after = self.system_metrics["cpu_temp_after_c"]
        temp_delta = (
            round(temp_after - temp_before, 2)
            if (temp_before is not None and temp_after is not None)
            else None
        )

        ram_before = self.system_metrics["ram_before"] or {}
        ram_after = self.system_metrics["ram_after"] or {}
        ram_delta_mb = (
            round(ram_after.get("used_mb", 0) - ram_before.get("used_mb", 0), 2)
            if (ram_before and ram_after)
            else None
        )

        log_summary = self.log_handler.summary()

        report: Dict[str, Any] = {
            "metadata": {
                "test_name": "hardware_smoke_test",
                "timestamp": datetime.datetime.now().isoformat(),
                "duration_seconds": round(elapsed, 2),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "python_version": platform.python_version(),
                },
                "config_dir": str(self.config_dir),
                "model_path": str(self.model_path),
            },
            "status_summary": self.test_status,
            "rgb_camera": {
                "configured_fps": rgb_fps_target,
                "configured_resolution": self.rgb_cfg.get("resolution", [1280, 720]),
                "frames_received": rgb_received,
                "actual_fps": round(rgb_received / elapsed, 2),
                "expected_frames": rgb_expected,
                "dropped_frames": rgb_dropped,
                "drop_rate_percent": round(rgb_drop_rate, 2),
            },
            "depth_camera": {
                "configured_fps": depth_fps_target,
                "depth_resolution": self.depth_cfg.get("depth_resolution", [640, 480]),
                "color_resolution": self.depth_cfg.get("color_resolution", [640, 480]),
                "frames_received": depth_received,
                "actual_fps": round(depth_received / elapsed, 2),
                "expected_frames": depth_expected,
                "dropped_frames": depth_dropped,
                "drop_rate_percent": round(depth_drop_rate, 2),
                "avg_valid_depth_pixel_percent": round(
                    float(np.mean(self.metrics["depth_valid_pixel_pcts"])), 2
                )
                if self.metrics["depth_valid_pixel_pcts"]
                else 0.0,
            },
            "imu": {
                "samples_received": self.metrics["imu_samples_received"],
                "sample_rate_hz": round(self.metrics["imu_samples_received"] / elapsed, 2),
                "configured_accel_fps": self.depth_cfg.get("accel_fps", 250),
                "configured_gyro_fps": self.depth_cfg.get("gyro_fps", 400),
                "valid_samples": self.metrics["imu_valid_samples"],
                "valid_percent": round(
                    (self.metrics["imu_valid_samples"] / float(max(self.metrics["imu_samples_received"], 1)))
                    * 100.0,
                    2,
                ),
                "latest_sample": self.metrics["latest_imu"],
            },
            "inference": {
                "model_path": str(self.model_path),
                "total_passes": len(lats),
                "inference_fps": round(len(lats) / elapsed, 2),
                "skipped_frames": self.metrics["inference_skipped_frames"],
                "latency_stats_ms": latency_stats,
                "total_detections": self.metrics["total_detections"],
                "class_counts": self.metrics["class_counts"],
            },
            "system_performance": {
                "cpu_temperature": {
                    "start_c": temp_before,
                    "end_c": temp_after,
                    "delta_c": temp_delta,
                },
                "ram_usage": {
                    "start": ram_before,
                    "end": ram_after,
                    "delta_mb": ram_delta_mb,
                },
                "cpu_utilization_percent": {
                    "start": self.system_metrics["cpu_percent_before"],
                    "end": self.system_metrics["cpu_percent_after"],
                },
            },
            "diagnostics": {
                "camera_init_errors": log_summary["init_errors"],
                "reconnect_attempts": log_summary["reconnect_attempts"],
                "timeout_errors": log_summary["timeout_errors"],
                "other_errors": log_summary["other_errors"],
            },
        }
        return report

    def save_report(self, report: Dict[str, Any]) -> Tuple[Path, Path]:
        """Save report to JSON and TXT format under output_dir."""
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"hardware_smoke_test_{ts_str}.json"
        txt_path = self.output_dir / f"hardware_smoke_test_{ts_str}.txt"

        # 1. Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # 2. Write Text Report
        txt_content = self._format_text_report(report)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        logger.info("Saved diagnostics JSON: %s", json_path)
        logger.info("Saved diagnostics TXT : %s", txt_path)
        return json_path, txt_path

    def _format_text_report(self, r: Dict[str, Any]) -> str:
        """Format report as a clean human-readable text document."""
        meta = r["metadata"]
        status = r["status_summary"]
        rgb = r["rgb_camera"]
        depth = r["depth_camera"]
        imu = r["imu"]
        inf = r["inference"]
        lat = inf["latency_stats_ms"]
        sys_p = r["system_performance"]
        diag = r["diagnostics"]

        tb = sys_p["cpu_temperature"]["start_c"]
        te = sys_p["cpu_temperature"]["end_c"]
        td = sys_p["cpu_temperature"]["delta_c"]
        tb_str = f"{tb:.1f}°C" if tb is not None else "N/A"
        te_str = f"{te:.1f}°C" if te is not None else "N/A"
        td_str = f"{td:+.1f}°C" if td is not None else "N/A"

        rb = sys_p["ram_usage"]["start"]
        re = sys_p["ram_usage"]["end"]
        rd = sys_p["ram_usage"]["delta_mb"]
        rb_str = f"{rb.get('used_mb', 0):.1f} MB ({rb.get('percent', 0):.1f}%)" if rb else "N/A"
        re_str = f"{re.get('used_mb', 0):.1f} MB ({re.get('percent', 0):.1f}%)" if re else "N/A"
        rd_str = f"{rd:+.1f} MB" if rd is not None else "N/A"

        lines = [
            "=" * 80,
            "AeroSentinel Hardware Integration Smoke Test Report",
            "=" * 80,
            f"Timestamp             : {meta['timestamp']}",
            f"Duration              : {meta['duration_seconds']} s",
            f"Platform              : {meta['platform']['system']} {meta['platform']['release']} ({meta['platform']['machine']})",
            f"Python Version        : {meta['platform']['python_version']}",
            f"Configuration         : {meta['config_dir']}",
            f"ONNX Model            : {meta['model_path']}",
            "",
            "--- COMPONENT VALIDATION SUMMARY ---",
            f"[{status['rgb_camera']['status']:4s}] RGB Camera (IMX708)   : {status['rgb_camera']['reason']}",
            f"[{status['depth_camera']['status']:4s}] Depth Camera (D435i)  : {status['depth_camera']['reason']}",
            f"[{status['imu']['status']:4s}] RealSense IMU (BMI055): {status['imu']['reason']}",
            f"[{status['onnx_inference']['status']:4s}] ONNX Inference        : {status['onnx_inference']['reason']}",
            f"[{status['overall']['status']:4s}] Overall Result        : {status['overall']['reason']}",
            "",
            "--- DETAILED SENSOR METRICS ---",
            f"RGB Camera (PiCamera Module 3):",
            f"  - Frames Received   : {rgb['frames_received']} / {rgb['expected_frames']} expected",
            f"  - Actual / Target   : {rgb['actual_fps']} FPS / {rgb['configured_fps']} FPS",
            f"  - Dropped Frames    : {rgb['dropped_frames']} ({rgb['drop_rate_percent']}%)",
            f"  - Resolution        : {rgb['configured_resolution']}",
            "",
            f"Depth Camera (Intel RealSense D435i):",
            f"  - Frames Received   : {depth['frames_received']} / {depth['expected_frames']} expected",
            f"  - Actual / Target   : {depth['actual_fps']} FPS / {depth['configured_fps']} FPS",
            f"  - Dropped Frames    : {depth['dropped_frames']} ({depth['drop_rate_percent']}%)",
            f"  - Depth Resolution  : {depth['depth_resolution']}",
            f"  - Avg Valid Depth % : {depth['avg_valid_depth_pixel_percent']}%",
            "",
            f"RealSense IMU (Bosch BMI055):",
            f"  - Samples Received  : {imu['samples_received']}",
            f"  - Sample Rate       : {imu['sample_rate_hz']} Hz (Target: Accel {imu['configured_accel_fps']} Hz / Gyro {imu['configured_gyro_fps']} Hz)",
            f"  - Valid Samples     : {imu['valid_samples']} ({imu['valid_percent']}%)",
            "",
            "--- ONNX INFERENCE PERFORMANCE ---",
            f"  - Total Passes      : {inf['total_passes']}",
            f"  - Inference Rate    : {inf['inference_fps']} FPS",
            f"  - Skipped Frames    : {inf['skipped_frames']}",
            f"  - Latency (ms)      : Min={lat['min_ms']}, Max={lat['max_ms']}, Mean={lat['mean_ms']}, Median={lat['median_ms']}, P95={lat['p95_ms']}",
            f"  - Detections Count  : {inf['total_detections']} detections ({inf['class_counts']})",
            "",
            "--- SYSTEM THERMALS & MEMORY ---",
            f"  - CPU Temperature   : {tb_str} -> {te_str} (Delta: {td_str})",
            f"  - RAM Utilization   : {rb_str} -> {re_str} (Delta: {rd_str})",
            f"  - CPU Load Start/End: {sys_p['cpu_utilization_percent']['start'] or 0.0:.1f}% -> {sys_p['cpu_utilization_percent']['end'] or 0.0:.1f}%",
            "",
            "--- DIAGNOSTIC ANOMALIES & LOG ERRORS ---",
            f"  - Camera Init Errors: {len(diag['camera_init_errors'])}",
        ]
        for err in diag["camera_init_errors"]:
            lines.append(f"      * {err}")

        lines.append(f"  - Reconnect Attempts: {len(diag['reconnect_attempts'])}")
        for rec in diag["reconnect_attempts"]:
            lines.append(f"      * {rec}")

        lines.append(f"  - Timeout Errors    : {len(diag['timeout_errors'])}")
        for to in diag["timeout_errors"]:
            lines.append(f"      * {to}")

        if diag["other_errors"]:
            lines.append(f"  - Other Errors      : {len(diag['other_errors'])}")
            for oe in diag["other_errors"]:
                lines.append(f"      * {oe}")

        lines.extend([
            "=" * 80,
            f"OVERALL STATUS: {status['overall']['status']}",
            "=" * 80,
        ])
        return "\n".join(lines)

    def print_summary(self, r: Dict[str, Any]) -> None:
        """Print formatted summary table to console."""
        txt = self._format_text_report(r)
        print("\n" + txt + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AeroSentinel Physical Hardware Integration Smoke Test (Pi 4B / Debian 13 ARM64)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to directory containing camera.yaml and ai.yaml (default: 'config')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.onnx",
        help="Path to YOLOv8n ONNX model file (default: 'models/yolov8n.onnx')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/diagnostics",
        help="Directory to save diagnostic reports (default: 'data/diagnostics')",
    )
    parser.add_argument(
        "--sync-inference",
        action="store_true",
        help="Run inference synchronously in main acquisition loop instead of worker thread",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.005,
        help="Camera poll loop interval in seconds (default: 0.005)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging output",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    test = HardwareSmokeTest(
        config_dir=args.config_dir,
        model_path=args.model,
        duration_s=args.duration,
        output_dir=args.output_dir,
        sync_inference=args.sync_inference,
        poll_interval_s=args.poll_interval,
    )

    try:
        report = test.run()
        overall_status = report.get("status_summary", {}).get("overall", {}).get("status", "FAIL")
        return 0 if overall_status == "PASS" else 1
    except Exception as exc:
        logger.exception("Fatal smoke test execution error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
