"""
AeroSentinel — Main Entry Point
=================================
Parses CLI arguments, loads configuration, initialises all subsystems,
and runs the main event loop.

Subsystems initialised:
  - CameraManager   (Module A)
  - PersonDetector  (Module B)
  - DepthProcessor + ObstacleDetector (Module C)
  - RGBDFusion      (Module D)
  - IMUProcessor    (Module E)
  - LocalizationManager (Module F)
  - HazardDetector  (Module G)
  - MissionManager  (Module H)
  - MAVLinkClient   (Module I — optional, read-only)
  - DashboardApp    (Module J — optional, runs in background thread)
  - DatabaseManager (Module K)
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure src/ is on the import path when run as a script
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from aerosentinel.utils.config_loader import ConfigLoader, ConfigError
from aerosentinel.utils.logger import get_logger, setup_logging
from aerosentinel.utils.performance import PerformanceMonitor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERSION = "0.1.0"
_TEAM_ID = "VK26-AU-PAI-0399"
_BANNER = r"""
  ___              ____            _   _            _
 / _ \  ___  _ __|  _ \ ___ _ __ | |_(_)_ __   ___| |
| | | |/ _ \| '__| |_) / _ \ '_ \| __| | '_ \ / _ \ |
| |_| |  __/| |  |  _ <  __/ |_) | |_| | | | |  __/ |
 \___/ \___||_|  |_| \_\___| .__/ \__|_|_| |_|\___|_|
                             |_|
  AeroSentinel UAV Disaster Search & Rescue System
  Version {version}  |  Team {team}
"""

_CONFIG_FILES = [
    "system.yaml",
    "camera.yaml",
    "depth.yaml",
    "ai.yaml",
    "fusion.yaml",
    "navigation.yaml",
    "mavlink.yaml",
    "mission.yaml",
]

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_running = True
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aerosentinel",
        description="AeroSentinel UAV Disaster Search & Rescue System",
    )
    parser.add_argument(
        "--config",
        metavar="DIR",
        default="config/",
        help="Path to the config directory (default: config/).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/simulation mode (no real hardware required).",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the log level set in system.yaml.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the web dashboard.",
    )
    parser.add_argument(
        "--dashboard-port",
        metavar="PORT",
        type=int,
        default=8080,
        help="Port for the web dashboard (default: 8080).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def _print_banner(version: str = _VERSION, team: str = _TEAM_ID) -> None:
    print(_BANNER.format(version=version, team=team))
    print(f"  Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python   : {platform.python_version()}")
    print(f"  PID      : {os.getpid()}")
    print()


def _load_config(config_dir: str) -> ConfigLoader:
    loader = ConfigLoader(config_dir=config_dir)
    for filename in _CONFIG_FILES:
        path = Path(config_dir) / filename
        if path.exists():
            try:
                loader.load(filename)
                logger.debug("Loaded %s", filename)
            except ConfigError as exc:
                logger.warning("Could not load %s: %s", filename, exc)
        else:
            logger.warning("Config file not found, skipping: %s", path)
    return loader


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


def _register_signal_handlers() -> None:
    global _running

    def _handle_signal(signum: int, frame: Any) -> None:
        global _running
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown…", sig_name)
        _running = False

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (OSError, AttributeError):
        pass  # SIGTERM not available on Windows


# ---------------------------------------------------------------------------
# Module A — Camera Management
# ---------------------------------------------------------------------------


def _init_cameras(cfg: ConfigLoader, mock: bool) -> Optional[Any]:
    """Initialise CameraManager with RGB + Depth drivers."""
    try:
        from aerosentinel.cameras.camera_manager import CameraManager

        camera_cfg = {
            "rgb_camera": cfg.get("rgb_camera", {}),
            "depth_camera": cfg.get("depth_camera", {}),
        }

        mgr = CameraManager(config=camera_cfg, mock=mock)
        mgr.start_all()
        logger.info("CameraManager started (mock=%s).", mock)
        return mgr
    except Exception as exc:
        logger.exception("Camera init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Module B — AI Survivor Detection
# ---------------------------------------------------------------------------


def _init_perception(cfg: ConfigLoader, mock: bool) -> Any:
    """Initialise PersonDetector with ONNX or Mock inference engine."""
    from aerosentinel.perception.detector import PersonDetector

    # Read configuration keys from detection.* (with fallback to ai.*)
    model_path = cfg.get("detection.model_path", cfg.get("ai.model_path", None))
    conf_threshold = float(
        cfg.get("detection.confidence_threshold", cfg.get("ai.confidence_threshold", 0.45))
    )
    input_resolution = tuple(
        cfg.get("detection.input_resolution", cfg.get("ai.input_resolution", [320, 320]))
    )
    fps_target = float(
        cfg.get("detection.inference_fps_target", cfg.get("ai.inference_fps_target", 3.0))
    )
    target_classes = cfg.get("detection.target_classes", [0])
    warmup_frames = int(cfg.get("detection.performance.warmup_frames", 3))

    if mock:
        from aerosentinel.perception.inference_engine import MockInferenceEngine

        engine = MockInferenceEngine(input_resolution=input_resolution)
        engine.load()
        logger.info("AI perception: using MockInferenceEngine (mock mode active).")
    else:
        # Hardware mode: MUST use the real ONNX model. Never silently fall back to mock.
        if not model_path:
            raise ValueError(
                "Hardware mode requires a valid model path in detection.model_path (none configured)."
            )

        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"ONNX model file not found at '{model_path}'. "
                "Hardware mode requires a valid ONNX model file. "
                "Download or export the model first (see models/README.md), or run with --mock for simulation."
            )

        from aerosentinel.perception.inference_engine import ONNXInferenceEngine

        logger.info("AI perception: loading ONNX model from %s...", model_path)
        engine = ONNXInferenceEngine(
            model_path=str(model_file),
            input_resolution=input_resolution,
            conf_threshold=conf_threshold,
        )
        engine.load()
        engine.warmup()
        logger.info("AI perception: ONNXInferenceEngine loaded successfully from %s.", model_path)

    detector = PersonDetector(
        engine=engine,
        target_class_ids=target_classes,
        conf_threshold=conf_threshold,
        inference_fps_target=fps_target,
        warmup_frames=warmup_frames,
    )
    logger.info(
        "PersonDetector initialised (conf_threshold=%.2f, fps_target=%.1f, input_res=%s).",
        conf_threshold,
        fps_target,
        input_resolution,
    )
    return detector


# ---------------------------------------------------------------------------
# Module C — Depth-Based Obstacle Detection
# ---------------------------------------------------------------------------


def _init_obstacle_detector(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise DepthProcessor and ObstacleDetector."""
    try:
        from aerosentinel.perception.depth_processor import DepthProcessor
        from aerosentinel.perception.obstacle_detector import ObstacleDetector

        depth_proc = DepthProcessor()
        obs_det = ObstacleDetector()
        logger.info("DepthProcessor + ObstacleDetector initialised.")
        return depth_proc, obs_det
    except Exception as exc:
        logger.error("Obstacle detector init failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Module D — RGB+Depth Fusion
# ---------------------------------------------------------------------------


def _init_fusion(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise RGBDFusion module."""
    try:
        from aerosentinel.perception.rgbd_fusion import RGBDFusion

        fusion_cfg = cfg.get("rgbd_fusion", {})
        fusion = RGBDFusion(config=fusion_cfg)
        logger.info("RGBDFusion initialised.")
        return fusion
    except Exception as exc:
        logger.error("Fusion init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Module E — IMU Processing
# ---------------------------------------------------------------------------


def _init_imu(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise IMUProcessor."""
    try:
        from aerosentinel.sensors.imu_processor import IMUProcessor

        imu_proc = IMUProcessor()
        logger.info("IMUProcessor initialised.")
        return imu_proc
    except Exception as exc:
        logger.error("IMU processor init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Module F — Navigation / Localization
# ---------------------------------------------------------------------------


def _init_localization(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise LocalizationManager."""
    try:
        from aerosentinel.localization.localization_interface import LocalizationManager

        nav_cfg = {}
        try:
            nav_cfg = {"source": cfg.get("navigation.localization_source", "none")}
        except Exception:
            pass

        loc_mgr = LocalizationManager(config=nav_cfg)
        logger.info("LocalizationManager initialised.")
        return loc_mgr
    except Exception as exc:
        logger.error("Localization init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Module G — Hazard Detection
# ---------------------------------------------------------------------------


def _init_hazard_detector(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise HazardDetector."""
    try:
        from aerosentinel.perception.hazard_detector import HazardDetector

        hd = HazardDetector()
        logger.info("HazardDetector initialised.")
        return hd
    except Exception as exc:
        logger.error("Hazard detector init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Module H — Mission Management + Module K — Storage
# ---------------------------------------------------------------------------


def _init_mission(cfg: ConfigLoader) -> tuple:
    """Initialise DatabaseManager and MissionManager."""
    try:
        from aerosentinel.storage.database import DatabaseManager
        from aerosentinel.mission.mission_manager import MissionManager

        db_path = cfg.get("mission.db_path", "data/aerosentinel.db")
        snapshot_dir = cfg.get("mission.snapshot_dir", "data/detections")

        db = DatabaseManager(db_path=db_path)
        db.connect()
        db.create_tables()

        mm = MissionManager(db_manager=db, snapshot_dir=snapshot_dir)
        logger.info("MissionManager + DatabaseManager initialised (db=%s).", db_path)
        return db, mm
    except Exception as exc:
        logger.error("Mission manager init failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Module I — MAVLink (Read-Only, Optional)
# ---------------------------------------------------------------------------


def _init_mavlink(cfg: ConfigLoader) -> Optional[Any]:
    """Initialise MAVLink read-only client if enabled in config."""
    mavlink_enabled = cfg.get("mavlink.enabled", False)
    if not mavlink_enabled:
        logger.info("MAVLink disabled in config (mavlink.enabled=false).")
        return None
    try:
        from aerosentinel.flight.mavlink_client import MAVLinkClient

        conn_str = cfg.get("mavlink.connection_string", "udp:0.0.0.0:14550")
        baud = cfg.get("mavlink.baud_rate", 57600)
        client = MAVLinkClient(connection_string=conn_str, baud_rate=baud, read_only=True)
        client.connect()
        logger.info("MAVLinkClient connected (read-only): %s", conn_str)
        return client
    except ImportError as exc:
        logger.warning("pymavlink not installed — MAVLink disabled. (%s)", exc)
        return None
    except Exception as exc:
        logger.error("MAVLink init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Module J — Ground Control Dashboard
# ---------------------------------------------------------------------------


def _init_dashboard(
    cfg: ConfigLoader,
    enabled: bool,
    system_state: Any,
    port: int = 8080,
) -> Optional[threading.Thread]:
    """
    Initialise and start the FastAPI dashboard in a background daemon thread.
    Returns the thread (already started) or None if disabled.
    """
    if not enabled:
        logger.info("Dashboard disabled (--no-dashboard).")
        return None
    try:
        from aerosentinel.dashboard.app import create_app
        import uvicorn

        dash_cfg = {
            "ws_video_fps": cfg.get("system.dashboard_video_fps", 7),
            "ws_telemetry_hz": cfg.get("system.dashboard_telemetry_hz", 2),
            "jpeg_quality": cfg.get("system.dashboard_jpeg_quality", 65),
        }
        app = create_app(config=dash_cfg, system_state=system_state)

        def _run_server() -> None:
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

        t = threading.Thread(target=_run_server, name="dashboard-server", daemon=True)
        t.start()
        logger.info(
            "Dashboard started on http://0.0.0.0:%d (background thread).", port
        )
        return t
    except ImportError as exc:
        logger.warning("FastAPI/uvicorn not installed — dashboard disabled. (%s)", exc)
        return None
    except Exception as exc:
        logger.error("Dashboard init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------


def _main_loop(
    monitor: PerformanceMonitor,
    camera_mgr: Optional[Any],
    detector: Optional[Any],
    depth_proc: Optional[Any],
    obs_det: Optional[Any],
    fusion: Optional[Any],
    imu_proc: Optional[Any],
    loc_mgr: Optional[Any],
    hazard_det: Optional[Any],
    mission_mgr: Optional[Any],
    mavlink_client: Optional[Any],
    system_state: Optional[Any],
    loop_hz: float = 10.0,
) -> None:
    """
    Main application event loop.

    Each tick (at ~loop_hz Hz):
      1. Acquire RGB + Depth frames
      2. Run AI detection
      3. Run depth/obstacle processing
      4. Fuse RGB + Depth
      5. Process IMU
      6. Assess hazards
      7. Update MAVLink telemetry → localization
      8. Update dashboard system state
    """
    global _running
    period = 1.0 / max(loop_hz, 1.0)
    logger.info("Main loop running at %.1f Hz (press Ctrl-C to stop)…", loop_hz)

    while _running:
        t0 = time.perf_counter()

        # ----------------------------------------------------------------
        # 1. Camera frames
        # ----------------------------------------------------------------
        rgb_frame = None
        depth_result = None

        if camera_mgr is not None:
            try:
                # Keep Pi Camera 3 acquisition operational
                rgb_result = camera_mgr.get_rgb_frame()
                if rgb_result is not None:
                    rgb_frame, _ = rgb_result

                depth_result = camera_mgr.get_depth_frame()
            except Exception as exc:
                logger.debug("Camera frame error: %s", exc)

        # ----------------------------------------------------------------
        # 2. AI detection (Module B) & RGB-D Fusion (Module D)
        # ----------------------------------------------------------------
        detections = []
        fused_detections = []
        annotated_frame = rgb_frame

        # Prioritize RealSense RGB + aligned Depth from the same acquisition cycle
        if depth_result is not None:
            try:
                depth_mm, rs_rgb, rs_ts = depth_result
                if detector is not None and rs_rgb is not None:
                    _, detections = detector.detect(rs_rgb)

                if fusion is not None and detections:
                    fused_detections = fusion.fuse(
                        detections, depth_mm, rgb_timestamp=rs_ts, depth_timestamp=rs_ts
                    )
                    annotated_frame = fusion.draw_fused_detections(rs_rgb, fused_detections)
                elif rs_rgb is not None:
                    annotated_frame = rs_rgb
            except Exception as exc:
                logger.debug("RealSense RGB-D detection/fusion error: %s", exc)
        elif detector is not None and rgb_frame is not None:
            # Fallback to Pi Camera RGB only when RealSense is unavailable
            try:
                annotated_frame, detections = detector.detect(rgb_frame)
            except Exception as exc:
                logger.debug("Detection error: %s", exc)

        # ----------------------------------------------------------------
        # 3. Depth + Obstacle processing (Module C)
        # ----------------------------------------------------------------
        obstacle_report = None

        if depth_result is not None and depth_proc is not None and obs_det is not None:
            try:
                depth_mm, color_bgr, _ts = depth_result
                # depth_proc.process() expects uint16 mm array
                processed_depth = depth_proc.process(depth_mm)
                depth_m = processed_depth.depth_meters   # float32 metres, 0=invalid
                obstacle_report = obs_det.analyze(depth_m)

                if system_state is not None and obstacle_report is not None:
                    system_state.latest_obstacle = {
                        "overall_status": obstacle_report.overall_status.name,
                        "warning_active": obstacle_report.warning_active,
                        "closest_obstacle_m": obstacle_report.closest_obstacle_m,
                        "left": {
                            "status": obstacle_report.left.status.name,
                            "min_distance_m": obstacle_report.left.min_distance_m,
                        },
                        "center": {
                            "status": obstacle_report.center.status.name,
                            "min_distance_m": obstacle_report.center.min_distance_m,
                        },
                        "right": {
                            "status": obstacle_report.right.status.name,
                            "min_distance_m": obstacle_report.right.min_distance_m,
                        },
                    }
                    # Push colourised depth visualisation to dashboard
                    if system_state is not None:
                        system_state.depth_frame = processed_depth.colormap
            except Exception as exc:
                logger.debug("Obstacle processing error: %s", exc)

        # ----------------------------------------------------------------
        # 4. Fallback Fusion (if not executed in Step 2)
        # ----------------------------------------------------------------
        if fusion is not None and detections and depth_result is not None and not fused_detections:
            try:
                depth_mm, _, _ts = depth_result
                fused_detections = fusion.fuse(detections, depth_mm, _ts, _ts)
            except Exception as exc:
                logger.debug("Fusion error: %s", exc)

        # ----------------------------------------------------------------
        # 5. IMU (Module E)
        # ----------------------------------------------------------------
        if camera_mgr is not None and imu_proc is not None:
            try:
                raw_imu = camera_mgr.get_imu_data()
                if raw_imu is not None:
                    processed_imu = imu_proc.process(raw_imu)
                    if system_state is not None:
                        system_state.latest_imu = {
                            "timestamp": processed_imu.timestamp,
                            "accel_x": processed_imu.accel_x,
                            "accel_y": processed_imu.accel_y,
                            "accel_z": processed_imu.accel_z,
                            "gyro_x": processed_imu.gyro_x,
                            "gyro_y": processed_imu.gyro_y,
                            "gyro_z": processed_imu.gyro_z,
                            "roll_estimate": processed_imu.roll_estimate,
                            "pitch_estimate": processed_imu.pitch_estimate,
                            "is_valid": processed_imu.is_valid,
                        }
            except Exception as exc:
                logger.debug("IMU processing error: %s", exc)

        # ----------------------------------------------------------------
        # 6. Hazard detection (Module G)
        # ----------------------------------------------------------------
        hazard_events = []

        if hazard_det is not None and (detections or obstacle_report is not None):
            try:
                hazard_events = hazard_det.assess(detections, obstacle_report)
                for event in hazard_events:
                    logger.info(
                        "HAZARD: %s confidence=%.2f dist=%s",
                        event.category.value,
                        event.confidence,
                        f"{event.distance_m:.2f}m" if event.distance_m else "N/A",
                    )
            except Exception as exc:
                logger.debug("Hazard assessment error: %s", exc)

        # ----------------------------------------------------------------
        # 7. MAVLink telemetry + localization (Modules F + I)
        # ----------------------------------------------------------------
        if mavlink_client is not None:
            try:
                tel = mavlink_client.get_telemetry()
                if system_state is not None:
                    system_state.latest_telemetry = tel
                    system_state.mavlink_connected = tel.get("is_connected", False)
                if loc_mgr is not None:
                    # Feed telemetry dict into localization manager as a simple namespace
                    class _TelProxy:
                        pass
                    proxy = _TelProxy()
                    for k, v in tel.items():
                        setattr(proxy, k, v)
                    loc_mgr.set_telemetry(proxy)
            except Exception as exc:
                logger.debug("MAVLink telemetry update error: %s", exc)

        # ----------------------------------------------------------------
        # 8. Update dashboard system state frames
        # ----------------------------------------------------------------
        if system_state is not None:
            if annotated_frame is not None:
                system_state.rgb_frame = annotated_frame
            if system_state.mission_active and mission_mgr is not None:
                # Record survivor detections during active mission
                active_mission = mission_mgr.get_active_mission()
                if active_mission is not None:
                    for fd in fused_detections:
                        try:
                            fd_dict = fd.to_dict()
                            det_dict = {
                                "class_name": "Potential Survivor",
                                "confidence": fd_dict.get("confidence", 0.0),
                                "distance_m": fd_dict.get("distance_m"),
                                "bbox_json": str(fd.detection.bbox),
                                "hazard_category": "PERSON_DETECTED",
                                "verification_status": "PENDING_HUMAN_REVIEW",
                                "lat": None,
                                "lon": None,
                            }
                            mission_mgr.record_detection(active_mission.id, det_dict)
                        except Exception as exc:
                            logger.debug("Detection record error: %s", exc)

        # ----------------------------------------------------------------
        # 9. Performance tick
        # ----------------------------------------------------------------
        monitor.tick()

        elapsed = time.perf_counter() - t0
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            monitor.drop_frame()
            logger.debug("Loop over-budget by %.1f ms", abs(sleep_time) * 1000)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    _print_banner()

    cfg = _load_config(args.config)

    if args.mock:
        cfg.set("system.mock_mode", True)
    if args.log_level:
        cfg.set("system.log_level", args.log_level)

    log_config: Dict[str, Any] = {
        "log_level": cfg.get("system.log_level", "INFO"),
        "log_dir": cfg.get("system.log_dir", "data/logs"),
    }
    setup_logging(log_config)
    logger.info("AeroSentinel v%s starting up…", _VERSION)
    logger.info(
        "Config dir: %s | Mock mode: %s",
        args.config,
        cfg.get("system.mock_mode", False),
    )

    _register_signal_handlers()

    mock = cfg.get("system.mock_mode", False)

    monitor = PerformanceMonitor(enable_background=True)
    monitor.start()

    # ----------------------------------------------------------------
    # Shared dashboard system state (populated by all modules)
    # ----------------------------------------------------------------
    try:
        from aerosentinel.dashboard.app import SystemState
        system_state = SystemState()
    except Exception:
        system_state = None

    try:
        # --- Module A: Cameras ---
        camera_mgr = _init_cameras(cfg, mock=mock)
        if system_state and camera_mgr is not None:
            cam_status = camera_mgr.status()
            system_state.rgb_camera_connected = cam_status.get("rgb", {}).get("connected", False)
            system_state.d435i_connected = cam_status.get("depth", {}).get("connected", False)

        # --- Module B: AI Perception ---
        detector = _init_perception(cfg, mock=mock)
        if system_state and detector is not None:
            system_state.ai_engine_running = True

        # --- Module C: Obstacle Detection ---
        depth_proc, obs_det = _init_obstacle_detector(cfg)
        if system_state and depth_proc is not None:
            system_state.depth_processing_running = True

        # --- Module D: Fusion ---
        fusion = _init_fusion(cfg)

        # --- Module E: IMU ---
        imu_proc = _init_imu(cfg)

        # --- Module F: Localization ---
        loc_mgr = _init_localization(cfg)

        # --- Module G: Hazard Detection ---
        hazard_det = _init_hazard_detector(cfg)

        # --- Module H + K: Mission + Storage ---
        db, mission_mgr = _init_mission(cfg)

        # --- Module I: MAVLink (read-only, optional) ---
        mavlink_client = _init_mavlink(cfg)

        # --- Module J: Dashboard ---
        dashboard_thread = _init_dashboard(
            cfg,
            enabled=not args.no_dashboard,
            system_state=system_state,
            port=args.dashboard_port,
        )

        logger.info("All modules initialised successfully. Starting main loop.")

        # --- Main Loop ---
        _main_loop(
            monitor=monitor,
            camera_mgr=camera_mgr,
            detector=detector,
            depth_proc=depth_proc,
            obs_det=obs_det,
            fusion=fusion,
            imu_proc=imu_proc,
            loc_mgr=loc_mgr,
            hazard_det=hazard_det,
            mission_mgr=mission_mgr,
            mavlink_client=mavlink_client,
            system_state=system_state,
            loop_hz=cfg.get("system.loop_hz", 10.0),
        )

    except Exception as exc:
        logger.exception("Fatal error in main: %s", exc)
        return 1
    finally:
        # --- Graceful teardown ---
        logger.info("Shutting down subsystems…")

        try:
            if mission_mgr is not None:
                mission_mgr.shutdown()
        except Exception as exc:
            logger.debug("Mission manager shutdown error: %s", exc)

        try:
            if mavlink_client is not None:
                mavlink_client.disconnect()
        except Exception as exc:
            logger.debug("MAVLink disconnect error: %s", exc)

        try:
            if camera_mgr is not None:
                camera_mgr.stop_all()
        except Exception as exc:
            logger.debug("Camera stop error: %s", exc)

        try:
            if detector is not None:
                detector.stop()
        except Exception as exc:
            logger.debug("Detector stop error: %s", exc)

        try:
            if db is not None:
                db.disconnect()
        except Exception as exc:
            logger.debug("DB disconnect error: %s", exc)

        monitor.stop()
        logger.info("AeroSentinel shutdown complete.")
        report = monitor.report()
        logger.info(
            "Session summary — uptime: %.1fs | FPS: %.2f | dropped: %d",
            report["uptime_s"],
            report["fps"],
            report["dropped_frames"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
