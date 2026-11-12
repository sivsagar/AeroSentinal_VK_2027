# Changelog

All notable changes to AeroSentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-11-12

### Added
- **Core System Architecture & Foundation (Phase 1):**
  - Dot-notation YAML `ConfigLoader` with environment variable overrides.
  - Multi-handler structured logging with rotating file limits.
  - Runtime `PerformanceMonitor` measuring FPS, latency, CPU %, RAM %, and SoC temperature.
  - Complete configuration files in `config/` (system, camera, depth, ai, fusion, navigation, mavlink, mission).
  - Production `main.py` entry point with graceful SIGINT/SIGTERM handlers and CLI options.
- **Sensor Drivers & Acquisition (Phase 2):**
  - `PiCameraDriver` with Picamera2/libcamera CSI acquisition.
  - `RealSenseCameraDriver` with pyrealsense2 stereo depth & IMU streaming.
  - `CameraManager` coordinating independent threads and bounded queues.
  - `IMUProcessor` low-pass filtering accelerometer/gyroscope readings with tilt calculation.
  - `SensorHealthMonitor` heartbeat detection.
  - Complete software simulation mock cameras (`MockRGBCamera`, `MockDepthCamera`, `MockIMU`).
- **AI Perception & RGB-D Fusion (Phase 3):**
  - `InferenceEngine` abstracting ONNX Runtime CPU execution for YOLOv8n.
  - `PersonDetector` generating "Potential Survivor" candidate overlays.
  - `DepthProcessor` with multi-stage filter chains (decimation, spatial, temporal, hole filling) and colormaps.
  - `ObstacleDetector` evaluating forward corridor sectors (Left, Center, Right).
  - `RGBDFusion` associating 2D bounding boxes with depth point data.
  - `HazardDetector` assigning unique IDs and human-verification status.
- **Mission Management & Storage (Phase 4):**
  - Thread-safe SQLite `DatabaseManager` in WAL mode.
  - `MissionManager` orchestrating mission lifecycles (Start, Pause, Resume, Stop).
  - `MissionRecorder` saving detection snapshots and metadata.
  - `ReportGenerator` creating JSON and CSV export summaries.
- **Ground Control Dashboard (Phase 5):**
  - FastAPI server serving single-page HTML5/CSS3/JavaScript dark-themed UI.
  - WebSocket manager broadcasting live RGB frames, depth colormaps, and telemetry.
- **MAVLink Integration (Phase 6):**
  - Passive read-only MAVLink client ingesting HEARTBEAT, BATTERY_STATUS, ATTITUDE, and GPS_RAW_INT.
  - Hardened safety lock blocking unauthorized flight actuation commands.
- **Optimization & Deployment (Phases 7 & 8):**
  - Automated diagnostic and installation scripts (`check_hardware.sh`, `install_dependencies.sh`, `setup_realsense.sh`, `start_aerosentinel.sh`, `benchmark_pi4.sh`).
  - Comprehensive documentation suite in `docs/`.
  - **166 passing automated tests** across unit, hardware-mock, and end-to-end simulation suites.
