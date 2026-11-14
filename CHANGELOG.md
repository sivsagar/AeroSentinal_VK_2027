# Changelog

All notable changes to AeroSentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.1] — 2026-11-14

### Fixed — RGB-D Depth Metadata Propagation & Tracking Lifecycle (bug report VK26-TRACK-001)

- **`RGBDFusion` (`rgbd_fusion.py`):** Added `valid_depth_percent` field to `FusedDetection`; introduced `__post_init__` safety invariant that forces `distance_valid=False` and `depth_method="invalid"` whenever `valid_depth_percent == 0`, `distance_m` is `None`/`NaN`/non-positive, or depth coverage falls below `min_valid_depth_pct` (default 5 %). Added `math.isfinite()` guard on all distance values.
- **`PersonTracker` (`person_tracker.py`) — new module:** IoU-based multi-person tracker with correct depth metadata propagation into every `TrackedPerson`. Stale EMA distance is never reported as `distance_valid=True` when the current frame is invalid. `mark_missed()` explicitly resets all depth fields. Monotonic track IDs via thread-safe global counter.
- **Timestamp injection:** `PersonTracker.update(ts=...)` and `TrackedPerson.update_from_fused(ts=...)` accept an optional simulated timestamp so the 60-second lifecycle test reports correct `age_seconds` without real wall-clock sleep.
- **Track creation timestamps:** New `TrackedPerson` objects initialise `first_seen_ts` and `last_seen_ts` from the injected `ts` (or `time.time()` in production), so `age_seconds` spans the actual mission duration.
- **Lifecycle events:** `created`, `confirmed`, `lost`, `reacquired`, `expired` events are recorded per-track and aggregated in `PersonTracker.lifecycle_summary()` as a JSON-serialisable dict.

### Added

- **`PersonTracker` module:** `TrackState`, `LifecycleEvent`, `TrackedPerson`, `PersonTracker` exported from `aerosentinel.perception`.
- **`main.py`:** `_init_tracker()` function; `PersonTracker` wired into `_main_loop()` between fusion (step 4) and IMU (step 5); confirmed tracks written to `system_state.recent_detections` each tick.
- **Regression tests** (`tests/simulation/multi_person_tracking_test.py` — 23 tests):
  - `test_60s_lifecycle_report`: 600-frame simulated 60-second session with two persons, sensor dropout, loss, and reacquisition.
  - `test_depth_dropout_does_not_make_valid`: All-zero depth must yield `distance_valid=False`.
  - `test_sparse_depth_below_threshold_is_invalid`: Coverage < 5 % threshold must yield `distance_valid=False`.
  - `test_valid_depth_percent_never_zero_when_distance_valid_true`: 300-frame mixed invariant check.
  - `TestFusedDetectionInvariant` (4 tests), `TestTrackedPersonDepthPropagation` (5 tests), `TestPersonTrackerLifecycle` (5 tests), `TestRGBDFusionValidDepthPercent` (4 tests).

### Test Results

- **189 passed, 0 failed** (pytest 9.2.0, Python 3.13, Windows/ARM64 compatible)
- Previous baseline: 166 passed. New tracking suite adds 23 tests.

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
