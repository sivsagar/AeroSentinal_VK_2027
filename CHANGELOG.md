# Changelog

All notable changes to AeroSentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.2] — 2026-11-14

### Fixed — Multi-Person Tracking Validation and Reliability

**Root causes addressed from hardware report (track IDs 6/7, age=3 frames, 60s→1.6s timestamps):**
- Root cause 1 — `valid_depth_percent=0.0` with `distance_valid=True`: Fixed in v0.1.1 (`FusedDetection.__post_init__` invariant). Confirmed in all 51 regression tests.
- Root cause 2 — `age_frames=3` after 60 s: Fixed in v0.1.1 (simulated timestamp injection). Now also fixed architecturally: `age_frames` increments on **every** tracker update (hit or missed), correctly representing track lifetime, not just matched frames.
- Root cause 3 — Timestamps spanning only 1–1.6 s: Fixed in v0.1.1 (`ts` parameter propagation). Covered by `TestReq6_Timestamps`.

### Added

- **`last_known_distance_m`** field on `TrackedPerson`: preserves the most recent valid EMA distance across invalid frames. `estimated_distance_m` is `None` when current frame is invalid; `last_known_distance_m` holds the last good reading for display/logging without implying validity.
- **`total_missed`** counter on `TrackedPerson`: cumulative missed frames since creation (never resets on reacquisition); distinct from `missed_frames` (consecutive only).
- **`total_detections_processed`**, **`total_depth_valid_detections`**, **`total_depth_invalid_detections`**, **`peak_concurrent_tracks`**, **`confirmed_track_count`**, **`tentative_track_count`**, **`lost_track_count`**, **`expired_track_count`** in `PersonTracker.lifecycle_summary()`.
- **`active_tracks()`**, **`lost_tracks()`**, **`expired_tracks()`** convenience accessors on `PersonTracker`.
- **`track_duration_s(now=None)`** method on `TrackedPerson`.
- **`config/tracking.yaml`**: dedicated tracker configuration file with all IoU, min-hits, max-missed, EMA-alpha, duplicate-suppression, depth range and coverage-threshold parameters documented.
- **`scripts/multi_person_tracking_test.py`**: complete diagnostic script producing JSON reports with session timing, FPS, dropped frames, YOLO inference counts, per-track depth history, and Pi execution commands.
- **`_CONFIG_FILES`** in `main.py` now includes `tracking.yaml`; `_init_tracker()` reads from `tracking.*` config keys.
- **51 regression tests** (`tests/simulation/multi_person_tracking_test.py`) covering all 12 required scenarios:
  1. Valid distance with nonzero valid-depth percentage
  2. Invalid depth with zero valid-depth percentage
  3. Stale distance preserved as `last_known_distance_m` after invalid frame
  4. Complete depth metadata propagation
  5. `age_frames` across hit AND missed updates
  6. First-seen / last-seen timestamps with injected and real wall-clock time
  7. Track creation and confirmation lifecycle events
  8. Temporary detection loss
  9. Reacquisition using the original track ID
  10. Track expiration
  11. Two-person tracking without ID duplication, peak concurrent count
  12. Accurate JSON lifecycle reporting with all required fields

### Test Results

- **217 passed, 0 failed** (pytest 9.2.0, Python 3.13)
- Previous baseline (v0.1.1): 189 passed. This release adds 28 new tests.

---



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
