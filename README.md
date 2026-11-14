# AeroSentinel — AI-Powered Autonomous Drone for Disaster Search & Rescue

**Team ID:** VK26-AU-PAI-0399  
**Track:** Physical AI for Resilience and Inclusion  
**Primary Target Platform:** Raspberry Pi 4 Model B (4GB / 8GB RAM, ARM64) | Debian 13 (Trixie) / Raspberry Pi OS 64-bit | Python 3.13  
**Sensor Suite:** Raspberry Pi Camera Module 3 Wide (Sony IMX708, CSI-2) | Intel RealSense D435i (USB 3.0 Active Stereo Depth + IMU)  

[![Test Suite](https://img.shields.io/badge/pytest-232%20passed-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Companion%20Computer-Raspberry%20Pi%204B-red.svg)]()
[![OS](https://img.shields.io/badge/OS-Debian%2013%20ARM64-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)]()
[![Inference](https://img.shields.io/badge/Inference-ONNX%20Runtime%20CPU%20(NEON)-orange.svg)]()
[![MAVLink](https://img.shields.io/badge/MAVLink-Strictly%20Read--Only-success.svg)]()
[![Offline Operation](https://img.shields.io/badge/Network-100%25%20Offline%20Capable-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

---

## 1. System Overview

**AeroSentinel** is an edge-native, real-time aerial perception and disaster search-and-rescue (SAR) software platform engineered specifically for power- and compute-constrained UAV companion computers. Designed to operate completely disconnected from cloud infrastructure, AeroSentinel fuses high-resolution wide-angle RGB search vision with active stereo infrared depth imaging and IMU telemetry to provide actionable situational awareness for emergency responders.

### Core Capabilities
1. **Lightweight Human / Survivor Detection:** Hardware-optimized YOLOv8n ONNX inference running on ARM Cortex-A72 CPU cores with NEON SIMD acceleration (no CUDA or external accelerator needed). Detections are labeled as *"Potential Survivor"* requiring human-in-the-loop verification.
2. **RGB-D 3D Distance Estimation:** Spatio-temporal association of 2D bounding boxes with aligned 16-bit millimeter-precision depth frames (Z16 format) to determine physical distance (in meters) to detected individuals.
3. **Multi-Person Tracking:** Multi-target tracker providing persistent track IDs across successive frames, tracking age, IoU/center-distance matching, detection loss coasting, reacquisition, and Exponential Moving Average (EMA) distance smoothing.
4. **Spatial Obstacle Proximity Assessment:** Depth-based spatial grid analysis dividing the field-of-view into Left, Centre, and Right sectors (with optional 3×3 grid support), calculating 10th-percentile distances, valid-depth coverage, and near-range occupancy to produce advisory warnings (`CLEAR`, `CAUTION`, `WARNING`, `CRITICAL`).
5. **Unified Ground Control Robotics Dashboard:** Dark-mode cockpit dashboard powered by FastAPI and WebSockets, rendering live annotated RealSense RGB video, false-color depth maps, dedicated Pi Camera 3 search feeds, multi-person tracking tables, regional hazard grids, hardware thermals, and deduplicated event timelines.
6. **Strictly Read-Only MAVLink Telemetry:** Passive listening over UART (`/dev/ttyAMA0`) for vehicle heartbeat, battery percentage, GPS position, and altitude. Absolutely zero flight-control authority, arming, or motor actuation commands can be issued by the companion computer.
7. **Offline SQLite Mission Recording:** Standalone, zero-cloud storage saving telemetry snapshots, candidate detections, obstacle events, and tracking metadata with automatic CSV/JSON report generation.

---

## 2. Hardware Architecture & Pinout

```
+-----------------------------------------------------------------------------------+
|                            AEROSENTINEL HARDWARE TOPOLOGY                         |
|                                                                                   |
|    +-------------------------+                     +-------------------------+    |
|    | Raspberry Pi Camera 3   |                     |  Intel RealSense D435i  |    |
|    | (12MP Sony IMX708 Wide) |                     |  (Stereo Depth + IMU)   |    |
|    +------------+------------+                     +------------+------------+    |
|                 | 15-pin CSI-2 Ribbon                           | USB 3.0 (Blue)  |
|                 v                                               v                 |
|    +-------------------------------------------------------------------------+    |
|    |                  Raspberry Pi 4 Model B (4GB / 8GB RAM)                 |    |
|    |      Broadcom BCM2711, Quad-core Cortex-A72 @ 1.8GHz, Debian 13 ARM64    |    |
|    +------------------------------------+------------------------------------+    |
|                                         | UART /dev/ttyAMA0 (Read-Only RX)        |
|                                         v                                         |
|    +-------------------------------------------------------------------------+    |
|    |             Autopilot Flight Controller (ArduPilot / PX4)               |    |
|    |       Retains exclusive authority over stabilization, arming & failsafes |    |
|    +-------------------------------------------------------------------------+    |
+-----------------------------------------------------------------------------------+
```

### Verified Hardware Configuration
- **Raspberry Pi 4 Model B:** Quad-core Cortex-A72 @ 1.8 GHz, 4GB/8GB RAM, Debian 13 (Trixie) ARM64 / Raspberry Pi OS 64-bit.
- **Pi Camera Module 3 Wide:** Sony IMX708, 120° diagonal FOV, 1280×720 @ 15 FPS via `libcamera` / `Picamera2`.
- **Intel RealSense D435i:**
  - Color Stream: 640×480 @ 15 FPS (RGB8)
  - Depth Stream: 640×480 @ 15 FPS (Z16 16-bit aligned depth)
  - Accelerometer: 250 Hz (configured to resolve hardware-compatible stream rates)
  - Gyroscope: 400 Hz
- **Camera-Relative Orientation:** RealSense camera is mounted facing backwards relative to the airframe forward flight vector. Sector proximity assessment preserves strict camera-relative semantics (`Left`, `Centre`, `Right`) to prevent incorrect coordinate inversions.

---

## 3. Software Subsystem Architecture

```
                                    +-------------------------------------------------------------+
                                    |                 AeroSentinel Core Service                   |
                                    +-------------------------------------------------------------+
                                                                  |
              +---------------------------------------------------+---------------------------------------------------+
              |                                                   |                                                   |
              v                                                   v                                                   v
   +----------------------+                           +-----------------------+                           +-----------------------+
   |  Picamera2 Driver    |                           | RealSense D435i Driver|                           |    MAVLink Driver     |
   | Wide FOV Search Feed |                           | RGB (640x480) & Depth |                           | Strict Read-Only Mode |
   +----------------------+                           +-----------------------+                           +-----------------------+
              |                                                   |                                                   |
              | Raw Frames                                        | Aligned Depth + Color                             | Telemetry (No PWM/Arm)
              v                                                   v                                                   v
   +----------------------+                           +-----------------------+                           +-----------------------+
   | Pi Camera Processor  |                           |  RGB-D Fusion Pipeline|                           | Telemetry Receiver    |
   | 1280x720 @ 15 FPS    |                           | (Aligned 16-bit Z16)  |                           | Alt, Lat, Lon, Batt   |
   +----------------------+                           +-----------------------+                           +-----------------------+
              |                                                   |                                                   |
              |                                     +-------------+-------------+                                     |
              |                                     |                           |                                     |
              |                                     v                           v                                     |
              |                           +-------------------+       +-------------------+                           |
              |                           | YOLOv8n Detection |       | Depth Proximity   |                           |
              |                           | ONNX Runtime CPU  |       | Spatial Grid      |                           |
              |                           +-------------------+       +-------------------+                           |
              |                                     |                           |                                     |
              |                                     v                           | Left / Centre / Right               |
              |                           +-------------------+                 | Sectors & Hazard Class              |
              |                           | Person Tracker    |                 |                                     |
              |                           | (EMA Distance)    |                 |                                     |
              |                           +-------------------+                 |                                     |
              |                                     |                           |                                     |
              +-------------------------------------+-------------+-------------+-------------------------------------+
                                                                  |
                                                                  v
                                              +---------------------------------------+
                                              |       SystemState (Thread-Safe)       |
                                              | - Active Tracks & Bounding Boxes      |
                                              | - Multi-Camera Frame Buffers          |
                                              | - Proximity Assessment & Events Queue |
                                              | - System Health & Metric Freshness    |
                                              +---------------------------------------+
                                                                  |
                                       +--------------------------+--------------------------+
                                       |                                                     |
                                       v                                                     v
                       +-------------------------------+                     +-------------------------------+
                       |      FastAPI REST Router      |                     |    WebSocket Broadcaster      |
                       |  /api/health, /api/tracks,    |                     |   ws://.../ws/telemetry       |
                       |  /api/obstacle, /api/video/*  |                     |  Multi-topic pushed @ 10 Hz   |
                       +-------------------------------+                     +-------------------------------+
                                       |                                                     |
                                       +--------------------------+--------------------------+
                                                                  |
                                                                  v
                                              +---------------------------------------+
                                              |  Unified Dark-Mode Web Dashboard UI   |
                                              |  Cockpit Grid, HUD Overlays, Events   |
                                              +---------------------------------------+
```

---

## 4. Quick Start & Execution

### Prerequisites & Environment Setup
- Target OS: Debian 13 (Trixie) ARM64 or Raspberry Pi OS (64-bit, Bookworm).
- Python: Version 3.9+ (tested on Python 3.13.1 / 3.13.5 ARM64).

```bash
# 1. Clone repository
git clone https://github.com/AeroSentinel/AeroSentinel_VK_2027.git aerosentinel
cd aerosentinel

# 2. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Execution Modes

#### A. Real Hardware Mode
Runs physical Picamera2, RealSense D435i camera, YOLOv8n ONNX inference, and MAVLink telemetry:
```bash
python src/aerosentinel/main.py --config config/
```

#### B. Full Mock / Simulation Mode
Runs with synthetic video generators and software-simulated sensors for offline development:
```bash
python src/aerosentinel/main.py --mock --config config/
```

#### C. Web Dashboard Access
Open any modern browser and navigate to:
```
http://localhost:8080   (or http://<pi4-ip>:8080)
```

---

## 5. Standalone Hardware Diagnostics & Bench Tools

AeroSentinel provides standalone diagnostic scripts to validate individual subsystems without starting the entire aerial mission loop:

### 1. Hardware Integration Smoke Test
Runs a comprehensive 60-second dual-camera acquisition, ONNX inference, IMU stream check, and thermal profiling test:
```bash
python scripts/hardware_smoke_test.py
```
Outputs a timestamped diagnostic summary in `data/diagnostics/`.

### 2. Multi-Person Tracking Diagnostic
Evaluates person detection, bounding box association, persistent track ID assignment, and distance estimation:
```bash
# Real hardware mode
python scripts/person_tracking_test.py --duration 30

# Mock mode
python scripts/person_tracking_test.py --mock --duration 30
```
Saves annotated frames in `data/tracking_output/` and exports tracking telemetry to JSON.

### 3. Obstacle Proximity Assessment Diagnostic
Runs spatial depth grid analysis (Left, Centre, Right), calculating 10th-percentile distances and near occupancy percentages:
```bash
# Real hardware mode
python scripts/obstacle_proximity_test.py --duration 30

# Mock mode
python scripts/obstacle_proximity_test.py --mock --duration 30
```
Saves colorized depth maps and annotated frames in `data/obstacle_output/`.

---

## 6. REST API & WebSocket Protocol Reference

The FastAPI dashboard exposes high-performance REST and streaming endpoints:

### REST Endpoints
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/status` | `GET` | Overall system state, mission ID, and subsystem summary. |
| `/api/health` | `GET` | Hardware CPU/RAM/Temp utilization, FPS counters, dropped frames, subsystem health, and metric staleness flag. |
| `/api/perception` | `GET` | Latest YOLO detections, survivor candidates, confidence, and distance. |
| `/api/tracks` | `GET` | Active multi-person tracks with persistent IDs, bbox, state, and smoothed distance. |
| `/api/obstacle` | `GET` | Regional proximity assessment (Left, Centre, Right), 10th percentile depth, and alert levels. |
| `/api/events` | `GET` | Chronological deduplicated mission events from SQLite storage (`?limit=50`). |
| `/api/video/rgb` | `GET` | RealSense RGB live MJPEG stream with detection/tracking HUD overlays. |
| `/api/video/depth`| `GET` | RealSense Depth live MJPEG stream with Turbo colorization. |
| `/api/video/picam`| `GET` | Dedicated Raspberry Pi Camera Module 3 Wide live MJPEG search stream. |

### WebSocket Protocol (`ws://<ip>:8080/ws/telemetry`)
Pushed at 10 Hz with typed JSON packets:
- `health`: Hardware utilization, temperatures, FPS counters, and subsystem health status.
- `tracks`: Actively tracked human candidates with persistent IDs and 3D distances.
- `obstacle`: 1×3 spatial proximity grid status (`CLEAR`, `CAUTION`, `WARNING`, `CRITICAL`).
- `events`: Deduplicated mission alert events with severity flags.
- `telemetry`: Read-only MAVLink state (altitude, heading, groundspeed, battery).
- `video_frame` / `picam_frame`: Base64 JPEG frames for fallback UI rendering.

---

## 7. Safety Mandate & Invariants

> [!CAUTION]
> **AeroSentinel is strictly an advisory perception and situational awareness system.**
>
> 1. **Zero Flight-Control Authority:** The companion computer never sends direct PWM, motor control, mode change, or arming commands.
> 2. **Isolated MAVLink Client:** Operates exclusively in passive read-only mode over UART.
> 3. **Non-Inverted Camera Coordinates:** The RealSense depth sensor is mounted facing backwards; sector proximity (`Left`, `Centre`, `Right`) is maintained in the camera's actual optical frame to prevent uncalibrated coordinate errors.
> 4. **Fail-Safe Sensor Degradation:** Incomplete, noisy, or zero depth readings are never assumed to be clear air; invalid areas are reported honestly as `UNKNOWN` or degraded coverage.
> 5. **Human-in-the-Loop:** All detections are classified as `Potential Survivor` requiring visual human verification before rescue dispatch.

---

## 8. Automated Test Suite (232 Passing Tests)

AeroSentinel contains **232 comprehensive automated tests** across unit, hardware-mock, safety, and full end-to-end simulation pipelines:

```bash
# Execute entire test suite
pytest tests/ -v

# Run by subsystem
pytest tests/unit/test_dashboard.py -v         # Dashboard, REST APIs, WebSockets
pytest tests/unit/test_person_tracking.py -v   # Multi-person tracking & reacquisition
pytest tests/unit/test_obstacle_detector.py -v # Spatial depth proximity assessment
pytest tests/unit/test_rgbd_fusion.py -v       # RGB-D alignment & distance estimation
pytest tests/unit/test_inference_engine.py -v  # ONNX inference & model loading
pytest tests/unit/test_cameras.py -v           # Picamera2 & RealSense D435i drivers
pytest tests/unit/test_mavlink_safety.py -v    # Read-only MAVLink safety guarantees
pytest tests/simulation/ -v                    # Full end-to-end multi-rate simulation
```

---

## 9. Performance Benchmarks on Raspberry Pi 4 Model B

| Pipeline Component | Configuration | Target FPS | Pi 4B Achieved |
| :--- | :--- | :--- | :--- |
| **Pi Camera 3 Wide** | 1280×720 RGB via Picamera2 | 15 FPS | 15.0 FPS |
| **RealSense RGB** | 640×480 RGB8 via librealsense | 15 FPS | 14.8 FPS |
| **RealSense Depth** | 640×480 Z16 Aligned Depth | 15 FPS | 14.8 FPS |
| **RealSense IMU** | Accel: 250 Hz, Gyro: 400 Hz | 250 / 400 Hz | Stable, zero drops |
| **YOLOv8n Inference** | ONNX Runtime CPU (320×320, 4 threads) | 3–5 FPS | ~3.4–4.2 FPS |
| **RGB-D Fusion & Tracking** | Median ROI Depth + Kalman/EMA Tracker | 15 FPS | > 40 FPS (< 25 ms) |
| **Obstacle Proximity** | Vectorized NumPy Spatial Grid ROI | 15 FPS | > 50 FPS (< 20 ms) |
| **Web Dashboard** | FastAPI + WebSockets + MJPEG (Dual Feed) | 10 FPS | 10.0 FPS |
| **Thermal Profile** | Sustained load with passive heatsink + fan | < 75°C | ~54–62°C |

---

## 10. Repository Structure

```
AeroSentinel_VK_2027/
├── config/                  # YAML configurations (cameras, AI, depth, obstacle, tracking)
├── src/aerosentinel/        # Core production package
│   ├── cameras/             # Pi Camera 3 Wide & RealSense D435i drivers
│   ├── perception/          # YOLOv8n ONNX engine, RGB-D fusion, obstacle detector, tracking
│   ├── sensors/             # IMU processing, dropout detection & sensor health watchdogs
│   ├── localization/        # Modular localization interfaces & SLAM adapter stubs
│   ├── navigation/          # Advisory navigation interface & proximity safety monitors
│   ├── flight/              # MAVLink read-only client & telemetry models
│   ├── mission/             # Mission manager, recorder & SQLite database persistence
│   ├── dashboard/           # FastAPI backend, WebSocket router, HTML5 dark cockpit UI
│   ├── storage/             # SQLite schema, query managers & event models
│   └── utils/               # Config loader, logger, performance & thermal monitors
├── scripts/                 # Standalone hardware diagnostics, tracking & obstacle tests
├── simulation/              # Mock camera, depth, and MAVLink simulators
├── tests/                   # 232 passing unit, regression & simulation tests
│   ├── unit/                # Subsystem unit tests
│   └── simulation/          # End-to-end pipeline simulation tests
└── models/                  # YOLOv8n ONNX detection models
```

---

## 11. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
