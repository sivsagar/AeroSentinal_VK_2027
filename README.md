# AeroSentinel — AI-Powered Autonomous Drone for Disaster Search & Rescue

**Team ID:** VK26-AU-PAI-0399  
**Track:** Physical AI for Resilience and Inclusion  
**Primary Platform:** Raspberry Pi 4 Model B (4GB, ARM64) | Raspberry Pi Camera Module 3 (CSI) | Intel RealSense D435i (USB 3.0)

[![Test Suite](https://img.shields.io/badge/pytest-166%20passed-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Companion%20Computer-Raspberry%20Pi%204B-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Offline Operation](https://img.shields.io/badge/Network-100%25%20Offline%20Capable-orange.svg)]()

---

## 1. System Overview

AeroSentinel is an edge-native, real-time perception and disaster search-and-rescue (SAR) software suite designed specifically for power- and compute-constrained UAV companion computers. It integrates high-resolution monocular RGB perception with active stereo infrared depth sensing and IMU telemetry to enable:

1. **Lightweight Human / Survivor Candidate Detection:** YOLOv8n ONNX inference optimized for ARM Cortex-A72 CPU execution (NEON accelerated, no CUDA required). All detections are labeled as *"Potential Survivor"* requiring human-in-the-loop verification.
2. **Real-Time Depth Obstacle Detection:** Region-of-Interest (ROI) spatial analysis dividing the forward corridor into Left, Center, and Right zones with calibrated Critical, Warning, and Safe distance thresholds.
3. **RGB-D Sensor Fusion:** Spatio-temporal association of 2D bounding boxes with depth point data to estimate distance (meters) to detected survivors without fabricating coordinates.
4. **Offline Local Mission Recording:** Fully functional without cellular or cloud connectivity. Autonomous mission recording into local SQLite storage with image snapshot capture and JSON/CSV mission report generation.
5. **Read-Only MAVLink Telemetry:** Passive listening for flight controller heartbeat, battery state, GPS fix (when available), and attitude without risking unauthorized motor actuation or flight interference.
6. **Responsive Ground Control Dashboard:** Lightweight FastAPI + WebSocket web dashboard served directly from the Raspberry Pi 4B over local Wi-Fi with live video, depth maps, telemetry, and system thermal monitoring.

---

## 2. Hardware Architecture & Pinout

```
+-----------------------------------------------------------------------------------+
|                            AEROSENTINEL HARDWARE TOPOLOGY                         |
|                                                                                   |
|    +-------------------------+                     +-------------------------+    |
|    | Raspberry Pi Camera 3   |                     |  Intel RealSense D435i  |    |
|    | (12MP Sony IMX708, CSI) |                     |  (Stereo Depth + IMU)   |    |
|    +------------+------------+                     +------------+------------+    |
|                 | 15-pin CSI                                    | USB 3.0 Blue    |
|                 v                                               v                 |
|    +-------------------------------------------------------------------------+    |
|    |                  Raspberry Pi 4 Model B (4GB RAM)                       |    |
|    |      Broadcom BCM2711, Quad-core Cortex-A72 @ 1.8GHz, Linux 64-bit      |    |
|    +------------------------------------+------------------------------------+    |
|                                         | UART /dev/ttyAMA0 (Read-Only)           |
|                                         v                                         |
|    +-------------------------------------------------------------------------+    |
|    |             Autopilot Flight Controller (ArduPilot / PX4)               |    |
|    |       Retains exclusive authority over stabilization, arming & failsafes |    |
|    +-------------------------------------------------------------------------+    |
+-----------------------------------------------------------------------------------+
```

### Hardware Constraints Respected
- **No Jetson / CUDA assumptions:** All neural inference is optimized for ONNX Runtime CPU execution on ARM64.
- **No Pi 5 assumptions:** Memory footprint capped to remain stable on 2GB–4GB Pi 4 boards.
- **No simulated hardware success:** If a physical camera or sensor is disconnected, health monitors report degraded/disconnected status; simulation/mock drivers exist in a strictly isolated module.
- **Read-Only Flight Controller Safety:** The companion computer never sends direct PWM or motor control commands.

---

## 3. Software Architecture (Modular Pipeline)

```
[ Pi Camera 3 ] ---> [ PicameraDriver ] \
                                         ---> [ RGBDFusion ] ---> [ HazardDetector ] ---> [ MissionRecorder ]
[ RealSense D435i ] -> [ RealSenseDriver] /         ^                                              |
      |                        | (Depth)            |                                              v
      v (IMU)                  v                    |                                       [ SQLite DB ]
[ IMUProcessor ]      [ DepthProcessor ]            |                                              |
      |                        |                    |                                              v
      v                        v                    |                                    [ ReportGenerator ]
[ HealthMonitor ]     [ ObstacleDetector ] --------+
                               |
                               v
                     [ SafetyMonitor ] <--- [ MAVLinkClient (Read-Only) ]
                               |
                               v
                     [ FastAPI + WebSocket Dashboard ] (Port 8080)
```

---

## 4. Quick Start Guide

### Prerequisites
- Raspberry Pi 4B running **Debian 13 (Trixie) ARM64** or **Raspberry Pi OS (64-bit, Bookworm)** / Ubuntu 22.04/24.04 ARM64.
- Python 3.9+ (tested on Python 3.13.5 ARM64).

### Installation

```bash
# 1. Clone repository
git clone https://github.com/AeroSentinel/AeroSentinel_VK_2027.git aerosentinel
cd aerosentinel

# 2. Run automated hardware & dependency diagnostics
bash scripts/check_hardware.sh

# 3. Install core dependencies in a Python venv
bash scripts/install_dependencies.sh

# 4. (Optional) Build RealSense D435i librealsense on Pi 4B
sudo bash scripts/setup_realsense.sh
```

### Execution

```bash
# Launch in Real Hardware Mode
bash scripts/start_aerosentinel.sh

# Launch in Full Mock / Simulation Mode (for laptop / non-Pi testing)
python -m aerosentinel.main --mock
```

Open your browser to:
`http://localhost:8080` (or `http://raspberrypi.local:8080`)

---

## 5. Running the Automated Test Suite

AeroSentinel contains **166 comprehensive automated tests** across unit, hardware-mock, and full pipeline simulation modules:

```bash
# Run the complete test suite
python -m pytest tests/ -v

# Run simulation pipeline tests
python -m pytest tests/simulation/ -v

# Run individual subsystems
python -m pytest tests/unit/test_config_loader.py
python -m pytest tests/unit/test_cameras.py
python -m pytest tests/unit/test_sensors.py
python -m pytest tests/unit/test_perception.py
python -m pytest tests/unit/test_mission.py
python -m pytest tests/unit/test_mavlink.py
python -m pytest tests/unit/test_dashboard.py
```

---

## 6. Safety Mandate

> [!CAUTION]
> **AeroSentinel operates in perception-only and read-only telemetry mode.**
>
> 1. The flight controller retains 100% exclusive authority over vehicle stabilization, motor actuation, arming/disarming, and failsafe Return-To-Launch (RTL).
> 2. Companion computer failure or crash **cannot** interrupt flight controller stabilization.
> 3. Detections are flagged as `Potential Survivor` requiring human-in-the-loop confirmation before dispatching search & rescue field personnel.
> 4. Invalid or noisy depth readings are never treated as safe clear air.

---

## 7. Performance Benchmarks on Raspberry Pi 4B

| Operation | Implementation | RPi 4 Target | Achieved / Feasible |
|-----------|----------------|--------------|----------------------|
| RGB Video Acquisition | Picamera2 (720p) | 15 FPS | 15 FPS (NEON queue) |
| Depth Acquisition | D435i (640x480) | 15 FPS | 15 FPS (USB 3.0) |
| YOLOv8n ONNX Inference | ONNX Runtime CPU (320x320) | 3–6 FPS | ~4.2 FPS (NEON, 4 threads) |
| Depth Obstacle Processing | NumPy Vectorized ROI | 15 FPS | >35 FPS (<28ms) |
| Dashboard Streaming | FastAPI + JPEG WS | 5–10 FPS | 7 FPS (~65% quality) |
| Overall CPU Temperature | Under sustained load | < 75°C | ~64°C (with fan/heatsink) |

---

## 8. Repository Structure

```
aerosentinel/
├── config/              # YAML configurations for all subsystems
├── src/aerosentinel/    # Core production package
│   ├── cameras/         # Pi Camera 3 & RealSense D435i drivers
│   ├── perception/      # YOLOv8n ONNX detector, depth, obstacle, fusion
│   ├── sensors/         # IMU processing & sensor health watchdogs
│   ├── localization/    # Modular localization interfaces & SLAM adapter stubs
│   ├── navigation/      # Advisory navigation interface & safety monitor
│   ├── flight/          # MAVLink read-only client & telemetry models
│   ├── mission/         # Mission manager, recorder, SQLite database, reports
│   ├── dashboard/       # FastAPI backend, WebSocket server & HTML5 UI
│   ├── storage/         # SQLite schema & query managers
│   └── utils/           # Config loader, logger, performance monitors
├── simulation/          # Mock camera, depth, and MAVLink simulators
├── scripts/             # Setup, benchmark, diagnostic, and startup scripts
├── tests/               # 166 passing unit & simulation tests
└── docs/                # Complete technical and operational documentation
```

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
