# AeroSentinel — Hardware Compatibility Guide

> **Version:** 0.1.0  
> **Platform:** Raspberry Pi 4B + Pi Camera Module 3 + Intel RealSense D435i  
> **OS:** Raspberry Pi OS Bookworm 64-bit (Debian 12)  
> **Updated:** 2026-11-12

---

## Table of Contents

1. [Raspberry Pi 4B Hardware Limits](#1-raspberry-pi-4b-hardware-limits)
2. [Pi Camera Module 3](#2-pi-camera-module-3)
3. [Intel RealSense D435i](#3-intel-realsense-d435i)
4. [AI Inference Feasibility on RPi 4](#4-ai-inference-feasibility-on-rpi-4)
5. [Alternative Lightweight Models](#5-alternative-lightweight-models)
6. [Known Incompatibilities & Workarounds](#6-known-incompatibilities--workarounds)
7. [Verified Package Versions](#7-verified-package-versions)
8. [librealsense Source Build Procedure](#8-librealsense-source-build-procedure)

---

## 1. Raspberry Pi 4B Hardware Limits

### CPU

| Property | Value |
|---|---|
| Model | Broadcom BCM2711 |
| Architecture | ARM Cortex-A72 (ARMv8-A, 64-bit) |
| Cores | 4 |
| Base Clock | 1.5 GHz |
| Overclocked (config.txt) | Up to 1.8–2.0 GHz (requires active cooling + stable 5V power) |
| L2 Cache | 1 MB |
| CUDA | ❌ **Not available** |
| OpenCL | ❌ Limited (VideoCore VI not useful for ML) |
| NEON SIMD | ✅ Available (used by ONNX Runtime, OpenCV) |

> **Critical constraint:** All AI inference must run on CPU with NEON SIMD acceleration. No GPU offloading is possible without an external accelerator (Coral USB, Hailo-8L).

### Memory

| Property | Value |
|---|---|
| RAM | 4 GB LPDDR4-3200 |
| Memory Bandwidth | ~25.6 GB/s |
| Swap | Configured to 2 GB on SSD (avoid SD card swap — wears out quickly) |

**Memory budget for AeroSentinel:**

| Component | Approx. RAM |
|---|---|
| OS + system daemons | 400 MB |
| Python 3.9 runtime | 150 MB |
| OpenCV + NumPy | 100 MB |
| YOLOv8n ONNX model | 50 MB |
| ONNX Runtime session | 100 MB |
| librealsense2 frame buffers | 60 MB |
| Picamera2 frame buffers | 30 MB |
| SQLite + WAL buffer | 20 MB |
| FastAPI + Uvicorn | 80 MB |
| Headroom | 500 MB |
| **Total** | **~1.49 GB** |

### Storage

| Recommendation | Detail |
|---|---|
| Boot/OS | 32 GB+ microSD Class 10 / A2 |
| Data storage | USB 3.0 SSD (Samsung T7, WD My Passport) — faster write, no SD wear |
| Log rotation | Max 100 MB per log file, 10 rotations = 1 GB total log space |

### Thermal

| Threshold | Action |
|---|---|
| < 70°C | Normal operation |
| 70–80°C | Warning logged every 60 s |
| 80–85°C | AeroSentinel reduces AI inference target FPS |
| > 85°C | RPi OS throttles CPU frequency |
| > 90°C | Emergency shutdown risk |

**Mandatory:** Active cooling (metal heatsink block + 5V fan). Passive cooling alone is insufficient for sustained full-load SAR operations.

### Power

| Component | Current Draw |
|---|---|
| RPi 4B (full load) | ~1.2 A @ 5V = 6 W |
| Pi Camera 3 (CSI) | ~0.3 A @ 5V = 1.5 W |
| RealSense D435i (USB 3.0) | ~0.9 A @ 5V = 4.5 W |
| Fan (5V) | ~0.2 A |
| **Total** | **~2.6 A @ 5V = 13 W** |

**Minimum PSU:** 5 V / 3.5 A USB-C. On a drone: use a 5 V BEC with ≥ 4 A continuous rating.

---

## 2. Pi Camera Module 3

### Specifications

| Property | Value |
|---|---|
| Sensor | Sony IMX708 |
| Megapixels | 12 MP |
| Sensor Size | 1/2.43" |
| Native Resolution | 4608 × 2592 |
| AeroSentinel Working Res | 1920 × 1080 @ 30 FPS |
| AI Input Res | 320 × 320 (resized in software) |
| Interface | MIPI CSI-2 (ribbon cable to CSI port) |
| FOV | 102° diagonal (wide angle variant) |
| Auto Focus | Yes (PDAF) — disable for drone stability: `Picamera2.set_controls({"AfMode": 0})` |
| HDR | Supported via `Picamera2` |
| Night Mode | Low-noise mode via gain control |

### Software Stack

```
Sony IMX708 sensor
      │
      ▼
  libcamera (kernel driver + ISP pipeline)
      │  /dev/video0 (or /dev/media*)
      ▼
  Picamera2 (Python API for libcamera)
      │  camera.capture_array() → numpy ndarray
      ▼
  AeroSentinel Module A (camera_manager.py)
```

### Key Picamera2 Configuration Notes

- **Install:** `sudo apt install python3-picamera2` (do NOT install via pip on Bookworm — use apt)
- **Preview:** Disable preview window in headless mode: `Picamera2(0).start_preview(False)`
- **Formats:** Use `RGB888` for direct NumPy/OpenCV compatibility
- **Autofocus off:** Essential for stable drone footage — fix focus at ~3–5 m for SAR
- **Buffer count:** Set `buffer_count=4` for smooth streaming; more buffers = more RAM
- **Thread safety:** `Picamera2` is not thread-safe; confine all calls to a single dedicated thread

### libcamera Version Requirements

| Package | Min Version | Notes |
|---|---|---|
| `libcamera0` | 0.0.5+ | Available in Bookworm apt |
| `python3-picamera2` | 0.3.12+ | Available in Bookworm apt |
| `rpicam-apps` | 1.2.0+ | Optional; for `rpicam-still` / `rpicam-vid` diagnostics |

---

## 3. Intel RealSense D435i

### Specifications

| Property | Value |
|---|---|
| Type | Active IR stereo depth + RGB + IMU |
| Depth Sensor | OV9282 (global shutter, monochrome) ×2 |
| Depth Range | 0.1 – 10 m (optimal: 0.3 – 3 m for SAR) |
| Depth Resolution | Up to 1280×720 @ 30 FPS; AeroSentinel uses **640×480 @ 15 FPS** |
| RGB Sensor | OV2740 |
| RGB Resolution | Up to 1920×1080; AeroSentinel uses **848×480 @ 15 FPS** |
| IMU | Bosch BMI055 (3-axis accel @ 250 Hz, 3-axis gyro @ 400 Hz) |
| Interface | **USB 3.1 Gen 1 (USB 3.0 required)** |
| USB Bandwidth | ~3.2 Gbps at max res; AeroSentinel config uses ~200 MB/s |
| Connector | USB-C (use USB-C to USB-A for RPi 4) |

### USB Port Assignment

> **Critical:** Always connect D435i to a **USB 3.0 (blue) port** on RPi 4. Using USB 2.0 (black ports) will cause:
> - Reduced depth resolution (max 480×270)
> - Reduced FPS (max 6 FPS)
> - Frequent USB bandwidth errors

Recommended physical port assignment:
- **USB 3.0 Port 1:** RealSense D435i
- **USB 3.0 Port 2:** SSD storage (if used)
- **USB 2.0 Ports:** Keyboard, mouse (dev only)

### librealsense ARM64 Build

Pre-built `pyrealsense2` wheels **are not available** for ARM64 on PyPI as of 2026. The library must be compiled from source.

See [Section 8](#8-librealsense-source-build-procedure) for the full build procedure.

**Critical build notes for RPi 4:**
- Use `librealsense` v2.54.x or later (earlier versions have ARM64 issues)
- Disable `BUILD_GRAPHICAL_EXAMPLES=OFF` (no display on headless RPi)
- Enable `BUILD_PYTHON_BINDINGS=ON`
- Disable `BUILD_WITH_CUDA=OFF` (no CUDA)
- Build takes ~60–90 minutes on RPi 4 (4 parallel jobs)
- Ensure at least 4 GB of swap during build (risk of OOM with only RAM)

### pyrealsense2 Usage Notes

- After source build, the `.so` wheel file is in `build/wrappers/python/`
- Install via: `pip install build/wrappers/python/pyrealsense2-*.whl`
- Or copy the `.so` to site-packages manually
- Test: `python3 -c "import pyrealsense2 as rs; print('pyrealsense2 imported successfully')"`

---

## 4. AI Inference Feasibility on RPi 4

### YOLOv8n ONNX — CPU Inference Benchmarks

**Model:** YOLOv8n (nano variant), exported to ONNX FP32  
**Runtime:** ONNX Runtime 1.16+ with CPU execution provider (NEON SIMD)

| Input Resolution | Inference FPS (est.) | Notes |
|---|---|---|
| 320 × 320 | **3–8 FPS** | ✅ Usable for SAR (target ≥ 3 FPS) |
| 416 × 416 | 1.5–4 FPS | ⚠ Borderline; thermal throttle risk |
| 640 × 640 | 0.5–1.5 FPS | ❌ Too slow for real-time SAR |

**Key factors affecting FPS:**
- Other modules running (expect lower end: 3–4 FPS at full system load)
- CPU temperature (throttling reduces clock, dropping to 1–2 FPS at 85°C)
- Swap activity (if RAM insufficient, inference stalls)
- ONNX Runtime thread count: set to `intra_op_num_threads=2` or `3` (leave 1 core free)

### ONNX Runtime Configuration for RPi 4

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 2       # Leave cores for camera/depth
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

session = rt.InferenceSession(
    "models/yolov8n.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)
```

### INT8 Quantization Impact

| Variant | Model Size | FPS Gain | Accuracy Loss |
|---|---|---|---|
| FP32 (default) | 6.3 MB | baseline | 0% |
| FP16 | 3.2 MB | +10–20% | < 1% |
| INT8 (dynamic quant) | 1.8 MB | +40–60% | 1–3% |

**Recommendation:** Start with FP32 for correctness validation; switch to INT8 after Phase 3 benchmarks.

---

## 5. Alternative Lightweight Models

| Model | Format | Input Size | Est. FPS (RPi 4) | mAP50 (COCO) | Notes |
|---|---|---|---|---|---|
| **YOLOv8n** | ONNX FP32 | 320×320 | 3–8 | 37.3% | ✅ Primary choice |
| YOLOv8n | ONNX INT8 | 320×320 | 5–12 | ~36% | ✅ Good trade-off |
| **YOLOv5n** | ONNX FP32 | 320×320 | 4–10 | 28.0% | ✅ Fallback if v8 too slow |
| YOLOv5n | ONNX INT8 | 320×320 | 7–15 | ~27% | ✅ Fastest YOLO option |
| **MobileNetSSD v2** | ONNX | 300×300 | 8–15 | 22.1% | ✅ Lowest CPU; weaker accuracy |
| EfficientDet-D0 | ONNX | 512×512 | 2–5 | 34.6% | ⚠ Heavier than YOLOv8n |
| NanoDet-Plus | ONNX | 320×320 | 10–20 | 30.4% | ✅ Excellent for RPi, less known |

**Recommendation for SAR:**  
Primary: YOLOv8n ONNX INT8 at 320×320  
Fallback (if thermal throttle): MobileNetSSD v2 at 300×300  
Person-class only filtering reduces post-processing overhead significantly.

---

## 6. Known Incompatibilities & Workarounds

### 6.1 Bluetooth UART Conflict

**Problem:** RPi 4 uses `/dev/ttyAMA0` for Bluetooth by default. MAVLink flight controllers also need a UART.

**Workaround:**
```bash
# Disable Bluetooth UART in /boot/firmware/config.txt
dtoverlay=disable-bt
systemctl disable hciuart
# Reboot — /dev/ttyAMA0 is now free for MAVLink
```

### 6.2 USB 3.0 + Wi-Fi 2.4GHz Interference

**Problem:** USB 3.0 devices (especially RealSense) emit RF noise that interferes with 2.4 GHz Wi-Fi.

**Workaround:**
- Use **5 GHz Wi-Fi** for GCS communication
- Use a **shielded USB 3.0 cable** for D435i
- Keep USB cable away from the Wi-Fi antenna on the RPi board

### 6.3 libcamera / OpenCV Conflict

**Problem:** `import cv2` and `Picamera2` can conflict when OpenCV is built with V4L2 support and tries to claim the camera device.

**Workaround:**
- Use `opencv-python-headless` (no GUI, no V4L2 backend by default)
- Do not call `cv2.VideoCapture(0)` — use Picamera2 exclusively for Pi Camera
- If using OpenCV with RealSense frames, pass NumPy arrays directly (no capture object)

### 6.4 pyrealsense2 Import Error After Source Build

**Problem:** `ImportError: libssl.so.1.1: cannot open shared object file`

**Workaround:**
```bash
# librealsense requires libssl 1.1 (older); Bookworm ships 3.x
wget http://security.debian.org/debian-security/pool/updates/main/o/openssl/libssl1.1_1.1.1n-0+deb10u6_arm64.deb
sudo dpkg -i libssl1.1_1.1.1n-0+deb10u6_arm64.deb
```

### 6.5 RealSense USB Bandwidth Exhaustion

**Problem:** At full depth + RGB + IMU streams, the D435i can exceed USB 3.0 bandwidth, causing frame drops.

**Workaround:** AeroSentinel uses reduced resolutions:
- Depth: 640×480 @ 15 FPS (not 1280×720 @ 30 FPS)
- RGB: 848×480 @ 15 FPS (not 1920×1080 @ 30 FPS)
- IMU: 200 Hz (acceptable reduction from 400 Hz max)

### 6.6 picamera2 and Headless Boot

**Problem:** `Picamera2()` raises error if no display is available during initialization.

**Workaround:**
```python
os.environ["DISPLAY"] = ""  # or unset
# OR use in service with:
# Environment=DISPLAY=
```

### 6.7 onnxruntime Wheel Not Available for arm64

**Problem:** `pip install onnxruntime` fails on ARM64 (no wheel on PyPI for some versions).

**Workaround:**
```bash
# Use the Microsoft-provided ARM64 wheel:
pip install https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-1.16.3-cp311-cp311-linux_aarch64.whl
# Or use the community-provided piwheels build:
pip install onnxruntime --extra-index-url https://www.piwheels.org/simple
```

---

## 7. Verified Package Versions

> Tested on: Raspberry Pi 4B, 4 GB RAM, Raspberry Pi OS Bookworm 64-bit (kernel 6.6.x)

### System Packages (apt)

```
python3                   3.11.2
python3-pip               23.0.1
python3-picamera2         0.3.12
libcamera0                0.0.5-1
libcamera-dev             0.0.5-1
libudev-dev               252-xx
cmake                     3.25.1
build-essential           12.9
git                       2.39.2
libjpeg-dev               2.1.5-xx
libssl-dev                3.0.11
```

### Python Packages (pip / piwheels)

```
numpy                     1.24.4
opencv-python-headless    4.8.1.78
onnxruntime               1.16.3        # aarch64 wheel
pyrealsense2              2.54.1        # source build, see Section 8
fastapi                   0.104.1
uvicorn[standard]         0.24.0
websockets                12.0
pymavlink                 2.4.41
pyyaml                    6.0.1
python-dotenv             1.0.0
structlog                 23.2.0
sqlalchemy                2.0.23
aiofiles                  23.2.1
jinja2                    3.1.2
pillow                    10.1.0
```

> **Note:** These versions represent known-good combinations. Always pin exact versions in `requirements.txt` for reproducibility.

---

## 8. librealsense Source Build Procedure

> **Time required:** 60–90 minutes on RPi 4B (4 GB RAM)  
> **Disk space:** ~3 GB during build, ~500 MB installed

### Prerequisites

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    git cmake build-essential \
    libssl-dev libusb-1.0-0-dev \
    pkg-config libgtk-3-dev \
    libglfw3-dev libgl1-mesa-dev \
    libglu1-mesa-dev \
    python3-dev python3-pip \
    libudev-dev
```

### Step 1: Clone Repository

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.54.1  # pin to known-good version
```

### Step 2: Set Up udev Rules

```bash
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Step 3: Set Up Swap (Prevent OOM During Build)

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Step 4: Configure CMake Build

```bash
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DBUILD_WITH_CUDA=OFF \
    -DBUILD_WITH_OPENMP=OFF \
    -DHWM_OVER_XU=OFF \
    -DBUILD_UNIT_TESTS=OFF
```

### Step 5: Build (Use All 4 Cores)

```bash
make -j4 2>&1 | tee build.log
```

> Monitor temperature: `watch -n 5 vcgencmd measure_temp` — ensure active cooling.

### Step 6: Install

```bash
sudo make install
sudo ldconfig
```

### Step 7: Install Python Bindings

```bash
# Option A: wheel install (if generated in build/)
pip install wrappers/python/pyrealsense2-*.whl

# Option B: copy .so directly
cp wrappers/python/pyrealsense2*.so $(python3 -c "import site; print(site.getsitepackages()[0])")/
```

### Step 8: Verify Installation

```bash
python3 -c "
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print(f'Detected {len(devices)} RealSense device(s)')
for d in devices:
    print(f'  Serial: {d.get_info(rs.camera_info.serial_number)}')
    print(f'  FW: {d.get_info(rs.camera_info.firmware_version)}')
"
```

### Step 9: Restore Swap to Normal

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Common Build Failures

| Error | Cause | Fix |
|---|---|---|
| `c++: internal compiler error: Killed` | OOM during compilation | Increase swap; use `-j2` instead of `-j4` |
| `CMake Error: GLFW3 not found` | Missing dev package | `sudo apt install libglfw3-dev` |
| `cannot find -ludev` | Missing udev dev | `sudo apt install libudev-dev` |
| `pyrealsense2 ImportError: libssl.so.1.1` | OpenSSL version mismatch | Install libssl1.1 from Buster repo (see §6.4) |
| CMake version too old | Bookworm CMake sufficient | If still error: `pip install cmake` |

---

*Document maintained by: AeroSentinel Hardware Team*  
*Verify versions before each new RPi OS release.*
