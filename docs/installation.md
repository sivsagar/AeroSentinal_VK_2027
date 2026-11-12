# AeroSentinel Installation Guide for Raspberry Pi 4B

## 1. Operating System
- **Recommended:** Raspberry Pi OS 64-bit (Debian 12 Bookworm)
- **Minimum MicroSD Card:** 32GB Class 10 / A2 High Endurance

## 2. Hardware Preparation
1. Connect the **Raspberry Pi Camera Module 3** to the 15-pin CSI camera port (label facing rear).
2. Connect the **Intel RealSense D435i** to one of the blue **USB 3.0** ports.
3. Ensure adequate power: minimum 5V 3.0A dedicated power supply or 5V 3A UBEC from drone power distribution board.

## 3. System Configuration
Enable camera interfaces in `/boot/firmware/config.txt`:
```ini
camera_auto_detect=1
dtoverlay=imx708
```

## 4. Software Installation
```bash
# Update base system packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git cmake build-essential \
    libopencv-dev python3-opencv v4l-utils libusb-1.0-0-dev

# Clone repository
git clone https://github.com/AeroSentinel/AeroSentinel_VK_2027.git aerosentinel
cd aerosentinel

# Set up virtual environment
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install Python requirements
pip install -r requirements.txt

# Configure RealSense udev rules
sudo bash scripts/setup_realsense.sh
```

## 5. Model Weights
Place the ONNX model weights in `models/yolov8n.onnx`:
```bash
# Export or download YOLOv8n ONNX model
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx', imgsz=320)"
mv yolov8n.onnx models/
```

