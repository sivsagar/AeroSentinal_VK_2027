# AeroSentinel Troubleshooting Guide

## 1. RealSense D435i Not Detected
**Symptoms:** `rs-enumerate-devices` or python `pyrealsense2` fails to detect D435i.
- **Cause 1: Insufficient USB Power.** The D435i requires up to 700mA during active IR projection. Ensure the Raspberry Pi 4 is powered with a genuine 5V 3.0A+ supply.
- **Cause 2: Connected to USB 2.0.** The D435i must be plugged into one of the blue USB 3.0 ports for stereo depth streaming.
- **Cause 3: Missing udev rules.** Run:
  ```bash
  sudo curl -sSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules -o /etc/udev/rules.d/99-realsense-libusb.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

## 2. Pi Camera Module 3 "No cameras available"
**Symptoms:** `libcamera-hello` reports `ERROR: *** no cameras available ***`.
- **Cause 1: Loose CSI ribbon cable.** Check contacts at both the camera board and the Pi 4 CSI port.
- **Cause 2: Firmware overlay disabled.** Ensure `/boot/firmware/config.txt` contains:
  ```ini
  camera_auto_detect=1
  dtoverlay=imx708
  ```

## 3. High CPU Temperature (>75°C) & Thermal Throttling
**Symptoms:** Latency spikes, dropped frames.
- Ensure an active cooling fan is running over the Broadcom BCM2711 SoC.
- In `config/ai.yaml`, reduce `inference_fps_target` from 5 to 3.
- In `config/camera.yaml`, ensure resolution is 720p or lower (avoid 1080p).

## 4. MAVLink No Heartbeat Received
**Symptoms:** MAVLink status reports `disconnected`.
- Check UART pin connections: Pi TX (Pin 8) to FC RX, Pi RX (Pin 10) to FC TX.
- Ensure serial console is disabled on `/dev/ttyAMA0`:
  `sudo raspi-config` -> Interface Options -> Serial Port -> Login shell: No, Hardware serial: Yes.
- Confirm flight controller `SERIAL2_PROTOCOL = 2` (MAVLink2) and `SERIAL2_BAUD = 57` (57600 baud).

