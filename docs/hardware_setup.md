# AeroSentinel Hardware Setup & Wiring Guide

## 1. Physical Component Layout
- **Raspberry Pi 4 Model B (4GB):** Companion computer mounted on anti-vibration silicon dampers.
- **Pi Camera Module 3:** Mounted on forward-facing vibration-isolated gimbal or fixed angled mount (15° downward tilt recommended for aerial search & rescue).
- **Intel RealSense D435i:** Mounted rigidly aligned with Pi Camera 3 facing forward along UAV centerline.
- **Heatsink & Active Fan:** Aluminum armor case or active 30mm fan powered from Pi 5V GPIO pin 4 and GND pin 6.

## 2. Electrical Wiring & Interfaces
```
Raspberry Pi 4B Header:
Pin 04 [5V] ----------> Cooling Fan (+)
Pin 06 [GND] ---------> Cooling Fan (-)
Pin 08 [GPIO14/TX] ---> Flight Controller TELEM2 RX (Passive / Read-Only)
Pin 10 [GPIO15/RX] <--- Flight Controller TELEM2 TX (Telemetry Stream)
Pin 14 [GND] ---------> Flight Controller GND

CSI Ribbon Cable -----> Raspberry Pi Camera Module 3 (IMX708)
USB 3.0 Blue Port ----> Intel RealSense D435i (USB-C to USB-A 3.0 cable)
```

## 3. Power Distribution
- Do NOT power the companion computer from the flight controller's internal 5V rail.
- Use a dedicated 5V 3.5A Step-down BEC connected directly to the UAV main LiPo battery.
- The RealSense D435i draws up to 700mA during active IR emitter projection. USB 3.0 port provides up to 900mA.

