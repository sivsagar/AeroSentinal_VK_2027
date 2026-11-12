# Camera Calibration Guide (RGB & Depth Alignment)

## 1. Why Calibration is Required
For accurate RGB-D fusion, 2D bounding boxes detected on the Pi Camera Module 3 RGB image must be accurately mapped into the 3D coordinate space of the Intel RealSense D435i.

## 2. Intrinsic Calibration
Use an 8x6 checkerboard pattern (30mm square size):
```bash
# Capture checkerboard images
python3 -c "
import cv2, numpy as np
# Capture 30 images from different angles
"
```
The intrinsic matrix `K` and distortion coefficients `D` are saved to:
`calibration/rgb_camera/intrinsics.yaml`

## 3. Extrinsic Calibration (RGB to D435i)
Stereo extrinsic transformation provides rotation matrix `R` and translation vector `T`:
$$P_{depth} = R \cdot P_{rgb} + T$$

Configuration stored in:
`calibration/rgbd_extrinsics/extrinsics.yaml`
```yaml
rotation_matrix:
  - [1.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0]
  - [0.0, 0.0, 1.0]
translation_vector:
  - 0.045 # 45mm baseline horizontal offset
  - 0.000
  - 0.010 # 10mm vertical offset
```
When `use_calibration_file: false` in `config/fusion.yaml`, coordinate normalization using shared FOV scaling is used.

