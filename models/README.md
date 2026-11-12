# AeroSentinel Models Directory

This directory stores AI model files for AeroSentinel's survivor detection pipeline.

## Required Models

### YOLOv8n ONNX (Primary)

**File:** `yolov8n.onnx`  
**Input:** 320×320 RGB  
**Output:** Bounding boxes, class IDs, confidence scores  
**Expected inference speed on RPi 4:** 3–8 FPS (FP32), 5–12 FPS (INT8)

**Download:**
```bash
# Option 1: Using ultralytics Python package
pip install ultralytics
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=320, simplify=True)
"
# Move to models/ directory
mv yolov8n.onnx models/

# Option 2: Direct download from Ultralytics releases
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
# Then export as above
```

**Validation:**
```bash
python3 -c "
import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession('models/yolov8n.onnx', providers=['CPUExecutionProvider'])
dummy = np.zeros((1, 3, 320, 320), dtype=np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: dummy})
print('Model output shape:', out[0].shape)
print('Model loaded successfully.')
"
```

---

### YOLOv8n INT8 ONNX (Optional — faster on RPi 4)

**File:** `yolov8n_int8.onnx`

**Generate from FP32 model:**
```bash
python3 -c "
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('models/yolov8n.onnx', 'models/yolov8n_int8.onnx', weight_type=QuantType.QInt8)
print('INT8 model created.')
"
```

---

### Alternative: YOLOv5n ONNX (Fallback)

**File:** `yolov5n.onnx`

```bash
git clone https://github.com/ultralytics/yolov5.git /tmp/yolov5
pip install -r /tmp/yolov5/requirements.txt
python3 /tmp/yolov5/export.py --weights yolov5n.pt --include onnx --imgsz 320
mv yolov5n.onnx models/
```

---

## Model Registry

| File | Type | Input Size | Est. FPS (RPi 4) | Status |
|---|---|---|---|---|
| `yolov8n.onnx` | YOLOv8n FP32 | 320×320 | 3–8 | Not downloaded |
| `yolov8n_int8.onnx` | YOLOv8n INT8 | 320×320 | 5–12 | Not downloaded |
| `yolov5n.onnx` | YOLOv5n FP32 | 320×320 | 4–10 | Not downloaded |

---

> **Note:** Model files (*.onnx, *.pt) are excluded from Git. Download or generate them before running AeroSentinel.
