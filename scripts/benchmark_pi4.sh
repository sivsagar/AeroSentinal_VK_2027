#!/usr/bin/env bash
# ==============================================================================
# AeroSentinel Raspberry Pi 4B Benchmarking Suite
# ==============================================================================
# Measures:
# 1. CPU synthetic matrix math latency
# 2. OpenCV frame processing throughput (FPS)
# 3. ONNX Runtime inference latency (CPU provider)
# 4. CPU temperature and RAM utilization during load
# ==============================================================================

set -eo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_FILE="data/logs/benchmark_${TIMESTAMP}.txt"
mkdir -p data/logs

echo "==========================================================" | tee "$OUT_FILE"
echo "  AeroSentinel Raspberry Pi 4B Performance Benchmark       " | tee -a "$OUT_FILE"
echo "  Timestamp: $(date)                                      " | tee -a "$OUT_FILE"
echo "==========================================================" | tee -a "$OUT_FILE"

# System info
echo -e "\n--- System Hardware & OS Details ---" | tee -a "$OUT_FILE"
uname -a | tee -a "$OUT_FILE"
if [ -f /proc/device-tree/model ]; then
    echo "Hardware Model: $(cat /proc/device-tree/model)" | tee -a "$OUT_FILE"
fi
free -h | tee -a "$OUT_FILE"

# CPU Temperature Check function
read_temp() {
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        raw=$(cat /sys/class/thermal/thermal_zone0/temp)
        echo "scale=1; $raw / 1000" | bc
    elif command -v vcgencmd &>/dev/null; then
        vcgencmd measure_temp | sed "s/temp=//;s/'C//"
    else
        echo "N/A"
    fi
}

echo -e "\nInitial CPU Temp: $(read_temp) °C" | tee -a "$OUT_FILE"

# Python Benchmark
python3 - << 'EOF' | tee -a "$OUT_FILE"
import time
import numpy as np

print("\n--- 1. NumPy Matrix Math Benchmark (Cortex-A72 NEON) ---")
N = 1000
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

t0 = time.perf_counter()
for _ in range(5):
    C = np.dot(A, B)
t1 = time.perf_counter()
avg_time = (t1 - t0) / 5.0
gflops = (2 * (N ** 3)) / (avg_time * 1e9)
print(f"Matrix 1000x1000 dot product: {avg_time*1000:.2f} ms ({gflops:.2f} GFLOPS)")

print("\n--- 2. OpenCV Image Resizing & Colormap Throughput ---")
try:
    import cv2
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    frames = 100
    t0 = time.perf_counter()
    for _ in range(frames):
        small = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    t1 = time.perf_counter()
    fps = frames / (t1 - t0)
    print(f"OpenCV 720p->320p Pipeline: {fps:.1f} FPS ({(t1 - t0)*1000/frames:.2f} ms/frame)")
except Exception as e:
    print(f"OpenCV benchmark skipped: {e}")

print("\n--- 3. ONNX Runtime Inference Benchmark ---")
try:
    from aerosentinel.perception.inference_engine import InferenceEngineFactory
    from aerosentinel.utils.config_loader import ConfigLoader
    
    cfg = ConfigLoader().load("config/ai.yaml")
    engine = InferenceEngineFactory.create(cfg.get("detection", {}))
    engine.load()
    engine.warmup()
    
    dummy = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = engine.infer(dummy)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    
    avg_lat = np.mean(latencies)
    print(f"Model: {engine.__class__.__name__}")
    print(f"Mean Inference Latency: {avg_lat:.2f} ms (~{1000/avg_lat:.1f} FPS)")
except Exception as e:
    print(f"ONNX benchmark skipped: {e}")
EOF

echo -e "\nFinal CPU Temp: $(read_temp) °C" | tee -a "$OUT_FILE"
echo "Benchmark completed. Results saved to $OUT_FILE"

