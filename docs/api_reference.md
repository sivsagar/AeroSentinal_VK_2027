# AeroSentinel API & Schema Reference

## REST API Endpoints (FastAPI Backend, Port 8080)

### 1. System Status
- **`GET /api/status`**
  Returns connectivity status across all subsystems.
  ```json
  {
    "rgb_camera": "connected",
    "d435i": "connected",
    "ai_engine": "running",
    "depth_processing": "running",
    "mavlink": "connected",
    "mission": "active",
    "network": "offline"
  }
  ```

### 2. Mission Lifecycle
- **`POST /api/mission/start`**
  Body: `{"name": "Sector A Recon", "operator": "Rescue Lead"}`
- **`POST /api/mission/pause`**
- **`POST /api/mission/resume`**
- **`POST /api/mission/stop`**
- **`GET /api/mission/current`**
- **`GET /api/mission/list`**

### 3. Detections & Hazard Ingestion
- **`GET /api/detections/recent?limit=20`**
  ```json
  {
    "detections": [
      {
        "id": 1,
        "timestamp": 1731427200.5,
        "class_name": "person",
        "confidence": 0.885,
        "distance_m": 4.12,
        "hazard_category": "PERSON_DETECTED",
        "verification_status": "PENDING_HUMAN_REVIEW",
        "lat": null,
        "lon": null
      }
    ],
    "count": 1
  }
  ```

### 4. Telemetry & Sensors
- **`GET /api/telemetry`**: Latest MAVLink battery, altitude, flight mode, and fix status.
- **`GET /api/imu`**: Latest low-pass filtered accelerometer and gyroscope data.
- **`GET /api/obstacle`**: Region-of-interest obstacle distances and warning levels.
- **`GET /api/performance`**: CPU%, RAM%, temperature, FPS, and dropped frames.

## WebSocket Real-Time Stream (`/ws`)
Broadcasts messages with a `type` tag:
- `"rgb_frame"`: Base64 JPEG encoded RGB video frame.
- `"depth_frame"`: Base64 JPEG encoded colorized depth colormap.
- `"telemetry"`: JSON vehicle telemetry.
- `"obstacle"`: JSON obstacle sector analysis.
- `"system_status"`: Subsystem health states.

