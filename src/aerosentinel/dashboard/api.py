"""
AeroSentinel Dashboard — REST API Endpoints
===========================================
Implements REST endpoints for ground control:
- GET /                      -> render index.html
- GET /api/status            -> status JSON of all subsystems
- GET /api/health            -> comprehensive hardware and performance health
- GET /api/perception        -> perception pipeline and inference status
- GET /api/tracks            -> active multi-person tracking records
- GET /api/tracking          -> alias for /api/tracks
- GET /api/obstacle          -> latest obstacle proximity report
- GET /api/events            -> chronological deduplicated event history
- POST /api/events           -> record new system/perception event
- GET /api/video/rgb         -> latest JPEG frame from RealSense RGB
- GET /api/video/depth       -> latest JPEG frame from RealSense Depth
- GET /api/video/picam       -> latest JPEG frame from Pi Camera Module 3
- GET /api/mission/current   -> active mission details
- GET /api/mission/list      -> all missions
- POST /api/mission/start    -> start mission
- POST /api/mission/pause    -> pause mission
- POST /api/mission/resume   -> resume mission
- POST /api/mission/stop     -> stop mission
- GET /api/detections/recent -> latest detections
- GET /api/telemetry         -> latest telemetry
- GET /api/imu               -> latest IMU readings
- GET /api/performance       -> system resource metrics
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MissionStartRequest(BaseModel):
    name: str = "Search & Rescue Mission"
    operator: str = "Ground Operator"
    notes: Optional[str] = None


class EventCreateRequest(BaseModel):
    event_type: str
    message: str
    severity: str = "INFO"
    metadata: Optional[Dict[str, Any]] = None


def _frame_to_jpeg_bytes(frame: Optional[np.ndarray], placeholder_text: str = "NO SIGNAL") -> bytes:
    """Encode numpy frame to JPEG bytes or return a dark placeholder image."""
    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
        success, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if success:
            return buf.tobytes()

    # Generate lightweight dark placeholder image
    placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        placeholder_text,
        (40, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (120, 120, 120),
        2,
        cv2.LINE_AA,
    )
    _, buf = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return buf.tobytes()


def build_router(system_state: Any, templates: Any, ws_manager: Any) -> APIRouter:
    """Factory creating APIRouter tied to active system state and websocket manager."""
    router = APIRouter()

    # ------------------------------------------------------------------
    # Frontend Page
    # ------------------------------------------------------------------

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        try:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "title": "AeroSentinel Unified Rescue Ground Control",
                    "status": system_state.to_status_dict(),
                },
            )
        except TypeError:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "title": "AeroSentinel Unified Rescue Ground Control",
                    "status": system_state.to_status_dict(),
                },
            )

    # ------------------------------------------------------------------
    # System Status & Health Endpoints
    # ------------------------------------------------------------------

    @router.get("/api/status")
    async def get_status():
        return JSONResponse(system_state.to_status_dict())

    @router.get("/api/health")
    async def get_health():
        if hasattr(system_state, "to_health_dict"):
            return JSONResponse(system_state.to_health_dict())
        return JSONResponse(system_state.to_status_dict())

    @router.get("/api/perception")
    async def get_perception():
        return JSONResponse({
            "ai_engine_running": getattr(system_state, "ai_engine_running", False),
            "tracking_running": getattr(system_state, "tracking_running", False),
            "obstacle_running": getattr(system_state, "obstacle_running", False),
            "active_track_count": len(getattr(system_state, "active_tracks", [])),
            "inference_fps": getattr(system_state, "inference_fps", 0.0),
            "camera_fps": getattr(system_state, "camera_fps", 0.0),
            "dropped_frames": getattr(system_state, "dropped_frames", 0),
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # Multi-Person Tracking Endpoints
    # ------------------------------------------------------------------

    @router.get("/api/tracks")
    async def get_tracks():
        tracks = getattr(system_state, "active_tracks", [])
        return JSONResponse({
            "track_count": len(tracks),
            "tracks": tracks,
            "timestamp": time.time(),
        })

    @router.get("/api/tracking")
    async def get_tracking_alias():
        """Alias for /api/tracks."""
        return await get_tracks()

    # ------------------------------------------------------------------
    # Obstacle Proximity Endpoint
    # ------------------------------------------------------------------

    @router.get("/api/obstacle")
    async def get_obstacle():
        obs = getattr(system_state, "latest_obstacle", None)
        if not obs:
            obs = {
                "overall_status": "CLEAR",
                "warning_active": False,
                "caution_active": False,
                "critical_active": False,
                "closest_obstacle_m": None,
                "grid_layout": "1x3",
                "left": {"status": "CLEAR", "min_distance_m": None},
                "center": {"status": "CLEAR", "min_distance_m": None},
                "right": {"status": "CLEAR", "min_distance_m": None},
                "regions": {},
                "safety_notice": "PROVISIONAL BENCH VALUES — ADVISORY ONLY",
            }
        return JSONResponse(obs)

    # ------------------------------------------------------------------
    # Chronological Event History Endpoints
    # ------------------------------------------------------------------

    @router.get("/api/events")
    async def get_events(limit: int = Query(default=50, ge=1, le=200)):
        if hasattr(system_state, "get_recent_events"):
            evts = system_state.get_recent_events(limit=limit)
        else:
            evts = []
        return JSONResponse({
            "event_count": len(evts),
            "events": evts,
            "timestamp": time.time(),
        })

    @router.post("/api/events")
    async def create_event(body: EventCreateRequest):
        if hasattr(system_state, "add_event"):
            evt = system_state.add_event(
                event_type=body.event_type,
                message=body.message,
                severity=body.severity,
                metadata=body.metadata,
            )
            return JSONResponse({"status": "created", "event": evt.to_dict()})
        return JSONResponse({"status": "acknowledged", "message": body.message})

    # ------------------------------------------------------------------
    # Live Video Frame Endpoints (JPEG image/jpeg)
    # ------------------------------------------------------------------

    @router.get("/api/video/rgb")
    async def get_video_rgb():
        frame = getattr(system_state, "rgb_frame", None)
        jpeg_bytes = _frame_to_jpeg_bytes(frame, placeholder_text="REALSENSE RGB OFFLINE")
        return Response(content=jpeg_bytes, media_type="image/jpeg")

    @router.get("/api/video/depth")
    async def get_video_depth():
        frame = getattr(system_state, "depth_frame", None)
        jpeg_bytes = _frame_to_jpeg_bytes(frame, placeholder_text="DEPTH MAP OFFLINE")
        return Response(content=jpeg_bytes, media_type="image/jpeg")

    @router.get("/api/video/picam")
    async def get_video_picam():
        frame = getattr(system_state, "picam_frame", None)
        jpeg_bytes = _frame_to_jpeg_bytes(frame, placeholder_text="PI CAMERA 3 OFFLINE")
        return Response(content=jpeg_bytes, media_type="image/jpeg")

    # ------------------------------------------------------------------
    # Mission Lifecycle Endpoints
    # ------------------------------------------------------------------

    @router.get("/api/mission/current")
    async def get_current_mission():
        return JSONResponse({
            "active": system_state.mission_active,
            "mission": system_state.current_mission,
        })

    @router.get("/api/mission/list")
    async def list_missions():
        return JSONResponse({"missions": system_state.missions})

    @router.post("/api/mission/start")
    async def start_mission(body: MissionStartRequest):
        system_state.mission_active = True
        system_state.current_mission = {
            "id": int(time.time()),
            "name": body.name,
            "operator": body.operator,
            "notes": body.notes,
            "start_time": time.time(),
            "status": "ACTIVE",
            "detection_count": 0,
        }
        system_state.missions.append(system_state.current_mission)
        logger.info("Mission started via API: %s", body.name)
        if hasattr(system_state, "add_event"):
            system_state.add_event("MISSION_STARTED", f"Mission '{body.name}' started", "INFO")
        return JSONResponse({"status": "started", "mission": system_state.current_mission})

    @router.post("/api/mission/pause")
    async def pause_mission():
        if not system_state.mission_active or not system_state.current_mission:
            raise HTTPException(status_code=400, detail="No active mission to pause")
        system_state.current_mission["status"] = "PAUSED"
        if hasattr(system_state, "add_event"):
            system_state.add_event("MISSION_PAUSED", "Mission execution paused", "INFO")
        return JSONResponse({"status": "paused", "mission": system_state.current_mission})

    @router.post("/api/mission/resume")
    async def resume_mission():
        if not system_state.current_mission:
            raise HTTPException(status_code=400, detail="No mission to resume")
        system_state.mission_active = True
        system_state.current_mission["status"] = "ACTIVE"
        if hasattr(system_state, "add_event"):
            system_state.add_event("MISSION_RESUMED", "Mission execution resumed", "INFO")
        return JSONResponse({"status": "resumed", "mission": system_state.current_mission})

    @router.post("/api/mission/stop")
    async def stop_mission():
        if not system_state.current_mission:
            raise HTTPException(status_code=400, detail="No mission to stop")
        system_state.mission_active = False
        system_state.current_mission["status"] = "COMPLETED"
        system_state.current_mission["end_time"] = time.time()
        curr = system_state.current_mission
        system_state.current_mission = None
        if hasattr(system_state, "add_event"):
            system_state.add_event("MISSION_STOPPED", "Mission completed and stopped", "INFO")
        return JSONResponse({"status": "stopped", "mission": curr})

    # ------------------------------------------------------------------
    # Telemetry and Performance Endpoints
    # ------------------------------------------------------------------

    @router.get("/api/detections/recent")
    async def get_recent_detections(limit: int = 20):
        dets = system_state.recent_detections[-limit:]
        return JSONResponse({"detections": dets, "count": len(dets)})

    @router.get("/api/telemetry")
    async def get_telemetry():
        tel = system_state.latest_telemetry or {
            "timestamp": time.time(),
            "battery_pct": 100.0,
            "alt_m": 0.0,
            "flight_mode": "MANUAL",
            "armed": False,
            "is_connected": False,
            "lat": None,
            "lon": None,
        }
        return JSONResponse(tel)

    @router.get("/api/imu")
    async def get_imu():
        imu = system_state.latest_imu or {
            "timestamp": time.time(),
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 9.81,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "roll_estimate": 0.0,
            "pitch_estimate": 0.0,
            "is_valid": False,
        }
        return JSONResponse(imu)

    @router.get("/api/performance")
    async def get_performance():
        if hasattr(system_state, "to_health_dict"):
            h = system_state.to_health_dict()
            return JSONResponse({
                "cpu_percent": h.get("cpu_percent") or 15.0,
                "ram_percent": h.get("ram_percent") or 30.0,
                "fps": h.get("camera_fps") or 15.0,
                "dropped_frames": h.get("dropped_frames", 0),
                "temperature_c": h.get("cpu_temp_c"),
                "timestamp": time.time(),
            })

        return JSONResponse({
            "cpu_percent": 15.0,
            "ram_percent": 30.0,
            "fps": 15.0,
            "dropped_frames": 0,
            "temperature_c": None,
            "timestamp": time.time(),
        })

    # Register WebSocket endpoint
    router.add_api_websocket_route("/ws", ws_manager.websocket_endpoint)

    return router
