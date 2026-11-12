"""
AeroSentinel Dashboard — REST API Endpoints
===========================================
Implements REST endpoints for ground control:
- GET / -> render index.html
- GET /api/status -> status JSON of all subsystems
- GET /api/mission/current -> active mission details
- GET /api/mission/list -> all missions
- POST /api/mission/start -> start mission
- POST /api/mission/pause -> pause mission
- POST /api/mission/resume -> resume mission
- POST /api/mission/stop -> stop mission
- GET /api/detections/recent -> latest detections
- GET /api/telemetry -> latest telemetry
- GET /api/imu -> latest IMU readings
- GET /api/performance -> system resource metrics
- GET /api/obstacle -> latest obstacle report
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MissionStartRequest(BaseModel):
    name: str = "Search & Rescue Mission"
    operator: str = "Ground Operator"
    notes: Optional[str] = None


def build_router(system_state: Any, templates: Any, ws_manager: Any) -> APIRouter:
    """Factory creating APIRouter tied to active system state and websocket manager."""
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        try:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "title": "AeroSentinel Mission Control",
                    "status": system_state.to_status_dict(),
                },
            )
        except TypeError:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "title": "AeroSentinel Mission Control",
                    "status": system_state.to_status_dict(),
                },
            )

    @router.get("/api/status")
    async def get_status():
        return JSONResponse(system_state.to_status_dict())

    @router.get("/api/mission/current")
    async def get_current_mission():
        return JSONResponse(
            {
                "active": system_state.mission_active,
                "mission": system_state.current_mission,
            }
        )

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
        return JSONResponse({"status": "started", "mission": system_state.current_mission})

    @router.post("/api/mission/pause")
    async def pause_mission():
        if not system_state.mission_active or not system_state.current_mission:
            raise HTTPException(status_code=400, detail="No active mission to pause")
        system_state.current_mission["status"] = "PAUSED"
        return JSONResponse({"status": "paused", "mission": system_state.current_mission})

    @router.post("/api/mission/resume")
    async def resume_mission():
        if not system_state.current_mission:
            raise HTTPException(status_code=400, detail="No mission to resume")
        system_state.mission_active = True
        system_state.current_mission["status"] = "ACTIVE"
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
        return JSONResponse({"status": "stopped", "mission": curr})

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
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
        except Exception:
            cpu = 15.0
            ram = 30.0

        return JSONResponse(
            {
                "cpu_percent": cpu,
                "ram_percent": ram,
                "fps": 15.0,
                "dropped_frames": 0,
                "temperature_c": None,
                "timestamp": time.time(),
            }
        )

    @router.get("/api/obstacle")
    async def get_obstacle():
        obs = system_state.latest_obstacle or {
            "overall_status": "SAFE",
            "warning_active": False,
            "closest_obstacle_m": None,
            "left": {"status": "SAFE", "min_distance_m": None},
            "center": {"status": "SAFE", "min_distance_m": None},
            "right": {"status": "SAFE", "min_distance_m": None},
        }
        return JSONResponse(obs)

    # Register WebSocket endpoint
    router.add_api_websocket_route("/ws", ws_manager.websocket_endpoint)

    return router
