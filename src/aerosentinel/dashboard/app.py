"""
AeroSentinel Dashboard — FastAPI Application Factory (Phase 5 Unified Perception)
==================================================================================
Ground-control robotics monitoring interface supporting RealSense RGB, RealSense
Depth, Raspberry Pi Camera 3 Wide, YOLOv8n inference, multi-person tracking,
real-time obstacle proximity assessment, and system health metrics.
"""

from __future__ import annotations

import collections
import datetime
import logging
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try FastAPI; fall back gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn[standard]")

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"


# ---------------------------------------------------------------------------
# Dashboard Event Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DashboardEvent:
    """Chronological system event for tracking, proximity, perception, and hardware."""

    id: str
    timestamp: float
    timestamp_iso: str
    event_type: str
    severity: str  # "INFO", "CAUTION", "WARNING", "CRITICAL", "ERROR"
    message: str
    count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "count": self.count,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Shared System State Container
# ---------------------------------------------------------------------------

class SystemState:
    """Shared mutable state container passed into FastAPI and the acquisition pipeline."""

    def __init__(self, db_manager: Optional[Any] = None) -> None:
        self.start_time: float = time.time()
        self.db_manager = db_manager

        # Subsystem connectivity & health flags
        self.rgb_camera_connected: bool = False      # RealSense RGB
        self.d435i_connected: bool = False           # RealSense Depth
        self.picam_connected: bool = False           # Pi Camera Module 3
        self.ai_engine_running: bool = False         # YOLOv8n ONNX Engine
        self.depth_processing_running: bool = False  # Depth filter chain
        self.tracking_running: bool = False          # PersonTracker
        self.obstacle_running: bool = False          # ObstacleDetector
        self.mavlink_connected: bool = False         # Read-only MAVLink
        self.mission_active: bool = False            # S&R Mission lifecycle

        # Live telemetry and metrics
        self.camera_fps: float = 0.0
        self.inference_fps: float = 0.0
        self.dropped_frames: int = 0
        self.cpu_percent: Optional[float] = None
        self.cpu_temp_c: Optional[float] = None
        self.ram_percent: Optional[float] = None
        self.ram_used_mb: Optional[float] = None
        self.ram_total_mb: Optional[float] = None
        self.metrics_timestamp: float = 0.0

        # Mission and perception payloads
        self.current_mission: Optional[Dict[str, Any]] = None
        self.missions: list = []
        self.recent_detections: list = []
        self.active_tracks: List[Dict[str, Any]] = []
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self.latest_imu: Optional[Dict[str, Any]] = None
        self.latest_obstacle: Optional[Dict[str, Any]] = None

        # Live frame buffers (numpy arrays or None)
        self.rgb_frame = None        # RealSense RGB (annotated)
        self.depth_frame = None      # RealSense Depth colormap
        self.picam_frame = None      # Pi Camera 3 Wide RGB

        # Chronological event history queue (thread-safe, bounded, deduplicated)
        self._events_lock = threading.Lock()
        self._events: Deque[DashboardEvent] = collections.deque(maxlen=200)
        self._last_event_signature: Optional[Tuple[str, str]] = None
        self._last_event_time: float = 0.0

    # ------------------------------------------------------------------
    # Network and System Status
    # ------------------------------------------------------------------

    def network_online(self) -> bool:
        """Check if local network socket connection is operational."""
        try:
            socket.setdefaulttimeout(0.5)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("8.8.8.8", 53))
            s.close()
            return True
        except Exception:
            return False

    def is_metrics_stale(self, threshold_s: float = 3.0) -> bool:
        """Return True if system metrics have not been refreshed within threshold."""
        if self.metrics_timestamp <= 0:
            return True
        return (time.time() - self.metrics_timestamp) > threshold_s

    def to_status_dict(self) -> Dict[str, Any]:
        """Convert system status to dictionary for API and WebSockets."""
        stale = self.is_metrics_stale()
        return {
            # Legacy compatibility keys
            "rgb_camera": "connected" if (self.rgb_camera_connected or self.picam_connected) else "disconnected",
            "d435i": "connected" if self.d435i_connected else "disconnected",
            "ai_engine": "running" if self.ai_engine_running else "stopped",
            "depth_processing": "running" if self.depth_processing_running else "stopped",
            "mavlink": "connected" if self.mavlink_connected else "disconnected",
            "mission": "active" if self.mission_active else "inactive",
            "network": "online" if self.network_online() else "offline",

            # Enhanced robotics health keys
            "picam_camera": "connected" if self.picam_connected else "disconnected",
            "realsense_rgb": "connected" if self.rgb_camera_connected else "disconnected",
            "realsense_depth": "connected" if self.d435i_connected else "disconnected",
            "person_tracker": "running" if self.tracking_running else "stopped",
            "obstacle_detector": "running" if self.obstacle_running else "stopped",
            "uptime_s": round(time.time() - self.start_time, 1),
            "is_stale": stale,
            "camera_fps": round(self.camera_fps, 1),
            "inference_fps": round(self.inference_fps, 1),
            "dropped_frames": self.dropped_frames,
            "cpu_percent": self.cpu_percent,
            "cpu_temp_c": self.cpu_temp_c,
            "ram_percent": self.ram_percent,
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "active_track_count": len(self.active_tracks),
        }

    def to_health_dict(self) -> Dict[str, Any]:
        """Detailed resource and performance health metrics dictionary."""
        stale = self.is_metrics_stale()
        return {
            "uptime_s": round(time.time() - self.start_time, 1),
            "is_stale": stale,
            "camera_fps": round(self.camera_fps, 1),
            "inference_fps": round(self.inference_fps, 1),
            "dropped_frames": self.dropped_frames,
            "cpu_percent": self.cpu_percent,
            "cpu_temp_c": self.cpu_temp_c,
            "ram_percent": self.ram_percent,
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "picam_connected": self.picam_connected,
            "realsense_rgb_connected": self.rgb_camera_connected,
            "realsense_depth_connected": self.d435i_connected,
            "ai_engine_running": self.ai_engine_running,
            "tracking_running": self.tracking_running,
            "obstacle_running": self.obstacle_running,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Event Logging and Deduplication
    # ------------------------------------------------------------------

    def add_event(
        self,
        event_type: str,
        message: str,
        severity: str = "INFO",
        mission_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DashboardEvent:
        """
        Record a chronological event with automatic deduplication for repeated warnings.

        Consecutive events sharing identical (event_type, message) within 3.0s
        increment the existing event count rather than cluttering the timeline.
        """
        now = time.time()
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        sig = (event_type, message)
        meta = metadata or {}

        with self._events_lock:
            # Check for deduplication with last event
            if (
                self._events
                and sig == self._last_event_signature
                and (now - self._last_event_time) < 3.0
            ):
                last_evt = self._events[0]  # newest is at index 0
                last_evt.count += 1
                last_evt.timestamp = now
                last_evt.timestamp_iso = now_iso
                self._last_event_time = now
                return last_evt

            # Create new distinct event
            evt = DashboardEvent(
                id=str(uuid.uuid4())[:8],
                timestamp=now,
                timestamp_iso=now_iso,
                event_type=event_type,
                severity=severity.upper(),
                message=message,
                count=1,
                metadata=meta,
            )
            self._events.appendleft(evt)
            self._last_event_signature = sig
            self._last_event_time = now

        # Optionally persist event in SQLite database
        if self.db_manager is not None:
            try:
                from aerosentinel.storage.models import Event
                import json
                db_evt = Event(
                    timestamp=now,
                    event_type=event_type,
                    severity=severity.upper(),
                    message=message,
                    mission_id=mission_id,
                    metadata_json=json.dumps(meta) if meta else None,
                )
                self.db_manager.save_event(db_evt)
            except Exception as exc:
                logger.debug("Failed to persist event to DB: %s", exc)

        return evt

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return list of recent events in reverse-chronological order."""
        with self._events_lock:
            evts = list(self._events)[:limit]
        return [e.to_dict() for e in evts]


# ---------------------------------------------------------------------------
# Dashboard Application
# ---------------------------------------------------------------------------

class DashboardApp:
    """FastAPI-based ground-control robotics dashboard."""

    def __init__(self) -> None:
        self.app: Optional[Any] = None
        self.state: Optional[SystemState] = None

    def create_app(
        self,
        config: Optional[Dict[str, Any]] = None,
        system_state: Optional[SystemState] = None,
    ) -> Any:
        """
        Build and return the FastAPI application instance.

        Parameters
        ----------
        config: optional dict with keys: ws_video_fps, ws_telemetry_hz,
                jpeg_quality, cors_origins
        system_state: SystemState instance; a default mock one is created
                      if not provided.
        """
        if not FASTAPI_AVAILABLE:
            raise RuntimeError(
                "FastAPI is not installed. Run: pip install fastapi uvicorn[standard]"
            )

        config = config or {}
        self.state = system_state or SystemState()

        app = FastAPI(
            title="AeroSentinel Mission Control",
            description="Unified Ground-Control Dashboard for Autonomous UAV S&R",
            version="1.2.0",
        )

        # CORS — allow local network access (tablets, laptops on same Wi-Fi)
        cors_origins = config.get("cors_origins", ["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Static files & Jinja2 templates
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

        # Attach shared objects to app.state
        app.state.system_state = self.state
        app.state.templates = templates
        app.state.dashboard_config = config

        # Initialize WebSocket Manager
        from aerosentinel.dashboard.api import build_router
        from aerosentinel.dashboard.websocket import WebSocketManager

        ws_manager = WebSocketManager(
            system_state=self.state,
            video_fps=config.get("ws_video_fps", 7),
            telemetry_hz=config.get("ws_telemetry_hz", 2),
            jpeg_quality=config.get("jpeg_quality", 65),
        )
        app.state.ws_manager = ws_manager

        # Attach API router
        api_router = build_router(self.state, templates, ws_manager)
        app.include_router(api_router)

        # Lifecycle events
        @app.on_event("startup")
        async def _startup():
            logger.info("AeroSentinel Dashboard starting up.")
            await ws_manager.start()

        @app.on_event("shutdown")
        async def _shutdown():
            logger.info("AeroSentinel Dashboard shutting down.")
            await ws_manager.stop()

        self.app = app
        return app

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run the dashboard server using uvicorn."""
        if self.app is None:
            self.create_app()
        uvicorn.run(self.app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Convenience Factory
# ---------------------------------------------------------------------------

def create_app(
    config: Optional[Dict[str, Any]] = None,
    system_state: Optional[SystemState] = None,
) -> Any:
    """Module-level factory used by tests and the CLI."""
    dash = DashboardApp()
    return dash.create_app(config=config, system_state=system_state)


if __name__ == "__main__":  # pragma: no cover
    dash = DashboardApp()
    dash.create_app()
    dash.run()
