"""
AeroSentinel Dashboard — FastAPI Application Factory (Phase 5)
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any, Dict, Optional

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
    logger.warning(
        "FastAPI/uvicorn not installed. "
        "Install with: pip install fastapi uvicorn[standard]\n"
        "Flask fallback: set DASHBOARD_BACKEND=flask and install flask flask-sock"
    )

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"


# ---------------------------------------------------------------------------
# Shared system-state container (mock-safe defaults)
# ---------------------------------------------------------------------------

class SystemState:
    """Shared mutable state container passed into the app at creation time."""

    def __init__(self):
        self.rgb_camera_connected: bool = False
        self.d435i_connected: bool = False
        self.ai_engine_running: bool = False
        self.depth_processing_running: bool = False
        self.mavlink_connected: bool = False
        self.mission_active: bool = False

        self.current_mission: Optional[Dict[str, Any]] = None
        self.missions: list = []
        self.recent_detections: list = []
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self.latest_imu: Optional[Dict[str, Any]] = None
        self.latest_obstacle: Optional[Dict[str, Any]] = None

        # live frame buffers (raw numpy arrays or None)
        self.rgb_frame = None
        self.depth_frame = None

    # ------------------------------------------------------------------
    def network_online(self) -> bool:
        try:
            socket.setdefaulttimeout(1)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("8.8.8.8", 53))
            s.close()
            return True
        except Exception:
            return False

    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "rgb_camera": "connected" if self.rgb_camera_connected else "disconnected",
            "d435i": "connected" if self.d435i_connected else "disconnected",
            "ai_engine": "running" if self.ai_engine_running else "stopped",
            "depth_processing": "running" if self.depth_processing_running else "stopped",
            "mavlink": "connected" if self.mavlink_connected else "disconnected",
            "mission": "active" if self.mission_active else "inactive",
            "network": "online" if self.network_online() else "offline",
        }


# ---------------------------------------------------------------------------
# Dashboard Application
# ---------------------------------------------------------------------------

class DashboardApp:
    """FastAPI-based ground-control dashboard."""

    def __init__(self):
        self.app: Optional[Any] = None
        self.state: Optional[SystemState] = None

    # ------------------------------------------------------------------
    def create_app(
        self,
        config: Optional[Dict[str, Any]] = None,
        system_state: Optional[SystemState] = None,
    ) -> Any:
        """
        Build and return the FastAPI application.

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
            description="Ground-control dashboard for autonomous UAV disaster S&R",
            version="1.0.0",
        )

        # ----------------------------------------------------------------
        # CORS — allow local network access (tablets, laptops on same Wi-Fi)
        # ----------------------------------------------------------------
        cors_origins = config.get("cors_origins", ["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ----------------------------------------------------------------
        # Static files & Jinja2 templates
        # ----------------------------------------------------------------
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

        # ----------------------------------------------------------------
        # Attach shared objects so routers can reach them
        # ----------------------------------------------------------------
        app.state.system_state = self.state
        app.state.templates = templates
        app.state.dashboard_config = config

        # ----------------------------------------------------------------
        # Routers & WebSocket
        # ----------------------------------------------------------------
        from aerosentinel.dashboard.api import build_router
        from aerosentinel.dashboard.websocket import WebSocketManager

        ws_manager = WebSocketManager(
            system_state=self.state,
            video_fps=config.get("ws_video_fps", 7),
            telemetry_hz=config.get("ws_telemetry_hz", 2),
            jpeg_quality=config.get("jpeg_quality", 65),
        )
        app.state.ws_manager = ws_manager

        api_router = build_router(self.state, templates, ws_manager)
        app.include_router(api_router)

        # ----------------------------------------------------------------
        # Lifecycle events
        # ----------------------------------------------------------------
        @app.on_event("startup")
        async def _startup():
            logger.info("AeroSentinel Dashboard starting up")
            await ws_manager.start()

        @app.on_event("shutdown")
        async def _shutdown():
            logger.info("AeroSentinel Dashboard shutting down")
            await ws_manager.stop()

        self.app = app
        return app

    # ------------------------------------------------------------------
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the dashboard using uvicorn."""
        if self.app is None:
            self.create_app()
        uvicorn.run(self.app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Convenience entry-point
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
