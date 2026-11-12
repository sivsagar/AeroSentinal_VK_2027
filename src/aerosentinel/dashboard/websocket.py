"""
AeroSentinel Dashboard — WebSocket Streaming Manager
====================================================
Broadcasts live RGB video, depth visualizations, detection events,
obstacle warnings, telemetry and performance metrics to connected dashboard clients.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, List, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages active WebSocket connections and runs periodic broadcast loops."""

    def __init__(
        self,
        system_state: Any,
        video_fps: int = 7,
        telemetry_hz: int = 2,
        jpeg_quality: int = 65,
    ) -> None:
        self.system_state = system_state
        self.video_interval = 1.0 / max(1, video_fps)
        self.telemetry_interval = 1.0 / max(1, telemetry_hz)
        self.jpeg_quality = jpeg_quality

        self._active_connections: List[WebSocket] = []
        self._running = False
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._active_connections.append(websocket)
        logger.debug("Dashboard client connected. Active: %d", len(self._active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._active_connections:
            self._active_connections.remove(websocket)
            logger.debug("Dashboard client disconnected. Active: %d", len(self._active_connections))

    async def broadcast_json(self, message: dict) -> None:
        if not self._active_connections:
            return
        payload = json.dumps(message)
        dead: List[WebSocket] = []
        for ws in self._active_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def websocket_endpoint(self, websocket: WebSocket) -> None:
        await self.connect(websocket)
        # Send initial status
        try:
            await websocket.send_text(
                json.dumps({
                    "type": "system_status",
                    "data": self.system_state.to_status_dict(),
                    "timestamp": time.time(),
                })
            )
            while True:
                data = await websocket.receive_text()
                # Handle possible client commands
                logger.debug("Received WS msg: %s", data)
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            logger.debug("WS connection exception: %s", e)
            self.disconnect(websocket)

    async def start(self) -> None:
        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("WebSocketManager broadcast loop started.")

    async def stop(self) -> None:
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        for ws in list(self._active_connections):
            try:
                await ws.close()
            except Exception:
                pass
        self._active_connections.clear()
        logger.info("WebSocketManager stopped.")

    async def _broadcast_loop(self) -> None:
        last_video = 0.0
        last_telemetry = 0.0

        while self._running:
            now = time.time()

            if now - last_telemetry >= self.telemetry_interval:
                last_telemetry = now
                await self._send_telemetry_and_status()

            if now - last_video >= self.video_interval:
                last_video = now
                await self._send_frames()

            await asyncio.sleep(0.02)

    async def _send_telemetry_and_status(self) -> None:
        if not self._active_connections:
            return
        status = self.system_state.to_status_dict()
        tel = self.system_state.latest_telemetry
        obs = self.system_state.latest_obstacle

        await self.broadcast_json({"type": "system_status", "data": status, "timestamp": time.time()})
        if tel:
            await self.broadcast_json({"type": "telemetry", "data": tel, "timestamp": time.time()})
        if obs:
            await self.broadcast_json({"type": "obstacle", "data": obs, "timestamp": time.time()})

    async def _send_frames(self) -> None:
        if not self._active_connections or not CV2_AVAILABLE:
            return

        rgb_frame = getattr(self.system_state, "rgb_frame", None)
        if rgb_frame is not None:
            try:
                _, buf = cv2.imencode(".jpg", rgb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                b64_rgb = base64.b64encode(buf).decode("utf-8")
                await self.broadcast_json({
                    "type": "rgb_frame",
                    "data": b64_rgb,
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.debug("Error encoding RGB frame: %s", e)

        depth_frame = getattr(self.system_state, "depth_frame", None)
        if depth_frame is not None:
            try:
                _, buf = cv2.imencode(".jpg", depth_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                b64_depth = base64.b64encode(buf).decode("utf-8")
                await self.broadcast_json({
                    "type": "depth_frame",
                    "data": b64_depth,
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.debug("Error encoding Depth frame: %s", e)
