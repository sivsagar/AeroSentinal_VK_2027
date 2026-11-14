"""
Unit and Integration Tests for Unified Ground Control Dashboard
================================================================
Covers:
- Dashboard loading & template rendering
- API endpoints (/api/status, /api/health, /api/perception, /api/tracks, /api/obstacle, /api/events, /api/video/*)
- WebSocket updates & broadcasting
- Multiple concurrent client connections
- Missing sensor values & graceful degradation
- Stale metrics detection
- Tracking updates
- Obstacle reports
- Event history & deduplication
- Resource cleanup
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from starlette.testclient import TestClient

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from aerosentinel.dashboard.app import DashboardApp, DashboardEvent, SystemState


class TestDashboardUnified:
    """Comprehensive test suite for the unified robotics ground control dashboard."""

    @pytest.fixture
    def test_env(self):
        state = SystemState()
        state.rgb_camera_connected = True
        state.d435i_connected = True
        state.picam_connected = True
        state.ai_engine_running = True
        state.tracking_running = True
        state.obstacle_running = True

        app_wrapper = DashboardApp()
        app = app_wrapper.create_app(system_state=state)
        client = TestClient(app)
        return {"state": state, "app": app, "wrapper": app_wrapper, "client": client}

    # 1. Dashboard Loading & HTML Template
    def test_get_index(self, test_env):
        client = test_env["client"]
        resp = client.get("/")
        assert resp.status_code == 200
        assert "AEROSENTINEL" in resp.text
        assert "RealSense D435i RGB Perception Feed" in resp.text
        assert "Obstacle Proximity Advisory" in resp.text
        assert "Multi-Person Tracking" in resp.text

    # 2. System Status API
    def test_get_status(self, test_env):
        client = test_env["client"]
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rgb_camera"] == "connected"
        assert data["d435i"] == "connected"
        assert data["picam_camera"] == "connected"
        assert data["ai_engine"] == "running"
        assert data["person_tracker"] == "running"
        assert data["obstacle_detector"] == "running"
        assert "uptime_s" in data

    # 3. Health Endpoint & Stale Detection
    def test_get_health_fresh_and_stale(self, test_env):
        client = test_env["client"]
        state = test_env["state"]

        # Initially stale before metric update
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_stale"] is True

        # Refresh metrics
        state.metrics_timestamp = time.time()
        state.cpu_percent = 22.4
        state.cpu_temp_c = 48.5
        state.ram_percent = 35.0
        state.ram_used_mb = 1200.0
        state.ram_total_mb = 4000.0
        state.camera_fps = 14.8

        resp_fresh = client.get("/api/health")
        data_fresh = resp_fresh.json()
        assert data_fresh["is_stale"] is False
        assert data_fresh["cpu_percent"] == 22.4
        assert data_fresh["cpu_temp_c"] == 48.5
        assert data_fresh["camera_fps"] == 14.8

        # Stale when past threshold
        state.metrics_timestamp = time.time() - 10.0
        assert state.is_metrics_stale(threshold_s=3.0) is True

    # 4. Perception Pipeline API
    def test_get_perception(self, test_env):
        client = test_env["client"]
        state = test_env["state"]
        state.inference_fps = 3.2
        state.active_tracks = [{"track_id": 1}, {"track_id": 2}]

        resp = client.get("/api/perception")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ai_engine_running"] is True
        assert data["active_track_count"] == 2
        assert data["inference_fps"] == 3.2

    # 5. Multi-Person Tracking API
    def test_get_tracks(self, test_env):
        client = test_env["client"]
        state = test_env["state"]
        sample_tracks = [
            {
                "track_id": 1,
                "confidence": 0.94,
                "distance_m": 2.85,
                "distance_valid": True,
                "state": "confirmed",
                "age_frames": 54,
                "last_seen_ts": time.time(),
            },
            {
                "track_id": 2,
                "confidence": 0.88,
                "distance_m": 4.10,
                "distance_valid": True,
                "state": "tentative",
                "age_frames": 3,
                "last_seen_ts": time.time(),
            },
        ]
        state.active_tracks = sample_tracks

        resp = client.get("/api/tracks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["track_count"] == 2
        assert len(data["tracks"]) == 2
        assert data["tracks"][0]["track_id"] == 1

        # Test alias endpoint /api/tracking
        resp_alias = client.get("/api/tracking")
        assert resp_alias.status_code == 200
        assert resp_alias.json()["track_count"] == 2

    # 6. Obstacle Proximity API
    def test_get_obstacle(self, test_env):
        client = test_env["client"]
        state = test_env["state"]

        # Default fallback when no obstacle report present
        resp_def = client.get("/api/obstacle")
        assert resp_def.status_code == 200
        data_def = resp_def.json()
        assert data_def["overall_status"] == "CLEAR"

        # Populated rich obstacle report
        state.latest_obstacle = {
            "overall_status": "WARNING",
            "warning_active": True,
            "caution_active": True,
            "critical_active": False,
            "closest_obstacle_m": 1.45,
            "grid_layout": "1x3",
            "closest_reliable_candidate": {"distance_m": 1.45, "region": "center"},
            "left": {"status": "CLEAR", "min_distance_m": 4.0},
            "center": {"status": "WARNING", "min_distance_m": 1.45},
            "right": {"status": "CLEAR", "min_distance_m": 3.8},
        }

        resp = client.get("/api/obstacle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_status"] == "WARNING"
        assert data["warning_active"] is True
        assert data["closest_obstacle_m"] == 1.45
        assert data["center"]["status"] == "WARNING"

    # 7. Event History API & Deduplication
    def test_event_history_and_deduplication(self, test_env):
        client = test_env["client"]
        state = test_env["state"]

        # Add initial event
        evt1 = state.add_event("PERSON_DETECTED", "Person detected at 2.5m", "INFO")
        assert evt1.count == 1

        # Add duplicate event within deduplication window
        evt2 = state.add_event("PERSON_DETECTED", "Person detected at 2.5m", "INFO")
        assert evt2.count == 2
        assert evt2.id == evt1.id

        # Add distinct event
        evt3 = state.add_event("PROXIMITY_CRITICAL", "Near-field hazard at 0.6m", "CRITICAL")
        assert evt3.count == 1
        assert evt3.id != evt1.id

        # Fetch events via API
        resp = client.get("/api/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] >= 2
        assert data["events"][0]["event_type"] == "PROXIMITY_CRITICAL"

        # Post new event via API
        post_resp = client.post("/api/events", json={
            "event_type": "CAMERA_UNAVAILABLE",
            "message": "RealSense depth frame timeout",
            "severity": "WARNING",
        })
        assert post_resp.status_code == 200
        assert post_resp.json()["status"] == "created"

    # 8. Video Frame Stream Endpoints
    def test_video_endpoints(self, test_env):
        client = test_env["client"]
        state = test_env["state"]

        # Blank placeholders when frames are None
        resp_rgb_ph = client.get("/api/video/rgb")
        assert resp_rgb_ph.status_code == 200
        assert resp_rgb_ph.headers["content-type"] == "image/jpeg"
        assert len(resp_rgb_ph.content) > 100

        # Feed real numpy image
        fake_frame = np.full((120, 160, 3), 150, dtype=np.uint8)
        state.rgb_frame = fake_frame
        state.depth_frame = fake_frame
        state.picam_frame = fake_frame

        resp_rgb = client.get("/api/video/rgb")
        assert resp_rgb.status_code == 200
        assert resp_rgb.headers["content-type"] == "image/jpeg"

        resp_depth = client.get("/api/video/depth")
        assert resp_depth.status_code == 200
        assert resp_depth.headers["content-type"] == "image/jpeg"

        resp_picam = client.get("/api/video/picam")
        assert resp_picam.status_code == 200
        assert resp_picam.headers["content-type"] == "image/jpeg"

    # 9. Mission Lifecycle API
    def test_mission_lifecycle(self, test_env):
        client = test_env["client"]

        resp_start = client.post("/api/mission/start", json={"name": "Alps Rescue", "operator": "Commander"})
        assert resp_start.status_code == 200
        assert resp_start.json()["status"] == "started"

        resp_curr = client.get("/api/mission/current")
        assert resp_curr.status_code == 200
        assert resp_curr.json()["active"] is True

        resp_pause = client.post("/api/mission/pause")
        assert resp_pause.status_code == 200
        assert resp_pause.json()["status"] == "paused"

        resp_resume = client.post("/api/mission/resume")
        assert resp_resume.status_code == 200
        assert resp_resume.json()["status"] == "resumed"

        resp_stop = client.post("/api/mission/stop")
        assert resp_stop.status_code == 200
        assert resp_stop.json()["status"] == "stopped"

    # 10. Missing Sensor Values & Graceful Degradation
    def test_missing_sensor_values_graceful_degradation(self, test_env):
        client = test_env["client"]
        state = test_env["state"]

        # Mark all sensors disconnected
        state.rgb_camera_connected = False
        state.d435i_connected = False
        state.picam_connected = False
        state.ai_engine_running = False
        state.tracking_running = False
        state.obstacle_running = False
        state.active_tracks = []
        state.latest_obstacle = None
        state.cpu_percent = None
        state.cpu_temp_c = None

        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["realsense_rgb"] == "disconnected"
        assert resp.json()["realsense_depth"] == "disconnected"
        assert resp.json()["picam_camera"] == "disconnected"

        resp_obs = client.get("/api/obstacle")
        assert resp_obs.status_code == 200
        assert resp_obs.json()["overall_status"] == "CLEAR"

        resp_tracks = client.get("/api/tracks")
        assert resp_tracks.status_code == 200
        assert resp_tracks.json()["track_count"] == 0

    # 11. WebSocket Connection & Broadcast Updates
    def test_websocket_broadcast_types(self, test_env):
        client = test_env["client"]
        with client.websocket_connect("/ws") as ws:
            init_msg = ws.receive_json()
            assert init_msg["type"] == "system_status"

    # 12. Multiple WebSocket Clients & Resource Cleanup
    def test_multiple_websocket_clients_and_cleanup(self, test_env):
        client = test_env["client"]
        app = test_env["app"]
        ws_mgr = app.state.ws_manager

        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                assert len(ws_mgr._active_connections) == 2
                msg1 = ws1.receive_json()
                msg2 = ws2.receive_json()
                assert msg1["type"] == "system_status"
                assert msg2["type"] == "system_status"

        # Connections must be cleaned up on disconnect
        assert len(ws_mgr._active_connections) == 0
