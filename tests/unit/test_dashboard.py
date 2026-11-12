"""
Unit Tests for Dashboard REST API and WebSocket (Phase 5)
=========================================================
"""

import sys
import os
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aerosentinel.dashboard.app import DashboardApp, SystemState


class TestDashboardAPI:
    @pytest.fixture
    def client(self):
        state = SystemState()
        state.rgb_camera_connected = True
        state.d435i_connected = True
        state.ai_engine_running = True
        app_wrapper = DashboardApp()
        app = app_wrapper.create_app(system_state=state)
        return TestClient(app)

    def test_get_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "AeroSentinel" in resp.text

    def test_get_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rgb_camera"] == "connected"
        assert data["d435i"] == "connected"
        assert data["ai_engine"] == "running"

    def test_mission_lifecycle(self, client):
        # Start
        resp = client.post("/api/mission/start", json={"name": "Rescue Alpha", "operator": "Pilot 1"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

        # Current
        resp = client.get("/api/mission/current")
        assert resp.status_code == 200
        assert resp.json()["active"] is True

        # Pause
        resp = client.post("/api/mission/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

        # Resume
        resp = client.post("/api/mission/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "resumed"

        # Stop
        resp = client.post("/api/mission/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_get_telemetry_default(self, client):
        resp = client.get("/api/telemetry")
        assert resp.status_code == 200
        data = resp.json()
        assert "battery_pct" in data

    def test_get_performance(self, client):
        resp = client.get("/api/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "ram_percent" in data

    def test_websocket_connect(self, client):
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "system_status"

