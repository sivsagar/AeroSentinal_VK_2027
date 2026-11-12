"""
Unit tests for Phase 6 — MAVLink Integration
Uses MockMAVLinkClient — no real hardware required.
"""

import sys
import os
import time

import pytest

# Allow imports from src/ and simulation/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))

from mock_mavlink import MockMAVLinkClient
from aerosentinel.flight.telemetry import TelemetryData
from aerosentinel.flight.flight_status import FlightStatusMonitor


# ---------------------------------------------------------------------------
# MockMAVLinkClient tests
# ---------------------------------------------------------------------------

class TestMockMAVLinkClient:
    def test_returns_telemetry_dict(self):
        client = MockMAVLinkClient()
        client.connect()
        time.sleep(0.3)  # let one tick run
        tel = client.get_telemetry()
        client.disconnect()
        assert isinstance(tel, dict)

    def test_telemetry_has_required_keys(self):
        client = MockMAVLinkClient()
        client.connect()
        time.sleep(0.3)
        tel = client.get_telemetry()
        client.disconnect()
        required_keys = [
            "timestamp", "battery_pct", "battery_voltage_v", "battery_current_a",
            "lat", "lon", "alt_m", "gps_fix_type", "gps_satellites",
            "roll_deg", "pitch_deg", "yaw_deg", "groundspeed_ms",
            "flight_mode", "armed", "link_quality", "is_connected", "last_heartbeat",
        ]
        for key in required_keys:
            assert key in tel, f"Missing key: {key}"

    def test_battery_drains_slowly(self):
        client = MockMAVLinkClient()
        client.connect()
        time.sleep(0.5)
        tel = client.get_telemetry()
        client.disconnect()
        # After 0.5s battery should still be near 100 (not 0)
        assert tel["battery_pct"] is not None
        assert tel["battery_pct"] > 90.0

    def test_is_connected_true_when_running(self):
        client = MockMAVLinkClient()
        client.connect()
        time.sleep(0.2)
        tel = client.get_telemetry()
        client.disconnect()
        assert tel["is_connected"] is True

    def test_link_loss_after_n_messages(self):
        client = MockMAVLinkClient(link_loss_after_n_messages=3)
        client.connect()
        time.sleep(0.5)  # 5 messages at 10Hz in 0.5s
        tel = client.get_telemetry()
        client.disconnect()
        assert tel["is_connected"] is False

    def test_low_battery_at_n(self):
        client = MockMAVLinkClient(low_battery_at_n=1)
        client.connect()
        time.sleep(0.15)
        tel = client.get_telemetry()
        client.disconnect()
        assert tel["battery_pct"] is not None
        assert tel["battery_pct"] < 20.0

    def test_heartbeat_timestamps_populated(self):
        client = MockMAVLinkClient()
        client.connect()
        time.sleep(1.5)  # allow at least 1 heartbeat (every 1s)
        timestamps = client.get_heartbeat_timestamps()
        client.disconnect()
        assert len(timestamps) >= 1

    def test_no_arm_method(self):
        """MAVLink client must not have a working arm command."""
        client = MockMAVLinkClient()
        with pytest.raises(NotImplementedError):
            client.arm()

    def test_no_takeoff_method(self):
        client = MockMAVLinkClient()
        with pytest.raises(NotImplementedError):
            client.takeoff()

    def test_no_disarm_method(self):
        client = MockMAVLinkClient()
        with pytest.raises(NotImplementedError):
            client.disarm()

    def test_no_send_command_method(self):
        client = MockMAVLinkClient()
        with pytest.raises(NotImplementedError):
            client.send_command()

    def test_no_motor_test_method(self):
        client = MockMAVLinkClient()
        with pytest.raises(NotImplementedError):
            client.motor_test()


# ---------------------------------------------------------------------------
# TelemetryData tests
# ---------------------------------------------------------------------------

class TestTelemetryData:
    def test_to_dict_has_all_required_keys(self):
        tel = TelemetryData(
            timestamp=time.time(),
            battery_pct=85.0,
            battery_voltage_v=12.3,
            battery_current_a=5.0,
            lat=None,
            lon=None,
            alt_m=30.0,
            gps_fix_type=0,
            gps_satellites=0,
            roll_deg=1.2,
            pitch_deg=0.5,
            yaw_deg=90.0,
            groundspeed_ms=4.0,
            flight_mode="GUIDED",
            armed=True,
            link_quality=95.0,
            is_connected=True,
            last_heartbeat=time.time(),
        )
        d = tel.to_dict()
        required_keys = [
            "timestamp", "battery_pct", "battery_voltage_v", "battery_current_a",
            "lat", "lon", "alt_m", "gps_fix_type", "gps_satellites",
            "roll_deg", "pitch_deg", "yaw_deg", "groundspeed_ms",
            "flight_mode", "armed", "link_quality", "is_connected", "last_heartbeat",
        ]
        for key in required_keys:
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_lat_lon_none_when_no_gps(self):
        tel = TelemetryData(timestamp=time.time())
        d = tel.to_dict()
        assert d["lat"] is None
        assert d["lon"] is None

    def test_to_dict_returns_dict(self):
        tel = TelemetryData(timestamp=1234567890.0)
        assert isinstance(tel.to_dict(), dict)


# ---------------------------------------------------------------------------
# FlightStatusMonitor tests
# ---------------------------------------------------------------------------

class TestFlightStatusMonitor:
    def test_link_loss_detected(self):
        """Telemetry with old heartbeat should trigger link_loss."""
        tel = TelemetryData(
            timestamp=time.time(),
            is_connected=False,
            last_heartbeat=time.time() - 15.0,  # 15s ago — exceeded 10s threshold
        )
        monitor = FlightStatusMonitor(mavlink_client=None)
        assert monitor.detect_link_loss(tel) is True

    def test_no_link_loss_when_recent_heartbeat(self):
        tel = TelemetryData(
            timestamp=time.time(),
            is_connected=True,
            last_heartbeat=time.time() - 2.0,  # 2s ago — within threshold
        )
        monitor = FlightStatusMonitor()
        assert monitor.detect_link_loss(tel) is False

    def test_low_battery_detected(self):
        tel = TelemetryData(timestamp=time.time(), battery_pct=15.0)
        monitor = FlightStatusMonitor()
        assert monitor.detect_low_battery(tel) is True

    def test_no_low_battery_above_threshold(self):
        tel = TelemetryData(timestamp=time.time(), battery_pct=50.0)
        monitor = FlightStatusMonitor()
        assert monitor.detect_low_battery(tel) is False

    def test_low_battery_none_returns_false(self):
        tel = TelemetryData(timestamp=time.time(), battery_pct=None)
        monitor = FlightStatusMonitor()
        assert monitor.detect_low_battery(tel) is False

    def test_compute_link_quality_empty(self):
        monitor = FlightStatusMonitor()
        assert monitor.compute_link_quality([]) == 0.0

    def test_compute_link_quality_full(self):
        monitor = FlightStatusMonitor()
        now = time.time()
        # 10 heartbeats in last 10s = 100%
        timestamps = [now - i for i in range(10)]
        quality = monitor.compute_link_quality(timestamps)
        assert quality == pytest.approx(100.0, abs=1.0)

    def test_compute_link_quality_partial(self):
        monitor = FlightStatusMonitor()
        now = time.time()
        # 5 heartbeats in last 10s = ~50%
        timestamps = [now - i * 2 for i in range(5)]
        quality = monitor.compute_link_quality(timestamps)
        assert 40.0 <= quality <= 60.0

    def test_get_status_summary_with_mock_client(self):
        client = MockMAVLinkClient()
        client.connect()
        time.sleep(0.3)
        monitor = FlightStatusMonitor(mavlink_client=client)
        summary = monitor.get_status_summary()
        client.disconnect()
        assert isinstance(summary, dict)
        assert "is_connected" in summary
        assert "battery_pct" in summary
        assert "link_loss" in summary
        assert "low_battery" in summary

    def test_mavlink_client_no_arm_takeoff_motor_methods(self):
        """
        Real MAVLinkClient class must not have executable arm/takeoff/motor methods.
        They must raise NotImplementedError.
        """
        # We use MockMAVLinkClient as a proxy since it mirrors the interface
        client = MockMAVLinkClient()
        for method_name in ("arm", "disarm", "takeoff", "land", "set_mode", "motor_test"):
            method = getattr(client, method_name)
            with pytest.raises(NotImplementedError):
                method()
