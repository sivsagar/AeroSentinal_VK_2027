"""
Unit tests for Phase 4 — Mission Management
Uses in-memory SQLite (:memory:) for all DB tests.
"""

import sys
import os
import time

import pytest

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aerosentinel.storage.database import DatabaseManager
from aerosentinel.storage.models import Detection, MissionStatus, SystemMetric, Telemetry
from aerosentinel.mission.mission_manager import MissionManager
from aerosentinel.mission.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """In-memory database for every test."""
    mgr = DatabaseManager(db_path=":memory:")
    mgr.connect()
    mgr.create_tables()
    yield mgr
    mgr.disconnect()


@pytest.fixture
def mm(db):
    """MissionManager backed by in-memory DB."""
    return MissionManager(db_manager=db, auto_save_interval=9999)


@pytest.fixture
def rg(db):
    """ReportGenerator backed by in-memory DB."""
    return ReportGenerator(db_manager=db, reports_dir="/tmp/aerosentinel_test_reports")


# ---------------------------------------------------------------------------
# DatabaseManager tests
# ---------------------------------------------------------------------------

class TestDatabaseManager:
    def test_creates_tables(self, db):
        """Tables must exist after create_tables()."""
        tables = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {row[0] for row in tables}
        assert "missions" in names
        assert "detections" in names
        assert "telemetry" in names
        assert "system_metrics" in names

    def test_create_mission_returns_id(self, db):
        mid = db.create_mission("Alpha SAR", "Pilot A", "Test mission")
        assert isinstance(mid, int)
        assert mid > 0

    def test_create_multiple_missions_unique_ids(self, db):
        id1 = db.create_mission("M1", "op", "")
        id2 = db.create_mission("M2", "op", "")
        assert id1 != id2

    def test_get_mission(self, db):
        mid = db.create_mission("Beta", "Pilot B", "note")
        m = db.get_mission(mid)
        assert m is not None
        assert m.name == "Beta"
        assert m.operator == "Pilot B"
        assert m.status == MissionStatus.CREATED

    def test_get_mission_not_found(self, db):
        m = db.get_mission(99999)
        assert m is None

    def test_get_all_missions(self, db):
        db.create_mission("M1", "", "")
        db.create_mission("M2", "", "")
        missions = db.get_all_missions()
        assert len(missions) >= 2

    def test_update_mission_status(self, db):
        mid = db.create_mission("Gamma", "", "")
        db.update_mission_status(mid, MissionStatus.ACTIVE)
        m = db.get_mission(mid)
        assert m.status == MissionStatus.ACTIVE

    def test_end_mission(self, db):
        mid = db.create_mission("Delta", "", "")
        now = time.time()
        db.end_mission(mid, now)
        m = db.get_mission(mid)
        assert m.end_time == pytest.approx(now, abs=1.0)

    def test_save_detection_stores_and_retrieves(self, db):
        mid = db.create_mission("SAR", "", "")
        det = Detection(
            mission_id=mid,
            timestamp=time.time(),
            class_name="person",
            confidence=0.92,
            distance_m=5.3,
            bbox_json='[10,20,100,200]',
            hazard_category="survivor",
            verification_status="UNVERIFIED",
            lat=None,
            lon=None,
        )
        det_id = db.save_detection(det)
        assert isinstance(det_id, int) and det_id > 0

        detections = db.get_mission_detections(mid)
        assert len(detections) == 1
        d = detections[0]
        assert d.class_name == "person"
        assert d.confidence == pytest.approx(0.92)
        assert d.distance_m == pytest.approx(5.3)
        assert d.lat is None
        assert d.lon is None

    def test_get_recent_detections_limit(self, db):
        mid = db.create_mission("SAR2", "", "")
        for i in range(15):
            db.save_detection(Detection(
                mission_id=mid,
                timestamp=time.time() + i,
                class_name=f"obj_{i}",
            ))
        recent = db.get_recent_detections(mid, limit=5)
        assert len(recent) == 5

    def test_save_telemetry(self, db):
        mid = db.create_mission("T", "", "")
        tel = Telemetry(mission_id=mid, timestamp=time.time(), battery_pct=78.5, altitude_m=25.0)
        tid = db.save_telemetry(tel)
        assert tid > 0

    def test_save_system_metric(self, db):
        mid = db.create_mission("SM", "", "")
        sm = SystemMetric(mission_id=mid, timestamp=time.time(), cpu_pct=55.0, fps=4.5)
        smid = db.save_system_metric(sm)
        assert smid > 0


# ---------------------------------------------------------------------------
# MissionManager tests
# ---------------------------------------------------------------------------

class TestMissionManager:
    def test_create_mission(self, mm):
        m = mm.create_mission("Test", "op", "notes")
        assert m is not None
        assert m.id is not None
        assert m.status == MissionStatus.CREATED

    def test_start_mission(self, mm):
        m = mm.create_mission("S1", "", "")
        result = mm.start_mission(m.id)
        assert result is True
        active = mm.get_active_mission()
        assert active is not None
        assert active.id == m.id
        assert active.status == MissionStatus.ACTIVE

    def test_stop_mission(self, mm):
        m = mm.create_mission("S2", "", "")
        mm.start_mission(m.id)
        result = mm.stop_mission(m.id, "normal stop")
        assert result is True
        assert mm.get_active_mission() is None

    def test_pause_and_resume_mission(self, mm):
        m = mm.create_mission("PR", "", "")
        mm.start_mission(m.id)
        assert mm.pause_mission(m.id) is True
        db_m = mm._db.get_mission(m.id)
        assert db_m.status == MissionStatus.PAUSED
        assert mm.resume_mission(m.id) is True
        db_m = mm._db.get_mission(m.id)
        assert db_m.status == MissionStatus.ACTIVE

    def test_abort_mission(self, mm):
        m = mm.create_mission("AB", "", "")
        mm.start_mission(m.id)
        result = mm.abort_mission(m.id, "emergency")
        assert result is True
        db_m = mm._db.get_mission(m.id)
        assert db_m.status == MissionStatus.ABORTED

    def test_status_transitions_invalid(self, mm):
        m = mm.create_mission("INVALID", "", "")
        # Cannot pause a CREATED mission
        assert mm.pause_mission(m.id) is False

    def test_start_nonexistent_mission(self, mm):
        result = mm.start_mission(99999)
        assert result is False

    def test_record_detection(self, mm):
        m = mm.create_mission("RD", "", "")
        mm.start_mission(m.id)
        det_id = mm.record_detection(m.id, {
            "class_name": "person",
            "confidence": 0.88,
            "distance_m": 7.2,
            "timestamp": time.time(),
        })
        assert det_id > 0

    def test_on_mission_start_callback(self, mm):
        events = []
        mm.on_mission_start(lambda m: events.append(("start", m.id)))
        m = mm.create_mission("CB", "", "")
        mm.start_mission(m.id)
        assert len(events) == 1
        assert events[0][0] == "start"

    def test_on_mission_stop_callback(self, mm):
        events = []
        mm.on_mission_stop(lambda m, r: events.append(("stop", m.status.value)))
        m = mm.create_mission("CB2", "", "")
        mm.start_mission(m.id)
        mm.stop_mission(m.id, "done")
        assert any(e[0] == "stop" for e in events)


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_generate_json_report_valid(self, db, rg):
        mid = db.create_mission("Report1", "op", "")
        db.update_mission_status(mid, MissionStatus.ACTIVE)
        db.set_mission_start_time(mid, time.time() - 60)

        for i in range(3):
            db.save_detection(Detection(
                mission_id=mid,
                timestamp=time.time(),
                class_name="person",
                confidence=0.9 - i * 0.05,
                distance_m=float(5 + i),
            ))

        report = rg.generate_json_report(mid)
        assert isinstance(report, dict)
        assert "mission" in report
        assert "summary" in report
        assert "detections" in report
        assert report["summary"]["total_detections"] == 3
        assert len(report["summary"]["detection_heatmap_data"]) == 3
        assert isinstance(report["summary"]["class_breakdown"], dict)

    def test_generate_json_report_no_detections(self, db, rg):
        mid = db.create_mission("Empty", "", "")
        report = rg.generate_json_report(mid)
        assert report["summary"]["total_detections"] == 0

    def test_generate_csv_report(self, db, rg):
        mid = db.create_mission("CSV", "", "")
        db.save_detection(Detection(mission_id=mid, timestamp=time.time(), class_name="car"))
        csv_str = rg.generate_csv_report(mid)
        assert "class_name" in csv_str
        assert "car" in csv_str

    def test_generate_json_report_missing_mission(self, rg):
        with pytest.raises(ValueError):
            rg.generate_json_report(99999)
