"""
AeroSentinel Database Manager
Thread-safe SQLite manager using Python's built-in sqlite3.
"""

import sqlite3
import threading
import time
import logging
from typing import List, Optional

from aerosentinel.storage.models import (
    ALL_TABLES_SQL,
    Detection,
    Mission,
    MissionStatus,
    SystemMetric,
    Telemetry,
)

logger = logging.getLogger(__name__)

_RETRY_COUNT = 5
_RETRY_DELAY = 0.1  # seconds


def _retry_on_locked(fn):
    """Decorator: retry up to _RETRY_COUNT times on OperationalError (locked)."""
    def wrapper(*args, **kwargs):
        for attempt in range(_RETRY_COUNT):
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < _RETRY_COUNT - 1:
                    time.sleep(_RETRY_DELAY)
                else:
                    raise
    return wrapper


class DatabaseManager:
    """Thread-safe SQLite database manager for AeroSentinel."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the database connection and enable WAL mode."""
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=10,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent reads alongside writes
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.commit()
        logger.debug("Database connected: %s", self._db_path)

    def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database disconnected.")

    # Alias
    close = disconnect

    def create_tables(self) -> None:
        """Create all tables if they do not already exist."""
        with self._lock:
            cur = self._conn.cursor()
            for sql in ALL_TABLES_SQL:
                cur.execute(sql)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        self.create_tables()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    # ------------------------------------------------------------------
    # Mission CRUD
    # ------------------------------------------------------------------

    @_retry_on_locked
    def create_mission(self, name: str, operator: str = "", notes: str = "") -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO missions (name, operator, notes, status) VALUES (?, ?, ?, ?)",
                (name, operator, notes, MissionStatus.CREATED.value),
            )
            self._conn.commit()
            return cur.lastrowid

    @_retry_on_locked
    def update_mission_status(self, mission_id: int, status: MissionStatus) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE missions SET status=? WHERE id=?",
                (status.value, mission_id),
            )
            self._conn.commit()

    @_retry_on_locked
    def end_mission(self, mission_id: int, end_time: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE missions SET end_time=? WHERE id=?",
                (end_time, mission_id),
            )
            self._conn.commit()

    @_retry_on_locked
    def set_mission_start_time(self, mission_id: int, start_time: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE missions SET start_time=? WHERE id=?",
                (start_time, mission_id),
            )
            self._conn.commit()

    def get_mission(self, mission_id: int) -> Optional[Mission]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM missions WHERE id=?", (mission_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_mission(row)

    def get_all_missions(self) -> List[Mission]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM missions ORDER BY id").fetchall()
        return [self._row_to_mission(r) for r in rows]

    @staticmethod
    def _row_to_mission(row: sqlite3.Row) -> Mission:
        return Mission(
            id=row["id"],
            name=row["name"],
            status=MissionStatus(row["status"]),
            start_time=row["start_time"],
            end_time=row["end_time"],
            notes=row["notes"],
            operator=row["operator"],
        )

    # ------------------------------------------------------------------
    # Detection CRUD
    # ------------------------------------------------------------------

    @_retry_on_locked
    def save_detection(self, detection: Detection) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO detections
                   (mission_id, timestamp, class_name, confidence, distance_m,
                    bbox_json, image_path, hazard_category, verification_status, lat, lon)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    detection.mission_id,
                    detection.timestamp,
                    detection.class_name,
                    detection.confidence,
                    detection.distance_m,
                    detection.bbox_json,
                    detection.image_path,
                    detection.hazard_category,
                    detection.verification_status,
                    detection.lat,
                    detection.lon,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_mission_detections(self, mission_id: int) -> List[Detection]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM detections WHERE mission_id=? ORDER BY timestamp",
                (mission_id,),
            ).fetchall()
        return [self._row_to_detection(r) for r in rows]

    def get_recent_detections(self, mission_id: int, limit: int = 10) -> List[Detection]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM detections WHERE mission_id=? ORDER BY timestamp DESC LIMIT ?",
                (mission_id, limit),
            ).fetchall()
        return [self._row_to_detection(r) for r in rows]

    @staticmethod
    def _row_to_detection(row: sqlite3.Row) -> Detection:
        return Detection(
            id=row["id"],
            mission_id=row["mission_id"],
            timestamp=row["timestamp"],
            class_name=row["class_name"],
            confidence=row["confidence"],
            distance_m=row["distance_m"],
            bbox_json=row["bbox_json"],
            image_path=row["image_path"],
            hazard_category=row["hazard_category"],
            verification_status=row["verification_status"],
            lat=row["lat"],
            lon=row["lon"],
        )

    # ------------------------------------------------------------------
    # Telemetry CRUD
    # ------------------------------------------------------------------

    @_retry_on_locked
    def save_telemetry(self, telemetry: Telemetry) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO telemetry
                   (mission_id, timestamp, battery_pct, altitude_m, speed_ms,
                    flight_mode, armed, lat, lon, roll, pitch, yaw)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    telemetry.mission_id,
                    telemetry.timestamp,
                    telemetry.battery_pct,
                    telemetry.altitude_m,
                    telemetry.speed_ms,
                    telemetry.flight_mode,
                    int(telemetry.armed) if telemetry.armed is not None else None,
                    telemetry.lat,
                    telemetry.lon,
                    telemetry.roll,
                    telemetry.pitch,
                    telemetry.yaw,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------
    # SystemMetric CRUD
    # ------------------------------------------------------------------

    @_retry_on_locked
    def save_system_metric(self, metric: SystemMetric) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO system_metrics
                   (mission_id, timestamp, cpu_pct, ram_pct, cpu_temp_c, fps, dropped_frames)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    metric.mission_id,
                    metric.timestamp,
                    metric.cpu_pct,
                    metric.ram_pct,
                    metric.cpu_temp_c,
                    metric.fps,
                    metric.dropped_frames,
                ),
            )
            self._conn.commit()
            return cur.lastrowid
