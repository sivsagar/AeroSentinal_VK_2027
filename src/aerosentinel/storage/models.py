"""
AeroSentinel Storage Models
SQLite schema definitions and dataclasses (no SQLAlchemy).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MissionStatus(str, Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


# ---------------------------------------------------------------------------
# SQL Schema Strings
# ---------------------------------------------------------------------------

MISSION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS missions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'CREATED',
    start_time  REAL,
    end_time    REAL,
    notes       TEXT,
    operator    TEXT
);
"""

DETECTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id          INTEGER NOT NULL,
    timestamp           REAL    NOT NULL,
    class_name          TEXT,
    confidence          REAL,
    distance_m          REAL,
    bbox_json           TEXT,
    image_path          TEXT,
    hazard_category     TEXT,
    verification_status TEXT    DEFAULT 'UNVERIFIED',
    lat                 REAL,
    lon                 REAL,
    FOREIGN KEY (mission_id) REFERENCES missions(id)
);
"""

TELEMETRY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS telemetry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id  INTEGER NOT NULL,
    timestamp   REAL    NOT NULL,
    battery_pct REAL,
    altitude_m  REAL,
    speed_ms    REAL,
    flight_mode TEXT,
    armed       INTEGER,
    lat         REAL,
    lon         REAL,
    roll        REAL,
    pitch       REAL,
    yaw         REAL,
    FOREIGN KEY (mission_id) REFERENCES missions(id)
);
"""

SYSTEM_METRIC_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS system_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id      INTEGER NOT NULL,
    timestamp       REAL    NOT NULL,
    cpu_pct         REAL,
    ram_pct         REAL,
    cpu_temp_c      REAL,
    fps             REAL,
    dropped_frames  INTEGER,
    FOREIGN KEY (mission_id) REFERENCES missions(id)
);
"""

EVENT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id      INTEGER,
    timestamp       REAL    NOT NULL,
    event_type      TEXT    NOT NULL,
    severity        TEXT    NOT NULL DEFAULT 'INFO',
    message         TEXT    NOT NULL,
    metadata_json   TEXT,
    FOREIGN KEY (mission_id) REFERENCES missions(id)
);
"""

ALL_TABLES_SQL = [
    MISSION_TABLE_SQL,
    DETECTION_TABLE_SQL,
    TELEMETRY_TABLE_SQL,
    SYSTEM_METRIC_TABLE_SQL,
    EVENT_TABLE_SQL,
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Mission:
    id: Optional[int]
    name: str
    status: MissionStatus = MissionStatus.CREATED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    notes: Optional[str] = None
    operator: Optional[str] = None


@dataclass
class Detection:
    mission_id: int
    timestamp: float
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    distance_m: Optional[float] = None
    bbox_json: Optional[str] = None
    image_path: Optional[str] = None
    hazard_category: Optional[str] = None
    verification_status: str = "UNVERIFIED"
    lat: Optional[float] = None
    lon: Optional[float] = None
    id: Optional[int] = None


@dataclass
class Telemetry:
    mission_id: int
    timestamp: float
    battery_pct: Optional[float] = None
    altitude_m: Optional[float] = None
    speed_ms: Optional[float] = None
    flight_mode: Optional[str] = None
    armed: Optional[bool] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    id: Optional[int] = None


@dataclass
class SystemMetric:
    mission_id: int
    timestamp: float
    cpu_pct: Optional[float] = None
    ram_pct: Optional[float] = None
    cpu_temp_c: Optional[float] = None
    fps: Optional[float] = None
    dropped_frames: Optional[int] = None
    id: Optional[int] = None


@dataclass
class Event:
    timestamp: float
    event_type: str
    message: str
    severity: str = "INFO"
    mission_id: Optional[int] = None
    metadata_json: Optional[str] = None
    id: Optional[int] = None
