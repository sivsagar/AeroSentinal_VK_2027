"""
AeroSentinel Mission Manager
Manages the full lifecycle of search-and-rescue missions.
"""

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

from aerosentinel.storage.database import DatabaseManager
from aerosentinel.storage.models import (
    Detection,
    Mission,
    MissionStatus,
    SystemMetric,
    Telemetry,
)

logger = logging.getLogger(__name__)


class MissionManager:
    """Manages mission lifecycle with auto-save and event callbacks."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        auto_save_interval: float = 30.0,
        snapshot_dir: str = "data/detections",
    ):
        self._db = db_manager
        self._auto_save_interval = auto_save_interval
        self._snapshot_dir = snapshot_dir

        self._active_mission_id: Optional[int] = None
        self._lock = threading.Lock()

        # Event callbacks
        self._on_mission_start: List[Callable] = []
        self._on_mission_stop: List[Callable] = []
        self._on_detection: List[Callable] = []

        # Auto-save background thread
        self._auto_save_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_mission_start(self, fn: Callable) -> None:
        self._on_mission_start.append(fn)

    def on_mission_stop(self, fn: Callable) -> None:
        self._on_mission_stop.append(fn)

    def on_detection(self, fn: Callable) -> None:
        self._on_detection.append(fn)

    def _emit(self, callbacks: List[Callable], *args, **kwargs) -> None:
        for cb in callbacks:
            try:
                cb(*args, **kwargs)
            except Exception as exc:
                logger.warning("Callback error: %s", exc)

    # ------------------------------------------------------------------
    # Mission lifecycle
    # ------------------------------------------------------------------

    def create_mission(
        self, name: str, operator: str = "", notes: str = ""
    ) -> Mission:
        mission_id = self._db.create_mission(name, operator, notes)
        mission = self._db.get_mission(mission_id)
        logger.info("Mission created: id=%d name=%s", mission_id, name)
        return mission

    def start_mission(self, mission_id: int) -> bool:
        mission = self._db.get_mission(mission_id)
        if mission is None:
            logger.error("start_mission: mission %d not found", mission_id)
            return False
        if mission.status not in (MissionStatus.CREATED, MissionStatus.PAUSED):
            logger.warning(
                "start_mission: invalid status %s for mission %d",
                mission.status,
                mission_id,
            )
            return False

        now = time.time()
        self._db.update_mission_status(mission_id, MissionStatus.ACTIVE)
        self._db.set_mission_start_time(mission_id, now)

        with self._lock:
            self._active_mission_id = mission_id

        self._start_auto_save()
        mission = self._db.get_mission(mission_id)
        self._emit(self._on_mission_start, mission)
        logger.info("Mission started: id=%d", mission_id)
        return True

    def pause_mission(self, mission_id: int) -> bool:
        mission = self._db.get_mission(mission_id)
        if mission is None or mission.status != MissionStatus.ACTIVE:
            return False
        self._db.update_mission_status(mission_id, MissionStatus.PAUSED)
        logger.info("Mission paused: id=%d", mission_id)
        return True

    def resume_mission(self, mission_id: int) -> bool:
        mission = self._db.get_mission(mission_id)
        if mission is None or mission.status != MissionStatus.PAUSED:
            return False
        self._db.update_mission_status(mission_id, MissionStatus.ACTIVE)
        logger.info("Mission resumed: id=%d", mission_id)
        return True

    def stop_mission(self, mission_id: int, reason: str = "") -> bool:
        return self._finish_mission(mission_id, MissionStatus.COMPLETED, reason)

    def abort_mission(self, mission_id: int, reason: str = "") -> bool:
        return self._finish_mission(mission_id, MissionStatus.ABORTED, reason)

    def _finish_mission(
        self, mission_id: int, final_status: MissionStatus, reason: str
    ) -> bool:
        mission = self._db.get_mission(mission_id)
        if mission is None:
            return False
        if mission.status not in (
            MissionStatus.ACTIVE,
            MissionStatus.PAUSED,
            MissionStatus.CREATED,
        ):
            return False

        now = time.time()
        self._db.update_mission_status(mission_id, final_status)
        self._db.end_mission(mission_id, now)

        with self._lock:
            if self._active_mission_id == mission_id:
                self._active_mission_id = None

        self._stop_auto_save()
        mission = self._db.get_mission(mission_id)
        self._emit(self._on_mission_stop, mission, reason)
        logger.info("Mission %s: id=%d reason=%s", final_status.value, mission_id, reason)
        return True

    # ------------------------------------------------------------------
    # Data recording
    # ------------------------------------------------------------------

    def record_detection(self, mission_id: int, fused_detection: dict) -> int:
        det = Detection(
            mission_id=mission_id,
            timestamp=fused_detection.get("timestamp", time.time()),
            class_name=fused_detection.get("class_name"),
            confidence=fused_detection.get("confidence"),
            distance_m=fused_detection.get("distance_m"),
            bbox_json=fused_detection.get("bbox_json"),
            image_path=fused_detection.get("image_path"),
            hazard_category=fused_detection.get("hazard_category"),
            verification_status=fused_detection.get("verification_status", "UNVERIFIED"),
            lat=fused_detection.get("lat"),  # None until real GPS
            lon=fused_detection.get("lon"),  # None until real GPS
        )
        det_id = self._db.save_detection(det)
        self._emit(self._on_detection, det)
        return det_id

    def record_telemetry(self, mission_id: int, telemetry_data: dict) -> int:
        tel = Telemetry(
            mission_id=mission_id,
            timestamp=telemetry_data.get("timestamp", time.time()),
            battery_pct=telemetry_data.get("battery_pct"),
            altitude_m=telemetry_data.get("altitude_m") or telemetry_data.get("alt_m"),
            speed_ms=telemetry_data.get("speed_ms") or telemetry_data.get("groundspeed_ms"),
            flight_mode=telemetry_data.get("flight_mode"),
            armed=telemetry_data.get("armed"),
            lat=telemetry_data.get("lat"),
            lon=telemetry_data.get("lon"),
            roll=telemetry_data.get("roll_deg"),
            pitch=telemetry_data.get("pitch_deg"),
            yaw=telemetry_data.get("yaw_deg"),
        )
        return self._db.save_telemetry(tel)

    def record_snapshot(self, mission_id: int, frame, detection_id: int) -> str:
        """Save a frame snapshot to disk. Returns the file path."""
        import os
        import json

        snapshot_dir = os.path.join(self._snapshot_dir, str(mission_id))
        os.makedirs(snapshot_dir, exist_ok=True)
        ts = time.time()
        filename = f"{ts:.3f}.jpg"
        filepath = os.path.join(snapshot_dir, filename)

        try:
            import cv2
            cv2.imwrite(filepath, frame)
        except Exception:
            # If cv2 not available, save a placeholder
            with open(filepath, "wb") as fh:
                fh.write(b"")

        # JSON metadata
        meta_path = filepath.replace(".jpg", ".json")
        with open(meta_path, "w") as fh:
            json.dump(
                {"mission_id": mission_id, "detection_id": detection_id, "timestamp": ts},
                fh,
            )

        return filepath

    def record_system_metric(self, mission_id: int, metric: dict) -> int:
        sm = SystemMetric(
            mission_id=mission_id,
            timestamp=metric.get("timestamp", time.time()),
            cpu_pct=metric.get("cpu_pct"),
            ram_pct=metric.get("ram_pct"),
            cpu_temp_c=metric.get("cpu_temp_c"),
            fps=metric.get("fps"),
            dropped_frames=metric.get("dropped_frames"),
        )
        return self._db.save_system_metric(sm)

    def get_active_mission(self) -> Optional[Mission]:
        with self._lock:
            mid = self._active_mission_id
        if mid is None:
            return None
        return self._db.get_mission(mid)

    # ------------------------------------------------------------------
    # Auto-save background thread
    # ------------------------------------------------------------------

    def _start_auto_save(self) -> None:
        if self._running:
            return
        self._running = True
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_loop, daemon=True, name="MissionAutoSave"
        )
        self._auto_save_thread.start()

    def _stop_auto_save(self) -> None:
        self._running = False

    def _auto_save_loop(self) -> None:
        while self._running:
            time.sleep(self._auto_save_interval)
            with self._lock:
                mid = self._active_mission_id
            if mid is not None:
                logger.debug("Auto-save heartbeat for mission %d", mid)

    def shutdown(self) -> None:
        """Graceful shutdown: stop auto-save, finalize any active mission."""
        self._stop_auto_save()
        with self._lock:
            mid = self._active_mission_id
        if mid is not None:
            logger.info("Shutdown: stopping active mission %d", mid)
            self.stop_mission(mid, reason="system_shutdown")
