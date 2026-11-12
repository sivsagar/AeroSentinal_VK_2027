"""
AeroSentinel Mission Recorder
Data recording helper: saves frames and detections to disk with metadata.
"""

import json
import logging
import os
import threading
import time
from typing import List, Optional

from aerosentinel.storage.database import DatabaseManager
from aerosentinel.storage.models import Detection

logger = logging.getLogger(__name__)


class MissionRecorder:
    """Thread-safe helper that wraps DatabaseManager for frame + detection recording."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        snapshot_root: str = "data/detections",
    ):
        self._db = db_manager
        self._snapshot_root = snapshot_root
        self._lock = threading.Lock()

    def record_frame_with_detections(
        self,
        mission_id: int,
        frame,
        fused_detections: List[dict],
        obstacle_report: Optional[dict] = None,
    ) -> None:
        """
        Save detections to DB. If detections are present, save a JPEG snapshot
        and a JSON metadata file alongside it.
        """
        ts = time.time()

        # Save each detection to database
        saved_ids = []
        for det_dict in fused_detections:
            det = Detection(
                mission_id=mission_id,
                timestamp=det_dict.get("timestamp", ts),
                class_name=det_dict.get("class_name"),
                confidence=det_dict.get("confidence"),
                distance_m=det_dict.get("distance_m"),
                bbox_json=det_dict.get("bbox_json"),
                hazard_category=det_dict.get("hazard_category"),
                verification_status=det_dict.get("verification_status", "UNVERIFIED"),
                lat=det_dict.get("lat"),   # None until GPS available
                lon=det_dict.get("lon"),   # None until GPS available
            )
            if fused_detections:
                image_path = self._save_snapshot(mission_id, frame, ts, fused_detections, obstacle_report)
                det.image_path = image_path
            det_id = self._db.save_detection(det)
            saved_ids.append(det_id)

        if not saved_ids:
            return

        logger.debug(
            "Recorded %d detections for mission %d at t=%.3f",
            len(saved_ids),
            mission_id,
            ts,
        )

    def _save_snapshot(
        self,
        mission_id: int,
        frame,
        ts: float,
        fused_detections: List[dict],
        obstacle_report: Optional[dict],
    ) -> str:
        """Thread-safely write JPEG + JSON metadata. Returns image path."""
        snapshot_dir = os.path.join(self._snapshot_root, str(mission_id))
        os.makedirs(snapshot_dir, exist_ok=True)

        base_name = f"{ts:.3f}"
        image_path = os.path.join(snapshot_dir, f"{base_name}.jpg")
        meta_path = os.path.join(snapshot_dir, f"{base_name}.json")

        with self._lock:
            # Save image
            if frame is not None:
                try:
                    import cv2
                    cv2.imwrite(image_path, frame)
                except Exception:
                    # Fallback: write empty placeholder
                    with open(image_path, "wb") as fh:
                        fh.write(b"")
            else:
                with open(image_path, "wb") as fh:
                    fh.write(b"")

            # Save JSON metadata
            metadata = {
                "mission_id": mission_id,
                "timestamp": ts,
                "detections": fused_detections,
                "obstacle_report": obstacle_report,
            }
            with open(meta_path, "w") as fh:
                json.dump(metadata, fh, indent=2, default=str)

        return image_path
