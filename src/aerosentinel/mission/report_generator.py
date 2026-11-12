"""
AeroSentinel Report Generator
Generates JSON and CSV reports for completed missions.
"""

import csv
import io
import json
import logging
import os
import time
from typing import Optional

from aerosentinel.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join("data", "missions", "reports")


class ReportGenerator:
    """Generate mission reports in JSON and CSV formats."""

    def __init__(self, db_manager: DatabaseManager, reports_dir: str = REPORTS_DIR):
        self._db = db_manager
        self._reports_dir = reports_dir

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def generate_json_report(self, mission_id: int) -> dict:
        """Build a complete mission report as a dictionary."""
        mission = self._db.get_mission(mission_id)
        if mission is None:
            raise ValueError(f"Mission {mission_id} not found")

        detections = self._db.get_mission_detections(mission_id)

        # Duration
        duration_s: Optional[float] = None
        if mission.start_time and mission.end_time:
            duration_s = mission.end_time - mission.start_time
        elif mission.start_time:
            duration_s = time.time() - mission.start_time

        # Heatmap data: list of distances (non-None)
        detection_heatmap_data = [
            d.distance_m for d in detections if d.distance_m is not None
        ]

        # Class breakdown
        class_counts: dict = {}
        for d in detections:
            key = d.class_name or "unknown"
            class_counts[key] = class_counts.get(key, 0) + 1

        return {
            "mission": {
                "id": mission.id,
                "name": mission.name,
                "operator": mission.operator,
                "status": mission.status.value,
                "start_time": mission.start_time,
                "end_time": mission.end_time,
                "notes": mission.notes,
                "duration_s": duration_s,
            },
            "summary": {
                "total_detections": len(detections),
                "class_breakdown": class_counts,
                "detection_heatmap_data": detection_heatmap_data,
            },
            "detections": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "distance_m": d.distance_m,
                    "bbox_json": d.bbox_json,
                    "image_path": d.image_path,
                    "hazard_category": d.hazard_category,
                    "verification_status": d.verification_status,
                    "lat": d.lat,
                    "lon": d.lon,
                }
                for d in detections
            ],
        }

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def generate_csv_report(self, mission_id: int) -> str:
        """Build a CSV string of all detections for a mission."""
        detections = self._db.get_mission_detections(mission_id)

        output = io.StringIO()
        fieldnames = [
            "id", "mission_id", "timestamp", "class_name", "confidence",
            "distance_m", "bbox_json", "image_path", "hazard_category",
            "verification_status", "lat", "lon",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for d in detections:
            writer.writerow(
                {
                    "id": d.id,
                    "mission_id": d.mission_id,
                    "timestamp": d.timestamp,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "distance_m": d.distance_m,
                    "bbox_json": d.bbox_json,
                    "image_path": d.image_path,
                    "hazard_category": d.hazard_category,
                    "verification_status": d.verification_status,
                    "lat": d.lat,
                    "lon": d.lon,
                }
            )
        return output.getvalue()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_report(self, mission_id: int, format: str = "json") -> str:
        """
        Save report to disk. Returns the file path.
        format: 'json' or 'csv'
        """
        os.makedirs(self._reports_dir, exist_ok=True)
        ts_str = f"{time.time():.0f}"
        filename = f"{mission_id}_{ts_str}.{format}"
        filepath = os.path.join(self._reports_dir, filename)

        if format == "json":
            report = self.generate_json_report(mission_id)
            with open(filepath, "w") as fh:
                json.dump(report, fh, indent=2)
        elif format == "csv":
            csv_content = self.generate_csv_report(mission_id)
            with open(filepath, "w", newline="") as fh:
                fh.write(csv_content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info("Report saved: %s", filepath)
        return filepath
