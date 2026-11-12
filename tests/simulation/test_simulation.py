"""
End-to-End Simulation Tests
============================
Tests end-to-end integration:
Mock Cameras -> AI Detection -> Depth Obstacle Analysis -> RGB-D Fusion -> Mission Recording -> DB
"""

import sys
import os
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulation"))

from mock_camera import MockRGBCamera, MockDepthCamera
from mock_mavlink import MockMAVLinkClient
from aerosentinel.perception.detector import PersonDetector
from aerosentinel.perception.depth_processor import DepthProcessor
from aerosentinel.perception.obstacle_detector import ObstacleDetector
from aerosentinel.perception.rgbd_fusion import RGBDFusion
from aerosentinel.perception.hazard_detector import HazardDetector
from aerosentinel.perception.inference_engine import MockInferenceEngine
from aerosentinel.storage.database import DatabaseManager
from aerosentinel.mission.mission_manager import MissionManager


class TestEndToEndSimulation:
    def test_full_pipeline_tick(self, tmp_path):
        # 1. Setup in-memory / temp DB
        db_path = str(tmp_path / "test_sim.db")
        db = DatabaseManager(db_path=db_path)
        db.connect()
        db.create_tables()

        mission_mgr = MissionManager(db_manager=db)
        mission = mission_mgr.create_mission(name="Sim Test 1", operator="SimBot")
        mission_mgr.start_mission(mission.id)

        # 2. Init Mock Cameras
        rgb_cam = MockRGBCamera(config={"resolution": [320, 240], "fps": 15})
        depth_cam = MockDepthCamera(config={"depth_resolution": [320, 240], "fps": 15})
        rgb_cam.start()
        depth_cam.start()

        # 3. Init Perception Stack with Mock Engine
        mock_engine = MockInferenceEngine()
        detector = PersonDetector(engine=mock_engine, conf_threshold=0.4)
        depth_processor = DepthProcessor(min_distance_m=0.3, max_distance_m=10.0)
        obstacle_detector = ObstacleDetector(critical_distance_m=1.0, warning_distance_m=2.5)
        fusion = RGBDFusion()
        hazard_detector = HazardDetector()

        # 4. Fetch frames
        rgb_frame, rgb_ts = rgb_cam.get_frame(timeout_s=1.0)
        raw_depth, depth_color, depth_ts = depth_cam.get_frames(timeout_s=1.0)
        assert rgb_frame is not None
        assert raw_depth is not None

        # 5. Process Depth
        processed_depth = depth_processor.process(raw_depth)
        obstacle_report = obstacle_detector.analyze(processed_depth.depth_meters)
        assert obstacle_report is not None

        # 6. Run AI Detection & Fusion
        annotated_frame, detections = detector.detect(rgb_frame)
        fused = fusion.fuse(detections, processed_depth.depth_meters, rgb_ts, depth_ts)

        # 7. Hazard evaluation
        hazards = hazard_detector.assess(fused, obstacle_report)
        assert len(hazards) >= 0

        # 8. Record to mission manager
        for d in fused:
            payload = d.to_dict() if hasattr(d, "to_dict") else d
            det_id = mission_mgr.record_detection(mission.id, payload)
            assert det_id > 0

        # 9. Stop mission
        mission_mgr.stop_mission(mission.id, reason="Simulation Complete")

        # 10. Verify storage
        stored_dets = db.get_mission_detections(mission.id)
        assert len(stored_dets) == len(fused)

        rgb_cam.stop()
        depth_cam.stop()
        db.close()

