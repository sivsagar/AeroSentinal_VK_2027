"""
tests/unit/test_hardware_init.py

Regression tests for Phase 2:
- Correct loading of AI model configuration
- ONNX engine loading before inference
- Mock engine loading in mock mode
- Missing ONNX model handling
- No silent mock fallback in hardware mode
- Both camera configurations being passed to CameraManager
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure src/, simulation/, and repo root are importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_SIM_DIR = _REPO_ROOT / "simulation"
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from aerosentinel.utils.config_loader import ConfigLoader
from aerosentinel.perception.inference_engine import ONNXInferenceEngine, MockInferenceEngine
from aerosentinel.perception.detector import PersonDetector
from aerosentinel.cameras.camera_manager import CameraManager
from aerosentinel.main import _init_cameras, _init_perception


@pytest.fixture
def config_loader() -> ConfigLoader:
    cfg_dir = _REPO_ROOT / "config"
    loader = ConfigLoader(config_dir=str(cfg_dir))
    loader.load("ai.yaml", "camera.yaml")
    return loader


class TestAIConfiguration:
    def test_ai_configuration_keys_loaded(self, config_loader: ConfigLoader):
        """Verify that ConfigLoader reads detection.* keys from config/ai.yaml."""
        model_path = config_loader.get("detection.model_path")
        conf_threshold = config_loader.get("detection.confidence_threshold")
        input_resolution = config_loader.get("detection.input_resolution")
        fps_target = config_loader.get("detection.inference_fps_target")

        assert model_path == "models/yolov8n.onnx"
        assert conf_threshold == 0.45
        assert input_resolution == [320, 320]
        assert fps_target == 3


class TestInferenceEngineLifecycle:
    def test_onnx_engine_infer_before_load_raises(self):
        """Calling infer() before load() on ONNXInferenceEngine must raise RuntimeError."""
        engine = ONNXInferenceEngine(model_path="models/yolov8n.onnx")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Engine not loaded"):
            engine.infer(dummy_frame)

    def test_onnx_engine_load_missing_file_raises_file_not_found(self):
        """Calling load() on ONNXInferenceEngine with missing file must raise FileNotFoundError."""
        engine = ONNXInferenceEngine(model_path="nonexistent_model_file.onnx")
        with pytest.raises(FileNotFoundError):
            engine.load()

    def test_mock_engine_infer_before_load_raises(self):
        """Calling infer() before load() on MockInferenceEngine must raise RuntimeError."""
        engine = MockInferenceEngine()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Call load\\(\\) before infer\\(\\)"):
            engine.infer(dummy_frame)

    def test_mock_engine_load_in_mock_mode(self, config_loader: ConfigLoader):
        """Mock mode in _init_perception must load MockInferenceEngine and execute inference cleanly."""
        detector = _init_perception(config_loader, mock=True)
        assert detector is not None
        assert isinstance(detector.engine, MockInferenceEngine)
        assert detector.engine._loaded is True

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, detections = detector.detect(dummy_frame)
        assert annotated is not None
        assert isinstance(detections, list)
        detector.stop()


class TestHardwareModeStrictness:
    def test_hardware_mode_missing_model_raises_file_not_found(self, config_loader: ConfigLoader):
        """Hardware mode (mock=False) must raise FileNotFoundError if ONNX model is missing."""
        config_loader.set("detection.model_path", "models/nonexistent_yolov8.onnx")

        with pytest.raises(FileNotFoundError) as exc_info:
            _init_perception(config_loader, mock=False)

        assert "ONNX model file not found" in str(exc_info.value)

    def test_hardware_mode_never_silently_substitutes_mock(self, config_loader: ConfigLoader):
        """Hardware mode must NEVER return a MockInferenceEngine when model is missing or invalid."""
        config_loader.set("detection.model_path", "missing_yolov8.onnx")
        with pytest.raises(FileNotFoundError):
            detector = _init_perception(config_loader, mock=False)


class TestCameraManagerConfiguration:
    def test_camera_manager_receives_both_rgb_and_depth_configs(self, config_loader: ConfigLoader):
        """_init_cameras must supply both rgb_camera and depth_camera configs to CameraManager."""
        mgr = _init_cameras(config_loader, mock=True)
        assert mgr is not None

        # Verify both sections are present in CameraManager config
        assert "rgb_camera" in mgr._config
        assert "depth_camera" in mgr._config

        rgb_cfg = mgr._config["rgb_camera"]
        depth_cfg = mgr._config["depth_camera"]

        assert rgb_cfg.get("enabled") is True
        assert rgb_cfg.get("resolution") == [1280, 720]
        assert depth_cfg.get("enabled") is True
        assert depth_cfg.get("depth_resolution") == [640, 480]

        # Verify drivers are instantiated
        status = mgr.status()
        assert "rgb" in status
        assert "depth" in status
        assert status["rgb"]["enabled"] is True
        assert status["depth"]["enabled"] is True

        mgr.stop_all()
