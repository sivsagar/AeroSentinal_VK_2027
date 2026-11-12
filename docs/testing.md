# AeroSentinel Testing & Verification Guide

## 1. Testing Philosophy
AeroSentinel adopts hardware-in-the-loop and software-in-the-loop test automation:
- All core logic, perception, fusion, and telemetry contracts are tested using mock drivers without requiring physical drone hardware.
- Real hardware imports are wrapped in fallback adapters with graceful degradation.
- Zero mock data is presented as real physical sensor measurements.

## 2. Test Execution Commands

```bash
# Run all automated tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src/aerosentinel --cov-report=term-missing

# Run simulation test only
python -m pytest tests/simulation/test_simulation.py -v

# Run unit tests per phase
python -m pytest tests/unit/test_config_loader.py -v  # Phase 1
python -m pytest tests/unit/test_logger.py -v         # Phase 1
python -m pytest tests/unit/test_performance.py -v    # Phase 1
python -m pytest tests/unit/test_cameras.py -v        # Phase 2
python -m pytest tests/unit/test_sensors.py -v        # Phase 2
python -m pytest tests/unit/test_perception.py -v     # Phase 3
python -m pytest tests/unit/test_mission.py -v        # Phase 4
python -m pytest tests/unit/test_dashboard.py -v      # Phase 5
python -m pytest tests/unit/test_mavlink.py -v        # Phase 6
```

## 3. Hardware Test Checklist (Before Flight)
1. Run `./scripts/check_hardware.sh` to verify CSI and USB 3.0 interfaces.
2. Confirm RealSense D435i returns depth packets at 15 FPS: `rs-enumerate-devices`.
3. Check Pi Camera Module 3: `libcamera-hello -t 5000`.
4. Inspect thermal baseline: ensure idle temp is below 50°C.
5. Verify MAVLink heartbeat reception over UART with baud 57600.

