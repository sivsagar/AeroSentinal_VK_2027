#!/usr/bin/env python3
"""
scripts/multi_person_tracking_test.py
======================================
AeroSentinel — 60-second multi-person tracking diagnostic script.

Runs a full 60-second simulated or hardware-assisted tracking session and
produces a comprehensive JSON report covering:
  - Actual test duration
  - Total tracker updates
  - Total YOLO inference passes
  - Total person detections
  - Total tracks created / confirmed / lost / reacquired / expired
  - Peak concurrent tracks
  - Per-track age, hits, duration, depth metadata history
  - RGB-D distance validity statistics
  - Invalid and stale depth measurement counts
  - Camera FPS and dropped-frame information

Usage (simulation mode — no hardware required):
    python scripts/multi_person_tracking_test.py --mock --duration 60

Usage (live hardware — Pi 4B with D435i):
    python scripts/multi_person_tracking_test.py --duration 60

Output:
    data/reports/tracking_diagnostic_<timestamp>.json

SAFETY NOTICE
-------------
This script is read-only with respect to flight control.
No MAVLink arming, takeoff, motor, PWM or navigation commands are issued.

DISCLAIMER
----------
Test results obtained in simulation/mock mode do NOT validate real-hardware
tracking reliability. Physical tracking performance depends on actual sensor
data quality, lighting conditions, UAV motion and camera calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure src/ and simulation/ are importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "simulation"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tracking_diagnostic")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AeroSentinel 60-second multi-person tracking diagnostic"
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Run in simulation/mock mode (no real hardware required).",
    )
    p.add_argument(
        "--duration", type=float, default=60.0,
        help="Test duration in seconds (default: 60).",
    )
    p.add_argument(
        "--fps", type=float, default=10.0,
        help="Simulated or target frame rate (default: 10 Hz).",
    )
    p.add_argument(
        "--output-dir", default="data/reports",
        help="Directory for the JSON report (default: data/reports).",
    )
    p.add_argument(
        "--config", default="config/",
        help="Config directory (default: config/).",
    )
    p.add_argument(
        "--min-hits", type=int, default=3,
        help="Min hits to confirm a track (default: 3).",
    )
    p.add_argument(
        "--max-missed", type=int, default=15,
        help="Max missed frames before LOST (default: 15).",
    )
    p.add_argument(
        "--ema-alpha", type=float, default=0.7,
        help="EMA alpha for distance smoothing (default: 0.7).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _synthetic_detections(frame_idx: int) -> List[Any]:
    """Return synthetic Detection objects simulating 1–2 persons."""
    try:
        from aerosentinel.perception.inference_engine import Detection
    except ImportError:
        return []

    now = time.time()
    dets = []

    # Person A — always visible, drifts slowly
    cx_a = 320 + int(10 * (frame_idx % 30 - 15))
    cy_a = 240
    dets.append(Detection(
        class_id=0, class_name="person", confidence=0.91,
        bbox=(cx_a - 50, cy_a - 70, cx_a + 50, cy_a + 70),
        timestamp=now,
    ))

    # Person B — appears at frame 50, leaves at frame 200, returns at frame 350
    total = frame_idx
    if 50 <= (total % 500) < 200 or (total % 500) >= 350:
        cx_b = 160 + int(5 * (frame_idx % 20 - 10))
        cy_b = 240
        dets.append(Detection(
            class_id=0, class_name="person", confidence=0.85,
            bbox=(cx_b - 40, cy_b - 60, cx_b + 40, cy_b + 60),
            timestamp=now,
        ))

    return dets


def _synthetic_depth(frame_idx: int, h: int = 480, w: int = 640):
    """Return a synthetic depth array (metres as float32)."""
    import numpy as np
    # Sensor dropout at frames 100–130
    if 100 <= (frame_idx % 500) < 130:
        return np.zeros((h, w), dtype=np.float32)
    base_dist = 4.0 + 0.001 * frame_idx
    return np.full((h, w), base_dist, dtype=np.float32)


# ---------------------------------------------------------------------------
# Hardware camera acquisition (optional)
# ---------------------------------------------------------------------------

def _acquire_hardware_frame(camera_mgr: Any) -> tuple:
    """
    Try to get (rgb_frame, depth_m, imu_data) from the hardware camera manager.
    Returns (None, None, None) on any error.
    """
    try:
        rgb_result = camera_mgr.get_rgb_frame()
        depth_result = camera_mgr.get_depth_frame()

        rgb_frame = rgb_result[0] if rgb_result else None
        depth_mm = depth_result[0] if depth_result else None

        import numpy as np
        depth_m = depth_mm.astype(np.float32) * 0.001 if depth_mm is not None else None

        imu = camera_mgr.get_imu_data()
        return rgb_frame, depth_m, imu
    except Exception as exc:
        logger.debug("Hardware frame acquisition error: %s", exc)
        return None, None, None


# ---------------------------------------------------------------------------
# Main diagnostic runner
# ---------------------------------------------------------------------------

def run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the tracking diagnostic and return the full report dict."""

    logger.info(
        "AeroSentinel tracking diagnostic | mock=%s | duration=%.0fs | fps=%.1f",
        args.mock, args.duration, args.fps,
    )

    # ----------------------------------------------------------------
    # Import core modules
    # ----------------------------------------------------------------
    try:
        from aerosentinel.perception.person_tracker import PersonTracker
        from aerosentinel.perception.rgbd_fusion import RGBDFusion
    except ImportError as e:
        logger.error("Cannot import AeroSentinel modules: %s", e)
        logger.error("Run from the repository root with: python scripts/multi_person_tracking_test.py")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Initialise tracker and fusion
    # ----------------------------------------------------------------
    tracker = PersonTracker(
        iou_threshold=0.3,
        min_hits=args.min_hits,
        max_missed=args.max_missed,
        max_expired_age_s=30.0,
        ema_alpha=args.ema_alpha,
    )
    fusion = RGBDFusion(
        min_valid_depth_m=0.1,
        max_valid_depth_m=10.0,
        min_valid_depth_pct=5.0,
    )

    # ----------------------------------------------------------------
    # Optionally initialise YOLO detector (mock or real)
    # ----------------------------------------------------------------
    detector = None
    detector_mode = "none"
    if not args.mock:
        try:
            from aerosentinel.perception.detector import PersonDetector
            detector = PersonDetector()
            detector_mode = "yolo_onnx"
            logger.info("YOLO detector initialised")
        except Exception as exc:
            logger.warning("YOLO detector unavailable (%s); using synthetic detections.", exc)
            detector_mode = "synthetic_fallback"
    else:
        detector_mode = "synthetic"

    # ----------------------------------------------------------------
    # Optionally initialise cameras (live mode)
    # ----------------------------------------------------------------
    camera_mgr = None
    if not args.mock:
        try:
            from aerosentinel.cameras.camera_manager import CameraManager
            camera_mgr = CameraManager(mock=False)
            camera_mgr.start()
            logger.info("CameraManager started (live hardware)")
        except Exception as exc:
            logger.warning("CameraManager unavailable (%s); using synthetic data.", exc)
            camera_mgr = None

    # ----------------------------------------------------------------
    # Session accounting
    # ----------------------------------------------------------------
    total_frames = 0
    total_inference_calls = 0
    total_person_detections = 0
    dropped_frames = 0
    frame_times: List[float] = []

    frame_period = 1.0 / args.fps
    session_start = time.monotonic()
    t_wall_start = time.time()

    logger.info("Starting %s-second tracking session…", args.duration)

    # ----------------------------------------------------------------
    # Main acquisition loop
    # ----------------------------------------------------------------
    while (time.monotonic() - session_start) < args.duration:
        t_frame_start = time.monotonic()
        frame_idx = total_frames
        t_wall = t_wall_start + (time.monotonic() - session_start)

        detections = []

        # --- Acquire frame and run detection ---
        if args.mock or camera_mgr is None:
            # Synthetic detection (no actual YOLO inference)
            detections = _synthetic_detections(frame_idx)
            depth_m = _synthetic_depth(frame_idx)
        else:
            rgb_frame, depth_m, _imu = _acquire_hardware_frame(camera_mgr)

            if rgb_frame is not None and detector is not None:
                try:
                    detections = detector.detect(rgb_frame)
                    total_inference_calls += 1
                except Exception as exc:
                    logger.debug("Inference error: %s", exc)
                    detections = []
            elif rgb_frame is None:
                dropped_frames += 1
                logger.debug("Dropped frame %d", frame_idx)

            if depth_m is None:
                import numpy as np
                depth_m = np.zeros((480, 640), dtype=np.float32)

        total_person_detections += len(detections)

        # --- RGB-D Fusion ---
        fused = fusion.fuse(detections, depth_m) if depth_m is not None else []

        # --- Tracker update ---
        tracker.update(fused, ts=t_wall)

        total_frames += 1
        elapsed = time.monotonic() - t_frame_start
        frame_times.append(elapsed)

        sleep_t = frame_period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            dropped_frames += 1

    actual_duration_s = time.monotonic() - session_start

    # ----------------------------------------------------------------
    # Shut down hardware
    # ----------------------------------------------------------------
    if camera_mgr is not None:
        try:
            camera_mgr.stop()
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Compute FPS statistics
    # ----------------------------------------------------------------
    achieved_fps = total_frames / actual_duration_s if actual_duration_s > 0 else 0.0
    avg_frame_ms = (sum(frame_times) / len(frame_times) * 1000.0) if frame_times else 0.0

    # ----------------------------------------------------------------
    # Build report
    # ----------------------------------------------------------------
    summary = tracker.lifecycle_summary()

    # Classify tracks for readability
    all_tracks = tracker.all_tracks()
    invalid_depth_track_frames = sum(
        sum(1 for _, evt, _ in t.lifecycle
            if evt.value == "depth_invalid")
        for t in all_tracks
    )
    valid_depth_track_frames = sum(
        sum(1 for _, evt, _ in t.lifecycle
            if evt.value == "depth_valid")
        for t in all_tracks
    )

    report: Dict[str, Any] = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "team_id": "VK26-AU-PAI-0399",
        "mode": "simulation" if (args.mock or camera_mgr is None) else "hardware",
        "detector_mode": detector_mode,
        "disclaimer": (
            "Software simulation results do NOT validate real-hardware tracking "
            "reliability. Physical performance depends on sensor quality, lighting, "
            "UAV motion and camera calibration."
        ),

        # Session timing
        "target_duration_s": args.duration,
        "actual_duration_s": round(actual_duration_s, 3),
        "target_fps": args.fps,
        "achieved_fps": round(achieved_fps, 2),
        "avg_frame_processing_ms": round(avg_frame_ms, 2),

        # Frame accounting
        "total_frames_processed": total_frames,
        "total_yolo_inference_calls": total_inference_calls,
        "total_person_detections": total_person_detections,
        "dropped_frames": dropped_frames,

        # Tracker configuration
        "tracker_config": {
            "iou_threshold": 0.3,
            "min_hits": args.min_hits,
            "max_missed": args.max_missed,
            "max_expired_age_s": 30.0,
            "ema_alpha": args.ema_alpha,
            "min_valid_depth_pct": 5.0,
        },

        # Lifecycle counts
        "total_tracks_created": summary["total_tracks_created"],
        "total_tracks_confirmed": summary["total_tracks_confirmed"],
        "total_tracks_lost": summary["total_tracks_lost"],
        "total_tracks_reacquired": summary["total_tracks_reacquired"],
        "total_tracks_expired": summary["total_tracks_expired"],
        "peak_concurrent_tracks": summary["peak_concurrent_tracks"],
        "active_track_count": summary["active_track_count"],
        "confirmed_track_count": summary["confirmed_track_count"],
        "lost_track_count": summary["lost_track_count"],
        "expired_track_count": summary["expired_track_count"],

        # Depth validity
        "total_depth_valid_detections": summary["total_depth_valid_detections"],
        "total_depth_invalid_detections": summary["total_depth_invalid_detections"],
        "total_track_depth_valid_frames": valid_depth_track_frames,
        "total_track_depth_invalid_frames": invalid_depth_track_frames,

        # Per-track detail
        "tracks": {
            tid: {
                "track_id": tdata["track_id"],
                "state": tdata["state"],
                "first_seen_ts": tdata["first_seen_ts"],
                "last_seen_ts": tdata["last_seen_ts"],
                "age_seconds": tdata["age_seconds"],
                "age_frames": tdata["age_frames"],
                "hit_count": tdata["hit_count"],
                "missed_frames": tdata["missed_frames"],
                "total_missed": tdata["total_missed"],
                "estimated_distance_m": tdata["estimated_distance_m"],
                "last_known_distance_m": tdata["last_known_distance_m"],
                "distance_valid": tdata["distance_valid"],
                "valid_depth_percent": tdata["valid_depth_percent"],
                "depth_method": tdata["depth_method"],
                "lifecycle_events": tdata["lifecycle_events"],
            }
            for tid, tdata in summary["tracks"].items()
        },
    }

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    report = run_diagnostic(args)

    # Save JSON report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"tracking_diagnostic_{ts_str}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Report saved → %s", out_path)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("AeroSentinel — Tracking Diagnostic Report")
    print("=" * 70)
    print(f"Mode              : {report['mode']} ({report['detector_mode']})")
    print(f"Duration          : {report['actual_duration_s']:.1f}s / {report['target_duration_s']:.0f}s target")
    print(f"Frames processed  : {report['total_frames_processed']}")
    print(f"Achieved FPS      : {report['achieved_fps']:.1f} Hz")
    print(f"Dropped frames    : {report['dropped_frames']}")
    print(f"YOLO inferences   : {report['total_yolo_inference_calls']}")
    print(f"Person detections : {report['total_person_detections']}")
    print()
    print(f"Tracks created    : {report['total_tracks_created']}")
    print(f"Tracks confirmed  : {report['total_tracks_confirmed']}")
    print(f"Tracks lost       : {report['total_tracks_lost']}")
    print(f"Tracks reacquired : {report['total_tracks_reacquired']}")
    print(f"Tracks expired    : {report['total_tracks_expired']}")
    print(f"Peak concurrent   : {report['peak_concurrent_tracks']}")
    print(f"Currently active  : {report['active_track_count']}")
    print()
    print(f"Depth valid dets  : {report['total_depth_valid_detections']}")
    print(f"Depth invalid dets: {report['total_depth_invalid_detections']}")
    print()
    print("Per-track summary:")
    for tid, tdata in report["tracks"].items():
        dist_str = (
            f"{tdata['estimated_distance_m']:.2f}m"
            if tdata["estimated_distance_m"] is not None
            else f"N/A (last: {tdata['last_known_distance_m']})"
        )
        print(
            f"  Track {tid:>3}: state={tdata['state']:<10} "
            f"hits={tdata['hit_count']:>4} age_frames={tdata['age_frames']:>4} "
            f"age={tdata['age_seconds']:.1f}s dist={dist_str} "
            f"depth_valid={tdata['distance_valid']}"
        )
    print()
    print(f"Full report → {out_path}")
    print("=" * 70)
    print(
        "\nNOTE: Simulation results do not confirm real-hardware tracking "
        "reliability.\nRun on physical Raspberry Pi 4B + D435i hardware for "
        "real validation."
    )

    # Raspberry Pi execution commands
    print("\n" + "=" * 70)
    print("Raspberry Pi execution commands:")
    print("  cd ~/AeroSentinal_VK_2027")
    print("  # Mock / simulation (no hardware):")
    print("  python3 scripts/multi_person_tracking_test.py --mock --duration 60")
    print("  # Live hardware (Pi Camera 3 + D435i):")
    print("  python3 scripts/multi_person_tracking_test.py --duration 60")
    print("=" * 70)


if __name__ == "__main__":
    main()

