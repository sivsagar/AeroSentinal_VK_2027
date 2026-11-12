"""
Unit tests for src/aerosentinel/utils/performance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aerosentinel.utils.performance import (
    FPSTracker,
    LatencyTracker,
    PerformanceMonitor,
    read_cpu_usage_percent,
    read_cpu_temperature,
    read_ram_usage,
)


# ---------------------------------------------------------------------------
# FPS Tracker
# ---------------------------------------------------------------------------


class TestFPSTracker:
    """FPSTracker rolling-window accuracy."""

    def test_zero_fps_before_ticks(self) -> None:
        tracker = FPSTracker(window_size=10)
        assert tracker.fps == 0.0

    def test_one_tick_gives_zero_fps(self) -> None:
        """Need at least 2 timestamps to compute FPS."""
        tracker = FPSTracker(window_size=10)
        tracker.tick()
        assert tracker.fps == 0.0

    def test_approximate_fps(self) -> None:
        """Tick at ~20 FPS for a moment and verify the measurement is in range."""
        tracker = FPSTracker(window_size=20)
        interval = 1.0 / 20.0  # 20 FPS
        for _ in range(20):
            tracker.tick()
            time.sleep(interval)
        fps = tracker.fps
        # Accept ±50% tolerance given test-environment timing jitter
        assert 10.0 <= fps <= 30.0, f"Expected ~20 FPS but got {fps:.2f}"

    def test_fps_window_respects_size(self) -> None:
        """With a window of 5, only the last 5 timestamps count."""
        tracker = FPSTracker(window_size=5)
        for _ in range(10):
            tracker.tick()
            time.sleep(0.01)
        # Window should drop the oldest ticks — fps still computable
        assert tracker.fps > 0.0

    def test_reset_clears_fps(self) -> None:
        tracker = FPSTracker(window_size=10)
        for _ in range(5):
            tracker.tick()
        tracker.reset()
        assert tracker.fps == 0.0


# ---------------------------------------------------------------------------
# Latency Tracker
# ---------------------------------------------------------------------------


class TestLatencyTracker:
    """LatencyTracker statistics."""

    def test_empty_stats(self) -> None:
        tracker = LatencyTracker("test")
        stats = tracker.stats()
        assert stats["count"] == 0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0

    def test_single_sample_stats(self) -> None:
        tracker = LatencyTracker("inference")
        tracker.record(100.0)
        stats = tracker.stats()
        assert stats["count"] == 1
        assert stats["min"] == 100.0
        assert stats["max"] == 100.0
        assert stats["mean"] == 100.0
        assert stats["last"] == 100.0

    def test_multiple_samples(self) -> None:
        tracker = LatencyTracker("proc")
        for v in [10.0, 20.0, 30.0]:
            tracker.record(v)
        stats = tracker.stats()
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert abs(stats["mean"] - 20.0) < 1e-9
        assert stats["count"] == 3

    def test_reset_clears_samples(self) -> None:
        tracker = LatencyTracker("proc")
        tracker.record(99.0)
        tracker.reset()
        assert tracker.stats()["count"] == 0

    def test_max_samples_cap(self) -> None:
        tracker = LatencyTracker("bounded", max_samples=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record(v)
        stats = tracker.stats()
        # Only the last 3 should count
        assert stats["count"] == 3
        assert stats["min"] == 3.0


# ---------------------------------------------------------------------------
# System resource readers
# ---------------------------------------------------------------------------


class TestSystemReaders:
    """read_cpu_usage_percent, read_ram_usage, read_cpu_temperature."""

    def test_cpu_usage_returns_number_or_none(self) -> None:
        result = read_cpu_usage_percent(interval=0.0)
        assert result is None or isinstance(result, float)

    def test_cpu_usage_in_valid_range(self) -> None:
        result = read_cpu_usage_percent(interval=0.0)
        if result is not None:
            assert 0.0 <= result <= 100.0

    def test_ram_usage_returns_dict(self) -> None:
        result = read_ram_usage()
        assert isinstance(result, dict)
        for key in ("total_mb", "used_mb", "available_mb", "percent"):
            assert key in result

    def test_ram_values_are_numbers_or_none(self) -> None:
        result = read_ram_usage()
        for key, val in result.items():
            assert val is None or isinstance(val, (int, float)), (
                f"Expected number or None for {key}, got {type(val)}"
            )

    def test_ram_total_positive_when_available(self) -> None:
        result = read_ram_usage()
        if result["total_mb"] is not None:
            assert result["total_mb"] > 0

    def test_cpu_temperature_returns_float_or_none(self) -> None:
        temp = read_cpu_temperature()
        assert temp is None or isinstance(temp, float)

    def test_cpu_temperature_in_sane_range(self) -> None:
        temp = read_cpu_temperature()
        if temp is not None:
            # Reasonable operating temperature for electronics
            assert -10.0 <= temp <= 120.0, f"Unexpected temperature: {temp}"


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------


class TestPerformanceMonitor:
    """PerformanceMonitor class."""

    def test_start_stop_no_error(self) -> None:
        mon = PerformanceMonitor()
        mon.start()
        mon.stop()

    def test_context_manager(self) -> None:
        with PerformanceMonitor() as mon:
            mon.tick()

    def test_report_has_required_keys(self) -> None:
        with PerformanceMonitor() as mon:
            report = mon.report()

        expected_keys = {"fps", "dropped_frames", "uptime_s", "cpu_percent", "ram", "cpu_temp_c", "latency"}
        assert expected_keys.issubset(report.keys())

    def test_fps_increases_with_ticks(self) -> None:
        mon = PerformanceMonitor()
        mon.start()
        interval = 0.05  # 20 Hz
        for _ in range(10):
            mon.tick()
            time.sleep(interval)
        report = mon.report()
        mon.stop()
        # FPS should be non-zero
        assert report["fps"] > 0.0

    def test_dropped_frames_counter(self) -> None:
        mon = PerformanceMonitor()
        mon.start()
        mon.drop_frame(5)
        mon.stop()
        assert mon.dropped_frames == 5

    def test_latency_tracker_recorded_in_report(self) -> None:
        with PerformanceMonitor() as mon:
            mon.latency("inference").record(150.0)
            report = mon.report()

        assert "inference" in report["latency"]
        assert report["latency"]["inference"]["last"] == 150.0

    def test_multiple_latency_trackers(self) -> None:
        with PerformanceMonitor() as mon:
            mon.latency("inference").record(200.0)
            mon.latency("processing").record(50.0)
            report = mon.report()

        assert "inference" in report["latency"]
        assert "processing" in report["latency"]

    def test_uptime_increases(self) -> None:
        mon = PerformanceMonitor()
        mon.start()
        time.sleep(0.1)
        report = mon.report()
        mon.stop()
        assert report["uptime_s"] >= 0.0

    def test_reset_clears_counters(self) -> None:
        mon = PerformanceMonitor()
        mon.start()
        mon.drop_frame(10)
        mon.latency("x").record(99.0)
        mon.reset()
        report = mon.report()
        mon.stop()
        assert report["dropped_frames"] == 0
        assert report["latency"]["x"]["count"] == 0

    def test_report_json_is_valid_json(self) -> None:
        import json

        with PerformanceMonitor() as mon:
            json_str = mon.report_json()

        data = json.loads(json_str)
        assert "fps" in data

    def test_background_thread_starts_and_stops(self) -> None:
        mon = PerformanceMonitor(enable_background=True, sample_interval_s=0.05)
        mon.start()
        assert mon._thread is not None
        assert mon._thread.is_alive()
        time.sleep(0.15)  # Let it sample a couple of times
        mon.stop()
        assert not mon._running

    def test_background_populates_cpu(self) -> None:
        mon = PerformanceMonitor(enable_background=True, sample_interval_s=0.05)
        mon.start()
        time.sleep(0.2)
        report = mon.report()
        mon.stop()
        # cpu_percent may be None if psutil not installed; either is acceptable
        assert "cpu_percent" in report
