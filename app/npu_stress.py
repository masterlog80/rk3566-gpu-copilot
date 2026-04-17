"""
NPU Stress Test worker for Rockchip RK3566 (Orange Pi CM4).

Primary mode  : uses the rknn-toolkit-lite2 Python package (rknnlite) to run
                repeated MobileNetV1 inference on all three NPU cores.
Fallback mode : simulates a realistic NPU workload so the UI can be exercised on
                any host (x86, CI, …) that does not have the RKNN runtime.
"""

from __future__ import annotations

import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# ── constants ────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/mobilenet_v1.rknn")

# sysfs paths for NPU utilisation (varies by BSP / kernel version)
_NPU_UTIL_PATHS = [
    "/sys/kernel/debug/rknpu/load",                          # debugfs (needs CAP_SYS_ADMIN)
    "/sys/devices/platform/fdab0000.npu/utilization",        # RK3566
    "/sys/devices/platform/fde40000.npu/utilization",        # RK3568
]

# sysfs thermal zones
_THERMAL_BASE = "/sys/class/thermal"


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_npu_utilisation() -> float | None:
    """Return NPU utilisation 0-100 or None if unavailable."""
    for path in _NPU_UTIL_PATHS:
        try:
            raw = open(path).read().strip()
            # possible formats:  "98"  |  "Core0: 98%, Core1: 97%, Core2: 99%"
            if "Core" in raw:
                nums = [int(t.rstrip("%,")) for t in raw.split() if t.rstrip("%,").isdigit()]
                if nums:
                    return sum(nums) / len(nums)
            else:
                return float(raw.split("%")[0].strip())
        except Exception:
            pass
    return None


def _read_temperature() -> float:
    """Return the highest temperature (°C) across all thermal zones."""
    best = 0.0
    try:
        zones = os.listdir(_THERMAL_BASE)
    except OSError:
        return best
    for z in zones:
        try:
            temp = int(open(f"{_THERMAL_BASE}/{z}/temp").read().strip()) / 1000.0
            if 20.0 <= temp <= 120.0:
                best = max(best, temp)
        except Exception:
            pass
    return best


# ── data class for one time-series sample ────────────────────────────────────

@dataclass
class Sample:
    t: float          # seconds since test start
    fps: float
    latency_ms: float
    temperature: float
    npu_util: float   # 0-100, may be 0 if unreadable


# ── main class ────────────────────────────────────────────────────────────────

# Maximum number of history samples sent to the client in a single status payload.
# Older samples are dropped; the chart down-samples further on the client side.
_MAX_HISTORY_SAMPLES = 120


class NPUStressTest:
    """Thread-safe NPU stress test controller."""

    def __init__(self, duration: int) -> None:
        self.duration = duration
        self.is_running = False
        self.stop_requested = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.metrics: dict[str, Any] = self._empty_metrics()
        self.history: list[dict] = []
        self.result: dict[str, Any] | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.stop_requested = False
        self.result = None
        self.history = []
        self.metrics = self._empty_metrics()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_requested = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8)
        self.is_running = False

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "is_running": self.is_running,
                "metrics": dict(self.metrics),
                "history": list(self.history[-_MAX_HISTORY_SAMPLES:]),
                "result": self.result,
            }

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "fps": 0.0,
            "latency_ms": 0.0,
            "temperature": 0.0,
            "npu_util": 0.0,
            "elapsed": 0.0,
            "progress": 0.0,
            "total_inferences": 0,
            "duration_s": 0,
            "mode": "idle",
        }

    def _run(self) -> None:
        try:
            self._run_rknn()
        except Exception as exc:
            print(
                f"[WARN] RKNN mode failed ({type(exc).__name__}: {exc}); "
                "falling back to simulation mode.",
                file=sys.stderr,
                flush=True,
            )
            self._run_simulated()
        finally:
            self.is_running = False

    # ── RKNN mode ─────────────────────────────────────────────────────────────

    def _run_rknn(self) -> None:
        from rknnlite.api import RKNNLite  # type: ignore[import-untyped]

        rknn = RKNNLite(verbose=False)
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        ret = rknn.load_rknn(MODEL_PATH)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        import numpy as np  # noqa: PLC0415

        input_data = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)
        start = time.perf_counter()
        count = 0
        latencies: list[float] = []

        try:
            while not self.stop_requested:
                elapsed = time.perf_counter() - start
                if elapsed >= self.duration:
                    break

                t0 = time.perf_counter()
                rknn.inference(inputs=[input_data])
                t1 = time.perf_counter()

                lat = (t1 - t0) * 1000
                latencies.append(lat)
                count += 1

                fps = count / elapsed if elapsed > 0 else 0.0
                mean_lat = sum(latencies[-200:]) / len(latencies[-200:])
                temp = _read_temperature()
                util = _read_npu_utilisation() or 0.0
                progress = min(100.0, elapsed / self.duration * 100)

                sample = {"t": round(elapsed, 2), "fps": round(fps, 1),
                          "latency_ms": round(mean_lat, 2),
                          "temperature": round(temp, 1), "npu_util": round(util, 1)}
                with self._lock:
                    self.metrics.update({
                        "fps": round(fps, 1),
                        "latency_ms": round(mean_lat, 2),
                        "temperature": round(temp, 1),
                        "npu_util": round(util, 1),
                        "elapsed": round(elapsed, 1),
                        "progress": round(progress, 1),
                        "total_inferences": count,
                        "duration_s": self.duration,
                        "mode": "rknn",
                    })
                    self.history.append(sample)
        finally:
            rknn.release()

        self._finalise()

    # ── Simulated mode ────────────────────────────────────────────────────────

    def _run_simulated(self) -> None:
        """Realistic simulation so the UI is fully exercisable without hardware."""
        start = time.perf_counter()
        count = 0

        # Baseline simulated values
        base_fps = 210.0 + random.uniform(-10, 10)
        base_lat = 1000.0 / base_fps
        base_temp = 45.0 + random.uniform(-2, 2)

        warmup_done = False

        while not self.stop_requested:
            elapsed = time.perf_counter() - start
            if elapsed >= self.duration:
                break

            # warm-up ramp during first 3 seconds
            ramp = min(1.0, elapsed / 3.0)
            jitter = random.gauss(0, 2)
            fps = base_fps * ramp + jitter
            lat = base_lat / max(ramp, 0.01) + random.gauss(0, 0.3)
            temp = base_temp + ramp * 18.0 + math.sin(elapsed * 0.2) * 2 + random.uniform(-0.5, 0.5)
            util = min(100.0, 70.0 * ramp + random.uniform(-3, 3))

            if ramp >= 1.0 and not warmup_done:
                warmup_done = True
                base_fps = fps  # lock in plateau

            progress = min(100.0, elapsed / self.duration * 100)
            # simulate ~240 FPS → 1 inference every ~4 ms
            inferences_this_tick = max(1, round(fps * 0.05))
            count += inferences_this_tick

            sample = {"t": round(elapsed, 2), "fps": round(fps, 1),
                      "latency_ms": round(lat, 2),
                      "temperature": round(temp, 1), "npu_util": round(util, 1)}
            with self._lock:
                self.metrics.update({
                    "fps": round(fps, 1),
                    "latency_ms": round(lat, 2),
                    "temperature": round(temp, 1),
                    "npu_util": round(util, 1),
                    "elapsed": round(elapsed, 1),
                    "progress": round(progress, 1),
                    "total_inferences": count,
                    "duration_s": self.duration,
                    "mode": "simulated",
                })
                self.history.append(sample)

            time.sleep(0.05)   # ~20 samples/s

        self._finalise()

    # ── finalise ──────────────────────────────────────────────────────────────

    def _finalise(self) -> None:
        with self._lock:
            h = self.history
            if h:
                avg_fps = sum(s["fps"] for s in h) / len(h)
                peak_temp = max(s["temperature"] for s in h)
                avg_lat = sum(s["latency_ms"] for s in h) / len(h)
                avg_util = sum(s["npu_util"] for s in h) / len(h)
            else:
                avg_fps = peak_temp = avg_lat = avg_util = 0.0

            self.result = {
                "avg_fps": round(avg_fps, 1),
                "peak_temp": round(peak_temp, 1),
                "avg_latency_ms": round(avg_lat, 2),
                "avg_npu_util": round(avg_util, 1),
                "total_inferences": self.metrics.get("total_inferences", 0),
                "duration_s": self.duration,
                "mode": self.metrics.get("mode", "unknown"),
            }
            self.metrics["progress"] = 100.0
