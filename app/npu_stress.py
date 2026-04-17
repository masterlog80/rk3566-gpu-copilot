"""
NPU Stress Test worker for Rockchip RK3566 (Orange Pi CM4).

Primary mode  : uses the rknn-toolkit-lite2 Python package (rknnlite) to run
                repeated inference (model selectable) on all three NPU cores.
Fallback mode : simulates a realistic NPU workload so the UI can be exercised on
                any host (x86, CI, …) that does not have the RKNN runtime.
"""

from __future__ import annotations

import math
import os
import re
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# ── constants ────────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
# Legacy single-model env var kept for backward-compatibility
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/resnet18_for_rk3566_rk3568.rknn")

# ── model registry ───────────────────────────────────────────────────────────
# Maps a test_type key to its model filename, input shape, human label,
# description shown in the UI, and baseline simulated FPS.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "resnet18": {
        "filename": "resnet18_for_rk3566_rk3568.rknn",
        "input_shape": (1, 224, 224, 3),
        "label": "ResNet18",
        "description": "ResNet18 – balanced baseline, exercises all NPU cores (~75 FPS)",
        "sim_fps": 75.0,
    },
    "mobilenet_v1": {
        "filename": "mobilenet_v1_for_rk3566_rk3568.rknn",
        "input_shape": (1, 224, 224, 3),
        "label": "MobileNetV1",
        "description": "MobileNetV1 – lightweight depthwise-separable convolutions, high-throughput test (~200 FPS)",
        "sim_fps": 200.0,
    },
    "mobilenet_v2": {
        "filename": "mobilenet_v2_for_rk3566_rk3568.rknn",
        "input_shape": (1, 224, 224, 3),
        "label": "MobileNetV2",
        "description": "MobileNetV2 – inverted residuals + linear bottlenecks, efficiency test (~160 FPS)",
        "sim_fps": 160.0,
    },
    "multi_thread": {
        "filename": "resnet18_for_rk3566_rk3568.rknn",
        "input_shape": (1, 224, 224, 3),
        "label": "Multi-thread (3×ResNet18)",
        "description": "Multi-thread – 3 concurrent ResNet18 inference threads, stresses all three NPU cores (~210 FPS aggregate)",
        "sim_fps": 210.0,
    },
}

_NUM_MULTITHREAD_WORKERS = 3
_WORKER_JOIN_TIMEOUT_S = 5

# sysfs paths for NPU utilisation (varies by BSP / kernel version)
_NPU_UTIL_PATHS = [
    "/sys/kernel/debug/rknpu/load",                          # debugfs (needs CAP_SYS_ADMIN)
    "/sys/devices/platform/fdab0000.npu/utilization",        # RK3566 sysfs
    "/sys/devices/platform/fde40000.npu/utilization",        # RK3568 sysfs
    # devfreq load – format "<busy_ns>@<total_ns>Hz" or "<busy_us>@<total_us>"
    "/sys/class/devfreq/fdab0000.npu/load",                  # RK3566 devfreq
    "/sys/class/devfreq/fde40000.npu/load",                  # RK3568 devfreq
    "/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/load",
    "/sys/devices/platform/fde40000.npu/devfreq/fde40000.npu/load",
]

# Emit the "all paths failed" warning at most once so it doesn't spam the log.
_npu_util_warned = False

# sysfs thermal zones
_THERMAL_BASE = "/sys/class/thermal"


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_npu_utilisation() -> float | None:
    """Return NPU utilisation 0-100 or None if unavailable."""
    global _npu_util_warned
    for path in _NPU_UTIL_PATHS:
        try:
            raw = open(path).read().strip()
            # format 1: "NPU load:  Core0:  98%, Core1:  97%, Core2:  99%"
            if "Core" in raw:
                nums = [int(t.rstrip("%,")) for t in raw.split() if t.rstrip("%,").isdigit()]
                if nums:
                    return sum(nums) / len(nums)
            # format 2: devfreq "<busy>@<total>" e.g. "123456789@200000000Hz"
            elif "@" in raw:
                parts = raw.split("@")
                busy = int(parts[0].strip())
                total_str = parts[1].strip().rstrip("Hz").strip()
                total = int(total_str)
                if total > 0:
                    return min(100.0, busy / total * 100.0)
            # format 3: "NPU load:  0%", plain "98%", or bare "98" / "98.5"
            else:
                m = re.search(r"(\d+(?:\.\d+)?)\s*%", raw)
                if m:
                    return float(m.group(1))
                return float(raw.strip())
        except Exception:
            pass
    if not _npu_util_warned:
        _npu_util_warned = True
        print(
            "[WARN] NPU utilisation: none of the known sysfs/debugfs paths are readable. "
            "Ensure the container is started with --privileged and debugfs is mounted "
            "(docker-compose.yml already does this via the /sys/kernel/debug bind-mount). "
            "NPU utilisation will be reported as 0%.",
            file=sys.stderr,
            flush=True,
        )
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

    def __init__(self, duration: int, test_type: str = "resnet18") -> None:
        self.duration = duration
        if test_type not in MODEL_REGISTRY:
            print(
                f"[WARN] Unknown test_type '{test_type}', falling back to 'resnet18'.",
                file=sys.stderr,
                flush=True,
            )
            test_type = "resnet18"
        self.test_type = test_type
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
            "test_type": "resnet18",
        }

    def _run(self) -> None:
        try:
            if self.test_type == "multi_thread":
                self._run_rknn_multithread()
            else:
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

    # ── RKNN mode (single-thread) ─────────────────────────────────────────────

    def _run_rknn(self) -> None:
        from rknnlite.api import RKNNLite  # type: ignore[import-untyped]
        import numpy as np  # noqa: PLC0415

        cfg = MODEL_REGISTRY[self.test_type]
        model_path = os.path.join(MODEL_DIR, cfg["filename"])
        input_shape: tuple[int, ...] = cfg["input_shape"]

        rknn = RKNNLite(verbose=False)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        input_data = np.random.randint(0, 256, input_shape, dtype=np.uint8)
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
                        "test_type": self.test_type,
                    })
                    self.history.append(sample)
        finally:
            rknn.release()

        self._finalise()

    # ── RKNN mode (multi-thread) ──────────────────────────────────────────────

    def _run_rknn_multithread(self) -> None:
        """Spawn _NUM_MULTITHREAD_WORKERS inference threads, each with its own
        RKNNLite instance, and aggregate their results into shared metrics."""
        from rknnlite.api import RKNNLite  # type: ignore[import-untyped]
        import numpy as np  # noqa: PLC0415

        cfg = MODEL_REGISTRY[self.test_type]
        model_path = os.path.join(MODEL_DIR, cfg["filename"])
        input_shape: tuple[int, ...] = cfg["input_shape"]

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Initialise one RKNNLite instance per worker thread
        instances: list[Any] = []
        try:
            for _ in range(_NUM_MULTITHREAD_WORKERS):
                rknn = RKNNLite(verbose=False)
                ret = rknn.load_rknn(model_path)
                if ret != 0:
                    raise RuntimeError(f"load_rknn failed: {ret}")
                ret = rknn.init_runtime()
                if ret != 0:
                    raise RuntimeError(f"init_runtime failed: {ret}")
                instances.append(rknn)
        except Exception:
            for inst in instances:
                inst.release()
            raise

        # Shared aggregates written by workers, read by the metrics loop below
        _agg_lock = threading.Lock()
        _agg: dict[str, Any] = {"count": 0, "latencies": []}
        _stop_workers = threading.Event()

        start = time.perf_counter()

        def _worker(rknn_inst: Any) -> None:
            local_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
            while not _stop_workers.is_set() and not self.stop_requested:
                if time.perf_counter() - start >= self.duration:
                    break
                t0 = time.perf_counter()
                rknn_inst.inference(inputs=[local_input])
                t1 = time.perf_counter()
                lat = (t1 - t0) * 1000
                with _agg_lock:
                    _agg["count"] += 1
                    _agg["latencies"].append(lat)

        workers = [
            threading.Thread(target=_worker, args=(inst,), daemon=True)
            for inst in instances
        ]
        for w in workers:
            w.start()

        try:
            while not self.stop_requested:
                elapsed = time.perf_counter() - start
                if elapsed >= self.duration:
                    break

                with _agg_lock:
                    count = _agg["count"]
                    recent = _agg["latencies"][-600:]
                    mean_lat = sum(recent) / len(recent) if recent else 0.0

                fps = count / elapsed if elapsed > 0 else 0.0
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
                        "test_type": self.test_type,
                    })
                    self.history.append(sample)

                time.sleep(0.05)
        finally:
            _stop_workers.set()
            for w in workers:
                w.join(timeout=_WORKER_JOIN_TIMEOUT_S)
            for inst in instances:
                inst.release()

        self._finalise()

    # ── Simulated mode ────────────────────────────────────────────────────────

    def _run_simulated(self) -> None:
        """Realistic simulation so the UI is fully exercisable without hardware."""
        start = time.perf_counter()
        count = 0

        # Baseline simulated values come from the model registry
        cfg = MODEL_REGISTRY.get(self.test_type, MODEL_REGISTRY["resnet18"])
        base_fps = cfg["sim_fps"] + random.uniform(-10, 10)
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
            # simulate inferences proportional to FPS at ~20 samples/s
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
                    "test_type": self.test_type,
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
                "test_type": self.metrics.get("test_type", self.test_type),
            }
            self.metrics["progress"] = 100.0
