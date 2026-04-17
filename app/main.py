"""
FastAPI application – RK3566 NPU Stress-Test Dashboard
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from npu_stress import MODEL_REGISTRY, NPUStressTest

app = FastAPI(title="RK3566 NPU Stress Test")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── single active test instance ───────────────────────────────────────────────
_test: NPUStressTest | None = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _status() -> dict[str, Any]:
    if _test is None:
        return {"is_running": False, "metrics": {}, "history": [], "result": None}
    return _test.get_status()


# ── API ───────────────────────────────────────────────────────────────────────

@app.post("/api/start")
async def api_start(
    duration: int = Query(default=60, ge=5, le=3600),
    test_type: str = Query(default="resnet18"),
):
    global _test
    if _test and _test.is_running:
        raise HTTPException(status_code=409, detail="A stress test is already running")
    if test_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown test_type '{test_type}'. Valid options: {list(MODEL_REGISTRY)}",
        )
    _test = NPUStressTest(duration=duration, test_type=test_type)
    _test.start()
    return {"status": "started", "duration": duration, "test_type": test_type}


@app.get("/api/test-types")
async def api_test_types():
    """Return the registry of available test types with their labels and descriptions."""
    return {
        key: {"label": val["label"], "description": val["description"]}
        for key, val in MODEL_REGISTRY.items()
    }


@app.post("/api/stop")
async def api_stop():
    global _test
    if _test is None or not _test.is_running:
        return {"status": "not_running"}
    # Run stop() in a thread so we don't block the event-loop during join()
    await asyncio.get_running_loop().run_in_executor(None, _test.stop)
    return {"status": "stopped"}


@app.get("/api/status")
async def api_status():
    return _status()


@app.get("/api/events")
async def api_events():
    """Server-Sent Events stream – pushes a JSON status payload every 500 ms."""

    async def generator():
        try:
            while True:
                payload = json.dumps(_status())
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text()
