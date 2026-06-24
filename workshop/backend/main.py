"""FastAPI app for the Nuclear Central Manager panel.

Routes:
  * ``GET  /api/state``   — which solution (student/reference) is active per part.
  * ``WS   /ws/run-day``  — stream the whole day as StepEvent JSON, one beat at a time.
  * ``GET  /`` (+ assets) — the built React front end (``frontend/dist``).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .engine import DayEngine
from .loader import solution_source

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIST = os.path.abspath(os.path.join(_HERE, "..", "frontend", "dist"))

PART_TITLES = {1: "Coolant Pump", 2: "Basin Temperature", 3: "Waste Machines"}

app = FastAPI(title="Nuclear Central Manager")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types so json.dumps never chokes."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


@app.get("/api/state")
def api_state() -> JSONResponse:
    parts = [{"part": p, "title": PART_TITLES[p], "source": solution_source(p)}
             for p in (1, 2, 3)]
    return JSONResponse({"parts": parts})


@app.websocket("/ws/run-day")
async def run_day(ws: WebSocket) -> None:
    await ws.accept()
    try:
        pace = float(ws.query_params.get("pace", "1.1"))
    except (TypeError, ValueError):
        pace = 0.7
    engine = DayEngine(pace=pace)
    try:
        async for ev in engine.run():
            await ws.send_json(_jsonable(ev.to_dict()))
        await ws.send_json({"part": 0, "step_id": "stream.end", "status": "info",
                            "title": "", "message": "", "payload": {}})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001 — surface engine errors to the client
        await ws.send_json({"part": 0, "step_id": "stream.error", "status": "exploded",
                            "title": "Engine error", "message": str(exc), "payload": {}})


# ---- serve the built front end (if it has been built) ---------------------- #
if os.path.isdir(_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    def _no_build() -> str:
        return (
            "<h1>Nuclear Central Manager — backend is up ☢️</h1>"
            "<p>The front end isn't built yet. From <code>workshop/frontend</code> run "
            "<code>npm install &amp;&amp; npm run build</code>, then reload.</p>"
            "<p>The API is live at <code>/api/state</code> and <code>/ws/run-day</code>.</p>"
        )
