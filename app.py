"""FastAPI server exposing Cortex on localhost.

Endpoints:
  GET  /                       -> dashboard (static HTML)
  POST /api/runs               -> start a new research run (async)
  GET  /api/runs               -> list runs
  GET  /api/runs/{id}          -> run record
  GET  /api/runs/{id}/experiments
  GET  /api/runs/{id}/events?after=<id>
  GET  /api/runs/{id}/report   -> raw markdown
  GET  /api/config             -> effective settings + LLM provider

Runs execute in a background thread so the UI can poll live progress.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cortex.config import settings
from cortex.engine import new_run_id, run_research
from cortex.llm import get_llm
from cortex.persistence import (
    get_run,
    init_db,
    list_events,
    list_experiments,
    list_runs,
)

app = FastAPI(title="Cortex", version="1.0.0")
STATIC = Path(__file__).parent / "static"
init_db()

_active: dict[str, threading.Thread] = {}


class StartRequest(BaseModel):
    goal: Optional[str] = ""
    max_iterations: Optional[int] = None


def _run_in_thread(run_id: str, goal: str, max_iter: Optional[int]) -> None:
    try:
        run_research(goal=goal or "", max_iterations=max_iter, run_id=run_id)
    finally:
        _active.pop(run_id, None)


@app.post("/api/runs")
def start_run(req: StartRequest):
    run_id = new_run_id()
    t = threading.Thread(
        target=_run_in_thread, args=(run_id, req.goal or "", req.max_iterations), daemon=True
    )
    _active[run_id] = t
    t.start()
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs")
def runs():
    return {"runs": list_runs(), "active": list(_active.keys())}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    r = get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    r["is_active"] = run_id in _active
    return r


@app.get("/api/runs/{run_id}/experiments")
def experiments(run_id: str):
    return {"experiments": list_experiments(run_id)}


@app.get("/api/runs/{run_id}/events")
def events(run_id: str, after: int = 0):
    evs = list_events(run_id, after)
    last = evs[-1]["id"] if evs else after
    return {"events": evs, "last_id": last, "active": run_id in _active}


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
def report(run_id: str):
    r = get_run(run_id)
    if not r or not r.get("report_path") or not Path(r["report_path"]).exists():
        raise HTTPException(404, "report not ready")
    return Path(r["report_path"]).read_text(encoding="utf-8")


@app.get("/api/config")
def config():
    return {
        "llm_provider": get_llm().provider,
        "dataset": settings.dataset,
        "primary_metric": settings.primary_metric,
        "max_iterations": settings.max_iterations,
        "improvement_threshold": settings.improvement_threshold,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
