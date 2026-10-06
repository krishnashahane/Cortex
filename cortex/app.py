"""Local-only FastAPI dashboard."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .engine import new_run_id, run_research
from .llm import get_llm
from .persistence import get_run, init_db, list_events, list_experiments, list_runs, upsert_run

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
MAX_ACTIVE_RUNS = 2

app = FastAPI(title="Cortex", version="1.1.0")
init_db()

_active: dict[str, threading.Thread] = {}
_active_lock = threading.Lock()


class StartRequest(BaseModel):
    goal: str = Field(default="", max_length=1000)
    max_iterations: int | None = Field(default=None, ge=1, le=30)


def _finish(run_id: str) -> None:
    with _active_lock:
        _active.pop(run_id, None)


def _worker(run_id: str, goal: str, max_iterations: int | None) -> None:
    try:
        run_research(
            goal=goal,
            max_iterations=max_iterations,
            run_id=run_id,
        )
    except Exception:
        pass
    finally:
        _finish(run_id)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs")
def start_run(request: StartRequest) -> dict[str, str]:
    with _active_lock:
        if len(_active) >= MAX_ACTIVE_RUNS:
            raise HTTPException(status_code=429, detail="too many active research runs")

        run_id = new_run_id()
        started_at = datetime.now(timezone.utc).isoformat()
        upsert_run(
            {
                "run_id": run_id,
                "goal": request.goal.strip() or "(auto)",
                "status": "running",
                "started_at": started_at,
                "finished_at": None,
                "iterations": 0,
                "best_score": 0.0,
                "best_experiment_id": None,
                "termination_reason": "",
                "report_path": None,
            }
        )

        thread = threading.Thread(
            target=_worker,
            args=(run_id, request.goal.strip(), request.max_iterations),
            daemon=True,
            name=f"cortex-{run_id}",
        )
        _active[run_id] = thread

    thread.start()
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs")
def runs() -> dict[str, object]:
    with _active_lock:
        active = list(_active)
    return {"runs": list_runs(), "active": active}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, object]:
    result = get_run(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="run not found")
    with _active_lock:
        result["is_active"] = run_id in _active
    return result


@app.get("/api/runs/{run_id}/experiments")
def experiments(run_id: str) -> dict[str, object]:
    if not get_run(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return {"experiments": list_experiments(run_id)}


@app.get("/api/runs/{run_id}/events")
def events(run_id: str, after: int = 0) -> dict[str, object]:
    if not get_run(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    rows = list_events(run_id, after)
    with _active_lock:
        active = run_id in _active
    return {
        "events": rows,
        "last_id": rows[-1]["id"] if rows else after,
        "active": active,
    }


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
def report(run_id: str) -> str:
    result = get_run(run_id)
    if not result or not result.get("report_path"):
        raise HTTPException(status_code=404, detail="report not ready")

    path = Path(str(result["report_path"])).resolve()
    root = Path(settings.reports_dir).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="report not ready")
    return path.read_text(encoding="utf-8")


@app.get("/api/config")
def config() -> dict[str, object]:
    return {
        "llm_provider": get_llm().provider,
        "dataset": settings.dataset,
        "primary_metric": settings.primary_metric,
        "max_iterations": settings.max_iterations,
        "improvement_threshold": settings.improvement_threshold,
        "patience": settings.patience,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
