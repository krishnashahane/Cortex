"""SQLite persistence for runs and experiments.

ChromaDB holds semantic memory; this relational store holds the structured
ground truth used by the dashboard and reports. Kept dependency-free
(stdlib sqlite3) and thread-safe for the FastAPI server.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Optional

from .config import settings

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    goal TEXT,
    status TEXT,
    started_at TEXT,
    finished_at TEXT,
    iterations INTEGER,
    best_score REAL,
    best_experiment_id TEXT,
    termination_reason TEXT,
    report_path TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    iteration INTEGER,
    hypothesis_id TEXT,
    config TEXT,
    metrics TEXT,
    primary_metric TEXT,
    score REAL,
    train_seconds REAL,
    status TEXT,
    notes TEXT,
    created_at TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    ts TEXT,
    agent TEXT,
    kind TEXT,
    message TEXT,
    data TEXT
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(settings.db_path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _lock, _conn() as c:
        c.executescript(_SCHEMA)


def upsert_run(run: dict[str, Any]) -> None:
    cols = [
        "run_id", "goal", "status", "started_at", "finished_at", "iterations",
        "best_score", "best_experiment_id", "termination_reason", "report_path",
    ]
    vals = [run.get(k) for k in cols]
    with _lock, _conn() as c:
        c.execute(
            f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
            f"ON CONFLICT(run_id) DO UPDATE SET "
            + ",".join(f"{k}=excluded.{k}" for k in cols if k != "run_id"),
            vals,
        )


def save_experiment(run_id: str, exp: dict[str, Any], created_at: str) -> None:
    with _lock, _conn() as c:
        c.execute(
            """INSERT INTO experiments
            (id,run_id,iteration,hypothesis_id,config,metrics,primary_metric,
             score,train_seconds,status,notes,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET score=excluded.score,
            metrics=excluded.metrics,status=excluded.status""",
            (
                exp["id"], run_id, exp["iteration"], exp["hypothesis_id"],
                json.dumps(exp["config"]), json.dumps(exp["metrics"]),
                exp["primary_metric"], exp["score"], exp["train_seconds"],
                exp.get("status", "completed"), exp.get("notes", ""), created_at,
            ),
        )


def log_event(run_id: str, ts: str, agent: str, kind: str, message: str, data: Any = None) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO events (run_id,ts,agent,kind,message,data) VALUES (?,?,?,?,?,?)",
            (run_id, ts, agent, kind, message, json.dumps(data, default=str) if data else None),
        )


def get_run(run_id: str) -> Optional[dict[str, Any]]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def list_experiments(run_id: str) -> list[dict[str, Any]]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM experiments WHERE run_id=? ORDER BY iteration", (run_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d["config"]) if d["config"] else {}
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            out.append(d)
        return out


def list_events(run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id", (run_id, after_id)
        ).fetchall()
        return [dict(r) for r in rows]
