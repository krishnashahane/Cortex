"""SQLite persistence."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from .config import settings

_lock = threading.RLock()

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    iterations INTEGER NOT NULL DEFAULT 0,
    best_score REAL NOT NULL DEFAULT 0,
    best_experiment_id TEXT,
    termination_reason TEXT,
    report_path TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    hypothesis_id TEXT,
    config TEXT NOT NULL,
    metrics TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    score REAL NOT NULL,
    train_seconds REAL NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_experiments_run_iteration ON experiments(run_id, iteration);
CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id, id);
"""


def _secure(path: str) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _conn() -> sqlite3.Connection:
    connection = sqlite3.connect(settings.db_path, timeout=10, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def init_db() -> None:
    with _lock, _conn() as connection:
        connection.executescript(_SCHEMA)
    _secure(settings.db_path)


def upsert_run(run: dict[str, Any]) -> None:
    columns = [
        "run_id", "goal", "status", "started_at", "finished_at", "iterations",
        "best_score", "best_experiment_id", "termination_reason", "report_path",
    ]
    values = [run.get(column) for column in columns]
    assignments = ",".join(f"{column}=excluded.{column}" for column in columns[1:])
    placeholders = ",".join("?" for _ in columns)
    query = (
        f"INSERT INTO runs ({','.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(run_id) DO UPDATE SET {assignments}"
    )
    with _lock, _conn() as connection:
        connection.execute(query, values)


def save_experiment(run_id: str, experiment: dict[str, Any], created_at: str) -> None:
    with _lock, _conn() as connection:
        connection.execute(
            """
            INSERT INTO experiments
            (id,run_id,iteration,hypothesis_id,config,metrics,primary_metric,
             score,train_seconds,status,notes,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              metrics=excluded.metrics,
              score=excluded.score,
              train_seconds=excluded.train_seconds,
              status=excluded.status,
              notes=excluded.notes
            """,
            (
                experiment["id"], run_id, experiment["iteration"],
                experiment.get("hypothesis_id"), json.dumps(experiment["config"], default=str),
                json.dumps(experiment["metrics"], default=str), experiment["primary_metric"],
                experiment["score"], experiment["train_seconds"],
                experiment.get("status", "completed"), experiment.get("notes", ""), created_at,
            ),
        )


def log_event(
    run_id: str,
    ts: str,
    agent: str,
    kind: str,
    message: str,
    data: Any = None,
) -> None:
    with _lock, _conn() as connection:
        connection.execute(
            "INSERT INTO events (run_id,ts,agent,kind,message,data) VALUES (?,?,?,?,?,?)",
            (
                run_id, ts, agent, kind, message,
                json.dumps(data, default=str) if data is not None else None,
            ),
        )


def get_run(run_id: str) -> dict[str, Any] | None:
    with _lock, _conn() as connection:
        row = connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 100))
    with _lock, _conn() as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (safe_limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def list_experiments(run_id: str) -> list[dict[str, Any]]:
    with _lock, _conn() as connection:
        rows = connection.execute(
            "SELECT * FROM experiments WHERE run_id=? ORDER BY iteration, created_at",
            (run_id,),
        ).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        item["config"] = json.loads(item["config"]) if item["config"] else {}
        item["metrics"] = json.loads(item["metrics"]) if item["metrics"] else {}
        output.append(item)
    return output


def list_events(run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    with _lock, _conn() as connection:
        rows = connection.execute(
            "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id",
            (run_id, max(0, int(after_id))),
        ).fetchall()
    return [dict(row) for row in rows]
