"""Report Writer agent — terminal node.

Renders the markdown report, persists the final run record, and marks the
run finished. Runs exactly once when the CEO decides to terminate.
"""
from __future__ import annotations

from typing import Any

from ..persistence import upsert_run
from ..reports import write_report
from .base import event, now_iso

AGENT = "ReportWriter"


def reporter_node(state: dict[str, Any]) -> dict[str, Any]:
    path = write_report(state)
    exps = state.get("experiments", [])

    upsert_run(
        {
            "run_id": state.get("run_id"),
            "goal": state.get("goal"),
            "status": "completed",
            "started_at": state.get("started_at"),
            "finished_at": now_iso(),
            "iterations": len(exps),
            "best_score": state.get("best_score", 0.0),
            "best_experiment_id": state.get("best_experiment_id"),
            "termination_reason": state.get("termination_reason", ""),
            "report_path": path,
        }
    )
    ev = event(state, AGENT, "report", f"Report written to {path}. Run complete.", {"path": path})
    return {"report_path": path, "events": [ev]}
