"""Research engine."""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from .agents.base import now_iso
from .config import settings
from .graph import get_graph
from .persistence import init_db, upsert_run


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:10]}"


def initial_state(
    run_id: str,
    goal: str = "",
    max_iterations: int | None = None,
) -> dict[str, Any]:
    iterations = settings.max_iterations if max_iterations is None else int(max_iterations)
    iterations = max(1, min(30, iterations))
    return {
        "run_id": run_id,
        "started_at": now_iso(),
        "goal": goal,
        "iteration": 0,
        "max_iterations": iterations,
        "papers": [],
        "recalled_papers": [],
        "hypotheses": [],
        "experiments": [],
        "critiques": [],
        "events": [],
        "best_score": -1.0,
        "best_experiment_id": None,
        "last_improvement": 1.0,
        "stagnation_rounds": 0,
        "should_continue": True,
        "termination_reason": "",
    }


def run_research(
    goal: str = "",
    max_iterations: int | None = None,
    run_id: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    init_db()
    resolved_id = run_id or new_run_id()
    state = initial_state(resolved_id, goal, max_iterations)
    upsert_run({
        "run_id": resolved_id,
        "goal": state["goal"] or "(auto)",
        "status": "running",
        "started_at": state["started_at"],
        "finished_at": None,
        "iterations": 0,
        "best_score": 0.0,
        "best_experiment_id": None,
        "termination_reason": "",
        "report_path": None,
    })

    graph = get_graph()
    limit = (state["max_iterations"] + 2) * 8
    try:
        if on_event is None:
            return graph.invoke(state, config={"recursion_limit": limit})
        final: dict[str, Any] = {}
        for update in graph.stream(
            state,
            config={"recursion_limit": limit},
            stream_mode="values",
        ):
            final = update
            for item in update.get("events", [])[-1:]:
                on_event(item)
        return final
    except Exception as exc:
        upsert_run({
            "run_id": resolved_id,
            "goal": state["goal"] or "(auto)",
            "status": "failed",
            "started_at": state["started_at"],
            "finished_at": now_iso(),
            "iterations": len(state.get("experiments", [])),
            "best_score": state.get("best_score", 0.0),
            "best_experiment_id": state.get("best_experiment_id"),
            "termination_reason": f"Run failed: {exc}",
            "report_path": None,
        })
        raise
