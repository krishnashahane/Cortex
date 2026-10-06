"""Run engine — drives the LangGraph graph to completion.

Provides a single `run_research` entry point used by both the CLI and the
FastAPI server. Handles run-record bootstrapping, recursion budget, and
returns the final state.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from .config import settings
from .graph import get_graph
from .persistence import init_db, upsert_run
from .agents.base import now_iso


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:10]}"


def initial_state(run_id: str, goal: str = "", max_iterations: Optional[int] = None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": now_iso(),
        "goal": goal,
        "iteration": 0,
        "max_iterations": max_iterations or settings.max_iterations,
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
    max_iterations: Optional[int] = None,
    run_id: Optional[str] = None,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    init_db()
    run_id = run_id or new_run_id()
    state = initial_state(run_id, goal, max_iterations)

    upsert_run(
        {
            "run_id": run_id,
            "goal": goal or "(auto)",
            "status": "running",
            "started_at": state["started_at"],
            "finished_at": None,
            "iterations": 0,
            "best_score": 0.0,
            "best_experiment_id": None,
            "termination_reason": "",
            "report_path": None,
        }
    )

    graph = get_graph()
    # Generous recursion limit: ~7 nodes per loop * max_iterations + margin.
    limit = (settings.max_iterations + 2) * 8
    config = {"recursion_limit": limit}

    if on_event:
        final: dict[str, Any] = {}
        for update in graph.stream(state, config=config, stream_mode="values"):
            final = update
            for ev in update.get("events", [])[-1:]:
                on_event(ev)
        return final

    return graph.invoke(state, config=config)
