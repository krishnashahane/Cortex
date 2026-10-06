"""CEO agent — orchestration & termination authority.

The CEO sits at the top of every loop. It:
  * sets the research goal on the first pass,
  * decides whether to continue or terminate based on the improvement
    threshold (<0.1% relative gain) and the iteration budget,
  * records the rationale for the decision.

Termination logic is deterministic so the "<0.1% improvement" rule is exact
and auditable; the LLM only adds a narrative rationale.
"""
from __future__ import annotations

from typing import Any

from ..config import settings
from ..llm import get_llm
from .base import event

AGENT = "CEO"


def ceo_node(state: dict[str, Any]) -> dict[str, Any]:
    iteration = state.get("iteration", 0)
    patch: dict[str, Any] = {}
    events = []

    if iteration == 0:
        goal = state.get("goal") or (
            f"Maximize {settings.primary_metric} on the {settings.dataset} "
            f"classification task through iterative experimentation."
        )
        patch["goal"] = goal
        patch["best_score"] = state.get("best_score", -1.0)
        patch["stagnation_rounds"] = 0
        events.append(event(state, AGENT, "kickoff", f"Research goal set: {goal}"))
        patch["should_continue"] = True
        patch["iteration"] = 1
        patch["events"] = events
        return patch

    # Evaluate termination after a completed iteration.
    last_improvement = state.get("last_improvement", 1.0)
    stagnation = state.get("stagnation_rounds", 0)
    max_iter = state.get("max_iterations", settings.max_iterations)

    reason = ""
    cont = True
    if iteration >= max_iter:
        cont = False
        reason = f"Reached iteration budget ({max_iter})."
    elif last_improvement < settings.improvement_threshold:
        if stagnation + 1 >= settings.patience:
            cont = False
            reason = (
                f"Improvement {last_improvement*100:.4f}% < "
                f"{settings.improvement_threshold*100:.3f}% for {settings.patience} round(s)."
            )
        else:
            patch["stagnation_rounds"] = stagnation + 1
            reason = "Low improvement; one more attempt within patience."
    else:
        patch["stagnation_rounds"] = 0

    decision = "CONTINUE" if cont else "TERMINATE"
    msg = f"Iteration {iteration} decision: {decision}. {reason}".strip()
    events.append(event(state, AGENT, "decision", msg, {"improvement": last_improvement}))

    patch["should_continue"] = cont
    patch["termination_reason"] = reason if not cont else ""
    if cont:
        patch["iteration"] = iteration + 1
    patch["events"] = events
    return patch


def route_after_ceo(state: dict[str, Any]) -> str:
    """Conditional edge: continue the loop or finalize."""
    return "research" if state.get("should_continue", False) else "report"
