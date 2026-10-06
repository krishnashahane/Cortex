"""CEO agent — orchestration and termination authority."""
from __future__ import annotations

from ..config import settings
from .base import event

AGENT = "CEO"


def ceo_node(state: dict) -> dict:
    iteration = state.get("iteration", 0)
    patch: dict = {}
    events = []

    if iteration == 0:
        goal = state.get("goal") or (
            f"Maximize {settings.primary_metric} on the {settings.dataset} "
            "classification task through iterative experimentation."
        )
        patch["goal"] = goal
        patch["best_score"] = state.get("best_score", -1.0)
        patch["stagnation_rounds"] = 0
        events.append(event(state, AGENT, "kickoff", f"Research goal set: {goal}"))
        patch["should_continue"] = True
        patch["iteration"] = 1
        patch["events"] = events
        return patch

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
                f"Improvement {last_improvement * 100:.4f}% < "
                f"{settings.improvement_threshold * 100:.3f}% for {settings.patience} round(s)."
            )
        else:
            patch["stagnation_rounds"] = stagnation + 1
            reason = "Low improvement; one more attempt within patience."
    else:
        patch["stagnation_rounds"] = 0

    decision = "CONTINUE" if cont else "TERMINATE"
    msg = f"Iteration {iteration} decision: {decision}. {reason}".strip()
    events.append(
        event(state, AGENT, "decision", msg, {"improvement": last_improvement})
    )

    patch["should_continue"] = cont
    patch["termination_reason"] = reason if not cont else ""
    if cont:
        patch["iteration"] = iteration + 1
    patch["events"] = events
    return patch


def route_after_ceo(state: dict) -> str:
    return "research" if state.get("should_continue", False) else "report"
