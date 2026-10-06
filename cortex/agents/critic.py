"""Critic agent — the "Critique" step.

Analyzes the latest experiment in context of the run's trajectory and emits
a structured critique that steers the next hypothesis. Offline, it applies
heuristics (overfitting signals, plateau detection, model-family coverage);
with an LLM it produces richer reasoning. Critiques are appended to memory.
"""
from __future__ import annotations

from typing import Any

from ..config import settings
from ..llm import get_llm
from .base import event

AGENT = "Critic"


def _heuristic_critique(state: dict[str, Any]) -> dict[str, Any]:
    last = state.get("last_result", {})
    metrics = last.get("metrics", {})
    improvement = state.get("last_improvement", 0.0)
    acc = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1", 0.0)

    notes = []
    suggestion = "Explore a different model family."
    if last.get("status") == "failed":
        notes.append("Experiment failed; revert to a known-good config.")
        suggestion = "Use random_forest with default params."
    else:
        if improvement < settings.improvement_threshold:
            notes.append("Marginal gain — current direction is plateauing.")
            suggestion = "Pivot model family or tune the current leader's key hyperparameter."
        else:
            notes.append(f"Solid gain (+{improvement*100:.3f}%).")
            suggestion = "Exploit: refine hyperparameters around this config."
        if abs(acc - f1) > 0.05:
            notes.append("Accuracy/F1 gap suggests class imbalance sensitivity.")

    return {
        "iteration": state.get("iteration", 1),
        "summary": " ".join(notes),
        "suggestion": suggestion,
        "metrics": metrics,
    }


def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm()
    critique = _heuristic_critique(state)

    if llm.provider != "offline":
        out = llm.complete_json(
            system="You are a rigorous ML experiment critic.",
            prompt=(
                f"Latest result: {state.get('last_result', {}).get('metrics', {})}\n"
                f"Improvement: {state.get('last_improvement', 0)*100:.4f}%\n"
                f"History size: {len(state.get('experiments', []))}\n"
                "Give a critique. JSON: {\"summary\":str,\"suggestion\":str}"
            ),
            fallback={},
        )
        if out.get("summary"):
            critique["summary"] = out["summary"]
        if out.get("suggestion"):
            critique["suggestion"] = out["suggestion"]

    ev = event(state, AGENT, "critique", critique["summary"], {"suggestion": critique["suggestion"]})
    return {"critiques": [critique], "events": [ev]}
