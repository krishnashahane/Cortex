"""Evaluator agent — the "Evaluate" step.

Scores the trained experiment against the primary metric, computes the
relative improvement vs the best score so far, persists the experiment to
SQLite and ChromaDB, and updates the run's best result. The improvement it
computes is exactly what the CEO uses for the <0.1% termination rule.
"""
from __future__ import annotations

from typing import Any

from ..config import settings
from ..memory import get_memory
from ..persistence import save_experiment
from .base import event, now_iso

AGENT = "Evaluator"


def evaluator_node(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("raw_result", {})
    metric = settings.primary_metric
    metrics = raw.get("metrics", {})
    score = float(metrics.get(metric, 0.0)) if raw.get("status") == "completed" else 0.0

    prev_best = state.get("best_score", -1.0)
    iteration = state.get("iteration", 1)

    # Relative improvement vs incumbent (guard against zero/negative baseline).
    if prev_best <= 0:
        improvement = 1.0 if score > 0 else 0.0
    else:
        improvement = max(0.0, (score - prev_best) / prev_best)

    exp = {
        "id": raw.get("experiment_id"),
        "iteration": iteration,
        "hypothesis_id": raw.get("hypothesis_id"),
        "config": raw.get("config", {}),
        "metrics": metrics,
        "primary_metric": metric,
        "score": score,
        "train_seconds": raw.get("train_seconds", 0.0),
        "status": raw.get("status", "completed"),
        "notes": raw.get("notes", ""),
    }

    # Persist to relational store + semantic memory.
    save_experiment(state.get("run_id", "unknown"), exp, now_iso())
    get_memory().add_experiment(
        exp["id"],
        f"{raw.get('model','')} config={exp['config']} -> {metric}={score:.4f}",
        {"iteration": iteration, "score": score, "model": raw.get("model", ""), "status": exp["status"]},
    )

    experiments = list(state.get("experiments", []))
    experiments.append(exp)

    patch: dict[str, Any] = {
        "experiments": experiments,
        "last_result": {**exp, "model": raw.get("model", "")},
        "last_improvement": improvement,
    }
    is_best = score > prev_best
    if is_best:
        patch["best_score"] = score
        patch["best_experiment_id"] = exp["id"]

    ev = event(
        state, AGENT, "evaluate",
        f"Score {metric}={score:.4f} (best={max(score, prev_best):.4f}, "
        f"improvement={improvement*100:.4f}%)",
        {"score": score, "improvement": improvement, "is_best": is_best},
    )
    patch["events"] = [ev]
    return patch
