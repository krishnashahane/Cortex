"""Trainer agent — the "Train" step.

Executes the planned experiment by invoking the real ML trainer. Robust to
bad configs: on failure it records a failed experiment rather than crashing
the graph, so the CEO can still reason about the run.
"""
from __future__ import annotations

from typing import Any

from ..config import settings
from ..ml.trainer import train_and_evaluate
from .base import event

AGENT = "Trainer"


def trainer_node(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan", {})
    config = plan.get("config", {})
    try:
        result = train_and_evaluate(config)
        raw = {
            "experiment_id": plan.get("experiment_id"),
            "hypothesis_id": plan.get("hypothesis_id"),
            "config": config,
            "metrics": result["metrics"],
            "train_seconds": result["train_seconds"],
            "model": result["model"],
            "status": "completed",
        }
        ev = event(
            state, AGENT, "train",
            f"Trained {result['model']} in {result['train_seconds']}s — "
            f"{settings.primary_metric}={result['metrics'].get(settings.primary_metric, 0):.4f}",
            {"metrics": result["metrics"]},
        )
    except Exception as exc:  # defensive: never break the loop
        raw = {
            "experiment_id": plan.get("experiment_id"),
            "hypothesis_id": plan.get("hypothesis_id"),
            "config": config,
            "metrics": {},
            "train_seconds": 0.0,
            "model": config.get("model", "unknown"),
            "status": "failed",
            "notes": str(exc),
        }
        ev = event(state, AGENT, "error", f"Training failed: {exc}")
    return {"raw_result": raw, "events": [ev]}
