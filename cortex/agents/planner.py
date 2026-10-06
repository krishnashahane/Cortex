"""Experiment Planner agent.

Turns a hypothesis into an executable experiment plan: validates the config,
fills defaults, and records the evaluation protocol. Keeps the Trainer simple
and ensures every experiment is reproducible (fixed seed, fixed split).
"""
from __future__ import annotations

from typing import Any

from ..config import settings
from ..ml.datasets import dataset_meta
from .base import event, short_id

AGENT = "ExperimentPlanner"

_VALID_MODELS = {"random_forest", "gradient_boosting", "logistic_regression", "svm", "knn"}


def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    hyp = state.get("current_hypothesis", {})
    config = dict(hyp.get("config", {}))
    model = (config.get("model") or "random_forest").lower()
    if model not in _VALID_MODELS:
        model = "random_forest"
    config["model"] = model

    plan = {
        "experiment_id": short_id("exp"),
        "hypothesis_id": hyp.get("id", "unknown"),
        "config": config,
        "dataset": settings.dataset,
        "dataset_meta": dataset_meta(settings.dataset),
        "primary_metric": settings.primary_metric,
        "protocol": "stratified 75/25 holdout, fixed seed",
    }
    ev = event(
        state, AGENT, "plan",
        f"Planned experiment {plan['experiment_id']} ({model}) on {settings.dataset}.",
        {"protocol": plan["protocol"]},
    )
    return {"plan": plan, "events": [ev]}
