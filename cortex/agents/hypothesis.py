"""Hypothesis Generator agent."""
from __future__ import annotations

from typing import Any

from ..llm import get_llm
from ..memory import get_memory
from .base import event, short_id

AGENT = "HypothesisGenerator"

_SCHEDULE = [
    {
        "model": "logistic_regression",
        "scale": True,
        "params": {"C": 1.0},
    },
    {
        "model": "random_forest",
        "params": {"n_estimators": 200, "max_depth": None},
    },
    {
        "model": "gradient_boosting",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
        },
    },
    {
        "model": "svm",
        "scale": True,
        "params": {"C": 2.0, "kernel": "rbf", "gamma": "scale"},
    },
    {
        "model": "random_forest",
        "params": {
            "n_estimators": 500,
            "max_depth": 12,
            "min_samples_split": 4,
        },
    },
    {
        "model": "gradient_boosting",
        "params": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 2,
            "subsample": 0.9,
        },
    },
    {
        "model": "knn",
        "scale": True,
        "params": {"n_neighbors": 7, "weights": "distance"},
    },
    {
        "model": "svm",
        "scale": True,
        "params": {"C": 8.0, "kernel": "rbf", "gamma": "scale"},
    },
]


def _offline_config(state: dict[str, Any]) -> dict[str, Any]:
    index = state.get("iteration", 1) - 1
    return dict(_SCHEDULE[index % len(_SCHEDULE)])


def hypothesis_node(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm()
    mem = get_memory()
    iteration = state.get("iteration", 1)
    insights = [
        p.get("insight", p.get("document", ""))
        for p in state.get("recalled_papers", [])
    ][:5]
    last = state.get("last_result", {})
    critiques = state.get("critiques", [])[-2:]
    tried = [e.get("config", {}) for e in state.get("experiments", [])]

    config: dict[str, Any] = {}
    statement = ""
    rationale = ""

    if llm.provider != "offline":
        out = llm.complete_json(
            system="You are an ML research scientist proposing the next experiment.",
            prompt=(
                f"Goal: {state.get('goal', '')}\n"
                f"Insights: {insights}\n"
                f"Best so far: {last.get('metrics', {})} via "
                f"{last.get('model', 'n/a')}\n"
                f"Already tried configs: {tried}\n"
                f"Recent critiques: {critiques}\n"
                "Propose ONE new config to try. Models: random_forest, "
                "gradient_boosting, logistic_regression, svm, knn. JSON: "
                "{\"statement\":str,\"rationale\":str,"
                "\"config\":{\"model\":str,\"scale\":bool,"
                "\"params\":{}}}"
            ),
            fallback={},
        )
        config = out.get("config", {}) or {}
        statement = str(out.get("statement", ""))
        rationale = str(out.get("rationale", ""))

    if not config.get("model"):
        config = _offline_config(state)
        statement = f"Try {config['model']} with {config.get('params', {})}."
        rationale = "Deterministic exploration and exploitation schedule."

    hid = short_id("hyp")
    hypothesis = {
        "id": hid,
        "statement": statement,
        "rationale": rationale,
        "config": config,
        "source": AGENT,
    }
    mem.add_hypothesis(
        hid,
        statement + " | " + rationale,
        {"iteration": iteration, "model": config.get("model", "")},
    )

    hypotheses = list(state.get("hypotheses", []))
    hypotheses.append(hypothesis)
    ev = event(
        state,
        AGENT,
        "hypothesis",
        f"Proposed: {statement}",
        {"config": config},
    )
    return {
        "hypotheses": hypotheses,
        "current_hypothesis": hypothesis,
        "events": [ev],
    }
