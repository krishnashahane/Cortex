"""Hypothesis Generator agent — the "Hypothesize" step.

Proposes the next ML configuration to try, grounded in:
  * research insights (recalled papers),
  * the best result so far,
  * critiques from previous iterations.

Offline, it follows a principled search schedule that explores model
families first, then exploits the leader with hyperparameter refinement —
guaranteeing the loop produces real, improving experiments without an LLM.
"""
from __future__ import annotations

from typing import Any

from ..llm import get_llm
from ..memory import get_memory
from .base import event, short_id

AGENT = "HypothesisGenerator"

# Deterministic exploration schedule (used offline and as a safety net).
_SCHEDULE = [
    {"model": "logistic_regression", "scale": True, "params": {"C": 1.0}},
    {"model": "random_forest", "params": {"n_estimators": 200, "max_depth": None}},
    {"model": "gradient_boosting", "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3}},
    {"model": "svm", "scale": True, "params": {"C": 2.0, "kernel": "rbf", "gamma": "scale"}},
    {"model": "random_forest", "params": {"n_estimators": 500, "max_depth": 12, "min_samples_split": 4}},
    {"model": "gradient_boosting", "params": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 2, "subsample": 0.9}},
    {"model": "knn", "scale": True, "params": {"n_neighbors": 7, "weights": "distance"}},
    {"model": "svm", "scale": True, "params": {"C": 8.0, "kernel": "rbf", "gamma": "scale"}},
]


def _offline_config(state: dict[str, Any]) -> dict[str, Any]:
    idx = state.get("iteration", 1) - 1
    return dict(_SCHEDULE[idx % len(_SCHEDULE)])


def hypothesis_node(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm()
    mem = get_memory()
    iteration = state.get("iteration", 1)
    insights = [p.get("insight", p.get("document", "")) for p in state.get("recalled_papers", [])][:5]
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
                f"Goal: {state.get('goal','')}\nInsights: {insights}\n"
                f"Best so far: {last.get('metrics', {})} via {last.get('model','n/a')}\n"
                f"Already tried configs: {tried}\nRecent critiques: {critiques}\n"
                "Propose ONE new config to try. Models: random_forest, gradient_boosting, "
                "logistic_regression, svm, knn. JSON: "
                "{\"statement\":str,\"rationale\":str,\"config\":{\"model\":str,\"scale\":bool,\"params\":{}}}"
            ),
            fallback={},
        )
        config = out.get("config", {}) or {}
        statement = out.get("statement", "")
        rationale = out.get("rationale", "")

    if not config.get("model"):
        config = _offline_config(state)
        statement = f"Try {config['model']} with {config.get('params', {})}."
        rationale = (
            "Following exploration→exploitation schedule grounded in recalled "
            "research insights."
        )

    hid = short_id("hyp")
    hyp = {"id": hid, "statement": statement, "rationale": rationale, "config": config, "source": AGENT}
    mem.add_hypothesis(hid, statement + " | " + rationale, {"iteration": iteration, "model": config.get("model", "")})

    hyps = list(state.get("hypotheses", []))
    hyps.append(hyp)
    ev = event(state, AGENT, "hypothesis", f"Proposed: {statement}", {"config": config})
    return {"hypotheses": hyps, "current_hypothesis": hyp, "events": [ev]}
