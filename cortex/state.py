"""Shared LangGraph state."""
from __future__ import annotations

import operator
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict


@dataclass
class Hypothesis:
    id: str
    statement: str
    rationale: str
    config: dict[str, Any]
    source: str = "hypothesis_generator"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    id: str
    iteration: int
    hypothesis_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    primary_metric: str
    score: float
    train_seconds: float
    status: str = "completed"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchState(TypedDict, total=False):
    run_id: str
    started_at: str
    goal: str
    iteration: int
    max_iterations: int
    papers: list[dict[str, Any]]
    recalled_papers: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    current_hypothesis: dict[str, Any]
    plan: dict[str, Any]
    raw_result: dict[str, Any]
    last_result: dict[str, Any]
    experiments: list[dict[str, Any]]
    critiques: Annotated[list[dict[str, Any]], operator.add]
    best_score: float
    best_experiment_id: str | None
    last_improvement: float
    stagnation_rounds: int
    should_continue: bool
    termination_reason: str
    events: Annotated[list[dict[str, Any]], operator.add]
    report_path: Optional[str]
