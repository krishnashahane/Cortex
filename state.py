"""Shared graph state and core data structures.

The LangGraph state is a plain TypedDict so it serializes cleanly and any
node can read/patch it. Domain objects (hypotheses, experiments) are typed
dataclasses for safety, stored inside the state as dicts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Optional, TypedDict
import operator


@dataclass
class Hypothesis:
    id: str
    statement: str
    rationale: str
    config: dict[str, Any]  # proposed ML configuration to test
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
    """The object that flows through the LangGraph graph."""

    run_id: str
    started_at: str
    goal: str
    iteration: int
    max_iterations: int

    # Accumulating context produced by agents
    papers: list[dict[str, Any]]
    recalled_papers: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    current_hypothesis: dict[str, Any]
    plan: dict[str, Any]
    raw_result: dict[str, Any]
    last_result: dict[str, Any]
    experiments: list[dict[str, Any]]
    critiques: Annotated[list[dict[str, Any]], operator.add]

    # Scoring / termination
    best_score: float
    best_experiment_id: Optional[str]
    last_improvement: float
    stagnation_rounds: int
    should_continue: bool
    termination_reason: str

    # Logging — append-only event stream for the dashboard
    events: Annotated[list[dict[str, Any]], operator.add]

    report_path: Optional[str]
