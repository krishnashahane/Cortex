"""Specialized Cortex agents.

Each agent is a pure function (state) -> partial_state, registered as a node
in the LangGraph graph. Agents:
    CEO, PaperReader, HypothesisGenerator, ExperimentPlanner,
    Trainer, Evaluator, Critic, ReportWriter.
"""
from .ceo import ceo_node, route_after_ceo
from .paper_reader import paper_reader_node
from .hypothesis import hypothesis_node
from .planner import planner_node
from .trainer_agent import trainer_node
from .evaluator import evaluator_node
from .critic import critic_node
from .reporter import reporter_node

__all__ = [
    "ceo_node",
    "route_after_ceo",
    "paper_reader_node",
    "hypothesis_node",
    "planner_node",
    "trainer_node",
    "evaluator_node",
    "critic_node",
    "reporter_node",
]
