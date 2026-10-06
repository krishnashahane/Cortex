"""LangGraph assembly — the autonomous research loop.

Topology:

    START → CEO ─┬─(continue)→ research → hypothesize → plan →
                 │              train → evaluate → critique → CEO  (loop)
                 └─(terminate)→ report → END

The CEO is the only decision point. The conditional edge `route_after_ceo`
sends control back into the loop or out to the Report Writer based on the
deterministic <0.1% improvement / iteration-budget rule.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import (
    ceo_node,
    critic_node,
    evaluator_node,
    hypothesis_node,
    paper_reader_node,
    planner_node,
    reporter_node,
    route_after_ceo,
    trainer_node,
)
from .state import ResearchState


def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("ceo", ceo_node)
    g.add_node("research", paper_reader_node)
    g.add_node("hypothesize", hypothesis_node)
    g.add_node("plan", planner_node)
    g.add_node("train", trainer_node)
    g.add_node("evaluate", evaluator_node)
    g.add_node("critique", critic_node)
    g.add_node("report", reporter_node)

    g.add_edge(START, "ceo")
    g.add_conditional_edges("ceo", route_after_ceo, {"research": "research", "report": "report"})
    g.add_edge("research", "hypothesize")
    g.add_edge("hypothesize", "plan")
    g.add_edge("plan", "train")
    g.add_edge("train", "evaluate")
    g.add_edge("evaluate", "critique")
    g.add_edge("critique", "ceo")  # back to CEO for the continue/terminate decision
    g.add_edge("report", END)

    return g.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
