"""Paper Reader agent — the "Research" step.

Gathers domain knowledge relevant to the goal and writes it to long-term
memory (ChromaDB `papers`). With an LLM it synthesizes findings; offline it
falls back to a curated knowledge base of ML technique notes so downstream
agents still get useful, queryable context.
"""
from __future__ import annotations

from typing import Any

from ..llm import get_llm
from ..memory import get_memory
from .base import event, short_id

AGENT = "PaperReader"

# Curated offline knowledge — concise, technique-oriented notes.
_KB = [
    ("Random forests reduce variance via bagging; tuning n_estimators (100-600) "
     "and max_depth balances bias/variance on tabular data.", "random_forest"),
    ("Gradient boosting often beats random forests on structured data when "
     "learning_rate is small (0.01-0.1) with more estimators and shallow trees.", "gradient_boosting"),
    ("Feature scaling (StandardScaler) is essential for SVM, kNN and logistic "
     "regression but irrelevant for tree ensembles.", "preprocessing"),
    ("Regularization strength C in logistic regression and SVM trades off margin "
     "vs misclassification; sweep on a log scale.", "regularization"),
    ("kNN performance is sensitive to n_neighbors and distance metric; scaling "
     "features first is critical.", "knn"),
    ("RBF-kernel SVMs capture nonlinear boundaries; jointly tune C and gamma.", "svm"),
]


def paper_reader_node(state: dict[str, Any]) -> dict[str, Any]:
    goal = state.get("goal", "")
    mem = get_memory()
    llm = get_llm()
    iteration = state.get("iteration", 1)

    findings: list[dict[str, Any]] = []
    if llm.provider != "offline":
        out = llm.complete_json(
            system="You are an ML research librarian.",
            prompt=(
                f"Goal: {goal}. List 4 concise, actionable modeling insights for a "
                f"tabular classification task. JSON: {{\"findings\":[{{\"insight\":str,\"topic\":str}}]}}"
            ),
            fallback={"findings": []},
        )
        for f in out.get("findings", [])[:6]:
            if isinstance(f, dict) and f.get("insight"):
                findings.append({"insight": f["insight"], "topic": f.get("topic", "general")})

    if not findings:  # offline or LLM gave nothing
        for insight, topic in _KB:
            findings.append({"insight": insight, "topic": topic})

    papers = list(state.get("papers", []))
    for f in findings:
        pid = short_id("paper")
        mem.add_paper(pid, f["insight"], {"topic": f["topic"], "iteration": iteration, "goal": goal})
        papers.append({"id": pid, **f})

    # Recall prior knowledge to feed hypothesis generation.
    recalled = mem.recall("papers", goal, k=5)

    ev = event(
        state, AGENT, "research",
        f"Gathered {len(findings)} insights; memory now holds {mem.count('papers')} papers.",
        {"topics": [f["topic"] for f in findings]},
    )
    return {"papers": papers[-12:], "recalled_papers": recalled, "events": [ev]}
