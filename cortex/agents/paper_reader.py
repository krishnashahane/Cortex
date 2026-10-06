"""Paper Reader agent."""
from __future__ import annotations

from typing import Any

from ..llm import get_llm
from ..memory import get_memory
from .base import event, short_id

AGENT = "PaperReader"

_KB = [
    (
        "Random forests reduce variance via bagging; tuning n_estimators (100-600) "
        "and max_depth balances bias/variance on tabular data.",
        "random_forest",
    ),
    (
        "Gradient boosting often beats random forests on structured data when "
        "learning_rate is small (0.01-0.1) with more estimators and shallow trees.",
        "gradient_boosting",
    ),
    (
        "Feature scaling (StandardScaler) is essential for SVM, kNN and logistic "
        "regression but irrelevant for tree ensembles.",
        "preprocessing",
    ),
    (
        "Regularization strength C in logistic regression and SVM trades off margin "
        "vs misclassification; sweep on a log scale.",
        "regularization",
    ),
    (
        "kNN performance is sensitive to n_neighbors and distance metric; scaling "
        "features first is critical.",
        "knn",
    ),
    (
        "RBF-kernel SVMs capture nonlinear boundaries; jointly tune C and gamma.",
        "svm",
    ),
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
                "tabular classification task. JSON: "
                "{\"findings\":[{\"insight\":str,\"topic\":str}]}"
            ),
            fallback={"findings": []},
        )
        for item in out.get("findings", [])[:6]:
            if isinstance(item, dict) and item.get("insight"):
                findings.append(
                    {
                        "insight": str(item["insight"]),
                        "topic": str(item.get("topic", "general")),
                    }
                )

    if not findings:
        findings = [
            {"insight": insight, "topic": topic}
            for insight, topic in _KB
        ]

    papers = list(state.get("papers", []))
    for finding in findings:
        pid = short_id("paper")
        mem.add_paper(
            pid,
            finding["insight"],
            {
                "topic": finding["topic"],
                "iteration": iteration,
                "goal": goal,
            },
        )
        papers.append({"id": pid, **finding})

    recalled = mem.recall("papers", goal, k=5)
    ev = event(
        state,
        AGENT,
        "research",
        f"Gathered {len(findings)} insights; memory now holds "
        f"{mem.count('papers')} papers.",
        {"topics": [item["topic"] for item in findings]},
    )
    return {
        "papers": papers[-12:],
        "recalled_papers": recalled,
        "events": [ev],
    }
