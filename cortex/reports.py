"""Markdown report generation.

Produces a self-contained research report per run: goal, leaderboard, the
full experiment trajectory, critiques, and the CEO's termination rationale.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings


def _fmt_metrics(m: dict[str, float]) -> str:
    if not m:
        return "—"
    return ", ".join(f"{k}={v:.4f}" for k, v in m.items())


def generate_markdown(state: dict[str, Any]) -> str:
    exps = sorted(state.get("experiments", []), key=lambda e: e.get("iteration", 0))
    ranked = sorted(exps, key=lambda e: e.get("score", 0), reverse=True)
    best = ranked[0] if ranked else {}
    metric = settings.primary_metric
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# Cortex Research Report — `{state.get('run_id','')}`")
    lines.append("")
    lines.append(f"*Generated {now}*")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(f"> {state.get('goal','')}")
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    lines.append(f"- **Iterations run:** {len(exps)}")
    lines.append(f"- **Best {metric}:** `{best.get('score', 0):.4f}`")
    lines.append(f"- **Best model:** `{best.get('config', {}).get('model', 'n/a')}`")
    lines.append(f"- **Termination reason:** {state.get('termination_reason') or 'n/a'}")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    lines.append(f"| Rank | Iter | Model | {metric} | All metrics | Train (s) |")
    lines.append("|---|---|---|---|---|---|")
    for i, e in enumerate(ranked[:10], 1):
        lines.append(
            f"| {i} | {e.get('iteration')} | `{e.get('config',{}).get('model','?')}` | "
            f"{e.get('score',0):.4f} | {_fmt_metrics(e.get('metrics',{}))} | "
            f"{e.get('train_seconds',0):.3f} |"
        )
    lines.append("")

    lines.append("## Experiment Trajectory")
    lines.append("")
    for e in exps:
        flag = " ⭐" if e.get("id") == best.get("id") else ""
        lines.append(f"### Iteration {e.get('iteration')}{flag}")
        lines.append("")
        lines.append(f"- Model: `{e.get('config',{}).get('model','?')}`")
        lines.append(f"- Params: `{e.get('config',{}).get('params', {})}`")
        lines.append(f"- Status: `{e.get('status','?')}`")
        lines.append(f"- Metrics: {_fmt_metrics(e.get('metrics',{}))}")
        if e.get("notes"):
            lines.append(f"- Notes: {e['notes']}")
        lines.append("")

    crit = state.get("critiques", [])
    if crit:
        lines.append("## Critic Log")
        lines.append("")
        for c in crit:
            lines.append(f"- **Iter {c.get('iteration')}**: {c.get('summary','')} "
                         f"→ _{c.get('suggestion','')}_")
        lines.append("")

    lines.append("## Knowledge Base")
    lines.append("")
    for p in state.get("papers", [])[:8]:
        lines.append(f"- *({p.get('topic','general')})* {p.get('insight','')}")
    lines.append("")
    lines.append("---")
    lines.append("*Produced autonomously by the Cortex multi-agent research system.*")
    return "\n".join(lines)


def write_report(state: dict[str, Any]) -> str:
    md = generate_markdown(state)
    path = Path(settings.reports_dir) / f"{state.get('run_id','run')}.md"
    path.write_text(md, encoding="utf-8")
    return str(path)
