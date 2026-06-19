"""Shared helpers for agents: event emission and id generation."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ..persistence import log_event


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def event(state: dict[str, Any], agent: str, kind: str, message: str, data: Any = None) -> dict[str, Any]:
    """Build an event dict and persist it. Returned for state['events'] merge."""
    ts = now_iso()
    run_id = state.get("run_id", "unknown")
    try:
        log_event(run_id, ts, agent, kind, message, data)
    except Exception:
        pass
    return {"ts": ts, "agent": agent, "kind": kind, "message": message, "data": data}
