"""Local semantic memory without a network service or third-party vector database."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from typing import Any

from .config import settings

_TOKEN = re.compile(r"[a-z0-9_]+")
_KIND_MAP = {
    "papers": "paper",
    "hypotheses": "hypothesis",
    "experiments": "experiment",
}
_lock = threading.RLock()


def _embedding(text: str, dimension: int = 256) -> list[float]:
    values = [0.0] * dimension
    tokens = _TOKEN.findall((text or "").lower())
    grams = tokens + [
        f"{left}_{right}"
        for left, right in zip(tokens, tokens[1:], strict=False)
    ]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % dimension
        values[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class Memory:
    """Small local semantic store using deterministic hashed embeddings."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or settings.db_path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with _lock, sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    document TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_memory_kind ON semantic_memory(kind)"
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _add(self, kind: str, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        if kind not in _KIND_MAP.values():
            raise ValueError(f"unsupported memory kind: {kind}")
        with _lock, sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute(
                """
                INSERT INTO semantic_memory(id,kind,document,metadata,embedding)
                VALUES (?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  kind=excluded.kind,
                  document=excluded.document,
                  metadata=excluded.metadata,
                  embedding=excluded.embedding
                """,
                (
                    doc_id,
                    kind,
                    text,
                    json.dumps(_clean(meta), default=str),
                    json.dumps(_embedding(text), separators=(",", ":")),
                ),
            )

    def add_paper(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self._add("paper", doc_id, text, meta)

    def add_hypothesis(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self._add("hypothesis", doc_id, text, meta)

    def add_experiment(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self._add("experiment", doc_id, text, meta)

    def recall(self, collection: str, query: str, k: int = 4) -> list[dict[str, Any]]:
        kind = _KIND_MAP.get(collection)
        if kind is None:
            raise ValueError(f"unsupported collection: {collection}")
        safe_k = max(1, min(int(k), 50))
        query_vector = _embedding(query)

        with _lock, sqlite3.connect(self.path, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, document, metadata, embedding
                FROM semantic_memory
                WHERE kind=?
                ORDER BY rowid DESC
                LIMIT 500
                """,
                (kind,),
            ).fetchall()

        scored = []
        for row in rows:
            vector = json.loads(row["embedding"])
            score = sum(left * right for left, right in zip(
                query_vector, vector, strict=False
            ))
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            scored.append(
                {
                    "id": row["id"],
                    "document": row["document"],
                    "metadata": metadata,
                    "_score": score,
                }
            )

        scored.sort(key=lambda item: item["_score"], reverse=True)
        for item in scored:
            item.pop("_score", None)
        return scored[:safe_k]

    def count(self, collection: str) -> int:
        kind = _KIND_MAP.get(collection)
        if kind is None:
            raise ValueError(f"unsupported collection: {collection}")
        with _lock, sqlite3.connect(self.path, timeout=10) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM semantic_memory WHERE kind=?", (kind,)
            ).fetchone()
        return int(row[0]) if row else 0


def _clean(meta: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            output[key] = value
        else:
            output[key] = json.dumps(value, default=str)
    return output


_memory: Memory | None = None


def get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory
