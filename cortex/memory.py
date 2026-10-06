"""Local ChromaDB semantic memory."""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings

from .config import settings

_TOKEN = re.compile(r"[a-z0-9_]+")


class HashingEmbedding(EmbeddingFunction):
    """Deterministic local embedding; no model download is required."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def name(self) -> str:
        return "cortex-hashing-256"

    def _one(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        tokens = _TOKEN.findall((text or "").lower())
        grams = tokens + [
            f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)
        ]
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimension
            values[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def __call__(self, input: Documents) -> Embeddings:
        return [self._one(text) for text in input]


class Memory:
    def __init__(self, path: str | None = None) -> None:
        client = chromadb.PersistentClient(
            path=path or settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        embedding = HashingEmbedding()
        self.papers = client.get_or_create_collection("papers", embedding_function=embedding)
        self.hypotheses = client.get_or_create_collection(
            "hypotheses", embedding_function=embedding
        )
        self.experiments = client.get_or_create_collection(
            "experiments", embedding_function=embedding
        )

    def add_paper(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self.papers.upsert(ids=[doc_id], documents=[text], metadatas=[_clean(meta)])

    def add_hypothesis(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self.hypotheses.upsert(ids=[doc_id], documents=[text], metadatas=[_clean(meta)])

    def add_experiment(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self.experiments.upsert(ids=[doc_id], documents=[text], metadatas=[_clean(meta)])

    def recall(self, collection: str, query: str, k: int = 4) -> list[dict[str, Any]]:
        coll = getattr(self, collection)
        count = int(coll.count())
        if count == 0:
            return []
        result = coll.query(query_texts=[query], n_results=max(1, min(int(k), count)))
        documents = (result.get("documents") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        return [
            {
                "id": ids[index],
                "document": documents[index],
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
            for index in range(len(documents))
        ]

    def count(self, collection: str) -> int:
        return int(getattr(self, collection).count())


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
