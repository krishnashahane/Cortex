"""Long-term memory backed by ChromaDB.

Three collections give every agent durable, queryable recall:
  * papers       — literature/knowledge the Paper Reader gathers
  * hypotheses   — proposed ideas with their rationale
  * experiments  — results, so the Critic/CEO can compare across runs

ChromaDB uses its built-in default embedding (all-MiniLM) so no external
embedding API is required. Persistence is on-disk under data/chroma.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from typing import Any, Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from .config import settings

log = logging.getLogger("cortex.memory")

_TOKEN = re.compile(r"[a-z0-9_]+")


class HashingEmbedding(EmbeddingFunction):
    """Deterministic, dependency-free hashing embedding.

    Avoids ChromaDB's default ONNX model download (79MB) so Cortex stays fully
    offline and starts instantly. Hashes tokens (with bigrams) into a fixed
    feature space and L2-normalizes — good enough for semantic-ish recall.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def name(self) -> str:  # required by Chroma's EF interface
        return "cortex-hashing-256"

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _TOKEN.findall((text or "").lower())
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(t) for t in input]


class Memory:
    def __init__(self, path: Optional[str] = None) -> None:
        self.client = chromadb.PersistentClient(path=path or settings.chroma_dir)
        ef = HashingEmbedding()
        self.papers = self.client.get_or_create_collection("papers", embedding_function=ef)
        self.hypotheses = self.client.get_or_create_collection("hypotheses", embedding_function=ef)
        self.experiments = self.client.get_or_create_collection("experiments", embedding_function=ef)

    # --------------------------------------------------------------- store
    def add_paper(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self.papers.upsert(ids=[doc_id], documents=[text], metadatas=[_clean(meta)])

    def add_hypothesis(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self.hypotheses.upsert(ids=[doc_id], documents=[text], metadatas=[_clean(meta)])

    def add_experiment(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        self.experiments.upsert(ids=[doc_id], documents=[text], metadatas=[_clean(meta)])

    # --------------------------------------------------------------- recall
    def recall(self, collection: str, query: str, k: int = 4) -> list[dict[str, Any]]:
        coll = getattr(self, collection)
        if coll.count() == 0:
            return []
        res = coll.query(query_texts=[query], n_results=min(k, coll.count()))
        out: list[dict[str, Any]] = []
        for i, doc in enumerate(res.get("documents", [[]])[0]):
            out.append(
                {
                    "id": res["ids"][0][i],
                    "document": doc,
                    "metadata": (res.get("metadatas") or [[]])[0][i],
                }
            )
        return out

    def count(self, collection: str) -> int:
        return getattr(self, collection).count()


def _clean(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata must be flat scalar values."""
    flat: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            flat[k] = v
        else:
            flat[k] = json.dumps(v, default=str)
    return flat


_singleton: Optional[Memory] = None


def get_memory() -> Memory:
    global _singleton
    if _singleton is None:
        _singleton = Memory()
    return _singleton
