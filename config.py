"""Central configuration for Cortex.

All tunables live here so the system stays modular and easy to operate.
Values can be overridden via environment variables (loaded from .env).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "cortex.db"

for _d in (DATA_DIR, REPORTS_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _flt(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # Research loop control
    max_iterations: int = _int("CORTEX_MAX_ITERATIONS", 8)
    # Terminate when relative improvement between iterations < this (0.1%)
    improvement_threshold: float = _flt("CORTEX_IMPROVEMENT_THRESHOLD", 0.001)
    # How many consecutive low-improvement rounds before stopping
    patience: int = _int("CORTEX_PATIENCE", 2)

    # ML task
    dataset: str = os.getenv("CORTEX_DATASET", "breast_cancer")
    primary_metric: str = os.getenv("CORTEX_PRIMARY_METRIC", "accuracy")
    random_state: int = _int("CORTEX_RANDOM_STATE", 42)

    # LLM
    llm_provider: str = os.getenv("CORTEX_LLM_PROVIDER", "auto")  # auto|anthropic|gemini|offline
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    anthropic_model: str = os.getenv("CORTEX_ANTHROPIC_MODEL", "claude-opus-4-8")
    gemini_model: str = os.getenv("CORTEX_GEMINI_MODEL", "gemini-2.5-flash")

    # Paths (kept as strings for serialization friendliness)
    chroma_dir: str = str(CHROMA_DIR)
    db_path: str = str(DB_PATH)
    reports_dir: str = str(REPORTS_DIR)

    tags: dict = field(default_factory=dict)


settings = Settings()
