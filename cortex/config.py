"""Central configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


DATA_DIR = Path(os.getenv("CORTEX_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
REPORTS_DIR = Path(os.getenv("CORTEX_REPORTS_DIR", str(ROOT / "reports"))).expanduser().resolve()
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "cortex.db"

for directory in (DATA_DIR, REPORTS_DIR, CHROMA_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(0o700)
    except OSError:
        pass


@dataclass(frozen=True)
class Settings:
    max_iterations: int = _int("CORTEX_MAX_ITERATIONS", 8, 1, 30)
    improvement_threshold: float = _float("CORTEX_IMPROVEMENT_THRESHOLD", 0.001, 0.0, 1.0)
    patience: int = _int("CORTEX_PATIENCE", 2, 1, 10)
    dataset: str = os.getenv("CORTEX_DATASET", "breast_cancer").strip().lower()
    primary_metric: str = os.getenv("CORTEX_PRIMARY_METRIC", "accuracy").strip().lower()
    random_state: int = _int("CORTEX_RANDOM_STATE", 42, 0, 2**31 - 1)
    llm_provider: str = os.getenv("CORTEX_LLM_PROVIDER", "auto").strip().lower()
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    anthropic_model: str = os.getenv("CORTEX_ANTHROPIC_MODEL", "claude-opus-4-8").strip()
    gemini_model: str = os.getenv("CORTEX_GEMINI_MODEL", "gemini-3.8-flash").strip()
    data_dir: str = str(DATA_DIR)
    chroma_dir: str = str(CHROMA_DIR)
    db_path: str = str(DB_PATH)
    reports_dir: str = str(REPORTS_DIR)


settings = Settings()

if settings.dataset not in {"breast_cancer", "wine", "digits", "iris"}:
    raise ValueError("CORTEX_DATASET must be one of: breast_cancer, wine, digits, iris")
if settings.primary_metric not in {"accuracy", "f1", "precision", "recall", "roc_auc"}:
    raise ValueError(
        "CORTEX_PRIMARY_METRIC must be accuracy, f1, precision, recall, or roc_auc"
    )
