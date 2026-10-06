"""The actual model trainer.

A Cortex "hypothesis" compiles to a concrete sklearn pipeline config. This
module turns a config dict into a fitted model and real metrics. This is
what makes the improvement-based termination meaningful: scores come from
genuine train/eval, not a simulation.

Config schema (all optional, with safe defaults):
{
  "model": "random_forest|logistic_regression|gradient_boosting|svm|knn",
  "scale": true,
  "params": { ...model hyperparameters... }
}
"""
from __future__ import annotations

import time
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import settings
from .datasets import load_split

_RANDOM_MODELS = {"random_forest", "gradient_boosting"}


def _build_estimator(model: str, params: dict[str, Any]):
    rs = settings.random_state
    if model == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=rs, **params)
    if model == "gradient_boosting":
        return GradientBoostingClassifier(random_state=rs, **params)
    if model == "svm":
        return SVC(probability=True, random_state=rs, **params)
    if model == "knn":
        return KNeighborsClassifier(**params)
    # default
    return RandomForestClassifier(random_state=rs, n_jobs=-1, **params)


# Whitelist of accepted hyperparameters per model -> guards against bad LLM output.
_ALLOWED = {
    "random_forest": {"n_estimators", "max_depth", "min_samples_split", "max_features", "criterion"},
    "gradient_boosting": {"n_estimators", "learning_rate", "max_depth", "subsample"},
    "logistic_regression": {"C", "penalty", "solver"},
    "svm": {"C", "kernel", "gamma"},
    "knn": {"n_neighbors", "weights", "p"},
}


def _sanitize(model: str, params: dict[str, Any]) -> dict[str, Any]:
    allowed = _ALLOWED.get(model, set())
    return {k: v for k, v in (params or {}).items() if k in allowed}


def train_and_evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """Train one model from a config and return metrics + timing."""
    model = (config.get("model") or "random_forest").lower()
    if model not in _ALLOWED:
        model = "random_forest"
    params = _sanitize(model, config.get("params", {}))
    scale = bool(config.get("scale", model in {"logistic_regression", "svm", "knn"}))

    X_tr, X_te, y_tr, y_te, meta = load_split(settings.dataset)

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", _build_estimator(model, params)))
    pipe = Pipeline(steps)

    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    train_seconds = time.perf_counter() - t0

    pred = pipe.predict(X_te)
    avg = "binary" if meta["n_classes"] == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred, average=avg, zero_division=0)),
        "precision": float(precision_score(y_te, pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_te, pred, average=avg, zero_division=0)),
    }
    # ROC-AUC when probabilities available
    try:
        proba = pipe.predict_proba(X_te)
        if meta["n_classes"] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_te, proba[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_te, proba, multi_class="ovr"))
    except Exception:
        pass

    return {
        "model": model,
        "scale": scale,
        "params": params,
        "metrics": metrics,
        "train_seconds": round(train_seconds, 4),
        "dataset_meta": meta,
    }
