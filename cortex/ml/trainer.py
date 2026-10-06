"""Real scikit-learn training and evaluation."""
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

_ALLOWED = {
    "random_forest": {"n_estimators", "max_depth", "min_samples_split", "max_features", "criterion"},
    "gradient_boosting": {"n_estimators", "learning_rate", "max_depth", "subsample"},
    "logistic_regression": {"C", "penalty", "solver"},
    "svm": {"C", "kernel", "gamma"},
    "knn": {"n_neighbors", "weights", "p"},
}


def _float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _sanitize(model: str, params: Any) -> dict[str, Any]:
    incoming = params if isinstance(params, dict) else {}
    incoming = {k: v for k, v in incoming.items() if k in _ALLOWED[model]}

    if model == "random_forest":
        max_features = incoming.get("max_features")
        if isinstance(max_features, str):
            max_features = max_features if max_features in {"sqrt", "log2"} else None
        elif isinstance(max_features, (int, float)):
            max_features = _float(max_features, 1.0, 0.1, 1.0)
        else:
            max_features = None
        depth = incoming.get("max_depth")
        return {
            "n_estimators": _int(incoming.get("n_estimators", 200), 200, 10, 600),
            "max_depth": None if depth in (None, "") else _int(depth, 12, 1, 30),
            "min_samples_split": _int(incoming.get("min_samples_split", 2), 2, 2, 50),
            "max_features": max_features,
            "criterion": incoming.get("criterion")
            if incoming.get("criterion") in {"gini", "entropy", "log_loss"}
            else "gini",
        }

    if model == "gradient_boosting":
        return {
            "n_estimators": _int(incoming.get("n_estimators", 200), 200, 10, 600),
            "learning_rate": _float(incoming.get("learning_rate", 0.1), 0.1, 0.001, 1.0),
            "max_depth": _int(incoming.get("max_depth", 3), 3, 1, 10),
            "subsample": _float(incoming.get("subsample", 1.0), 1.0, 0.1, 1.0),
        }

    if model == "logistic_regression":
        solver = incoming.get("solver")
        if solver not in {"lbfgs", "liblinear"}:
            solver = "lbfgs"
        return {
            "C": _float(incoming.get("C", 1.0), 1.0, 0.0001, 1000.0),
            "penalty": "l2",
            "solver": solver,
        }

    if model == "svm":
        gamma = incoming.get("gamma", "scale")
        if not isinstance(gamma, (int, float, str)):
            gamma = "scale"
        elif isinstance(gamma, str) and gamma not in {"scale", "auto"}:
            gamma = "scale"
        elif isinstance(gamma, (int, float)):
            gamma = _float(gamma, 0.01, 0.000001, 10.0)
        kernel = incoming.get("kernel")
        if kernel not in {"linear", "poly", "rbf", "sigmoid"}:
            kernel = "rbf"
        return {
            "C": _float(incoming.get("C", 1.0), 1.0, 0.0001, 1000.0),
            "kernel": kernel,
            "gamma": gamma,
        }

    weights = incoming.get("weights")
    if weights not in {"uniform", "distance"}:
        weights = "uniform"
    return {
        "n_neighbors": _int(incoming.get("n_neighbors", 5), 5, 1, 50),
        "weights": weights,
        "p": _int(incoming.get("p", 2), 2, 1, 10),
    }


def _build(model: str, params: dict[str, Any]):
    if model == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=settings.random_state, **params)
    if model == "gradient_boosting":
        return GradientBoostingClassifier(random_state=settings.random_state, **params)
    if model == "svm":
        return SVC(probability=True, random_state=settings.random_state, **params)
    if model == "knn":
        return KNeighborsClassifier(**params)
    return RandomForestClassifier(random_state=settings.random_state, n_jobs=-1, **params)


def train_and_evaluate(config: dict[str, Any]) -> dict[str, Any]:
    config = config if isinstance(config, dict) else {}
    requested = str(config.get("model", "random_forest")).lower()
    model = requested if requested in _ALLOWED else "random_forest"
    params = _sanitize(model, config.get("params"))
    scale = bool(config.get("scale", model in {"logistic_regression", "svm", "knn"}))

    X_train, X_test, y_train, y_test, meta = load_split(settings.dataset)
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("classifier", _build(model, params)))
    pipeline = Pipeline(steps)

    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    predictions = pipeline.predict(X_test)
    average = "binary" if meta["n_classes"] == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, average=average, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average=average, zero_division=0)),
    }

    try:
        probabilities = pipeline.predict_proba(X_test)
        if meta["n_classes"] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities, multi_class="ovr"))
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
