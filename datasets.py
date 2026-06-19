"""Dataset loading. Uses scikit-learn's bundled datasets so Cortex runs
fully offline with real data — no downloads required.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from sklearn import datasets as skd
from sklearn.model_selection import train_test_split

from ..config import settings

_LOADERS = {
    "breast_cancer": skd.load_breast_cancer,
    "wine": skd.load_wine,
    "digits": skd.load_digits,
    "iris": skd.load_iris,
}


@lru_cache(maxsize=8)
def load_split(name: str) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    loader = _LOADERS.get(name, skd.load_breast_cancer)
    bunch = loader()
    X, y = bunch.data, bunch.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=settings.random_state, stratify=y
    )
    meta = {
        "name": name,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "n_classes": int(len(set(y))),
    }
    return X_tr, X_te, y_tr, y_te, meta


def dataset_meta(name: str) -> dict[str, Any]:
    return load_split(name)[4]
