"""Dataset loading from scikit-learn's bundled datasets."""
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
    normalized = name.strip().lower()
    loader = _LOADERS.get(normalized)
    if loader is None:
        allowed = ", ".join(sorted(_LOADERS))
        raise ValueError(f"unsupported dataset {name!r}; choose one of: {allowed}")

    bunch = loader()
    X, y = bunch.data, bunch.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=settings.random_state,
        stratify=y,
    )
    meta = {
        "name": normalized,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "n_classes": int(len(set(y))),
    }
    return X_train, X_test, y_train, y_test, meta


def dataset_meta(name: str) -> dict[str, Any]:
    return load_split(name)[4]
