from cortex.ml.trainer import train_and_evaluate


def test_trainer_produces_real_metrics():
    result = train_and_evaluate({"model": "random_forest", "params": {"n_estimators": 10}})
    assert result["model"] == "random_forest"
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["train_seconds"] >= 0.0


def test_trainer_bounds_untrusted_parameters():
    result = train_and_evaluate({
        "model": "random_forest",
        "params": {
            "n_estimators": 10_000_000,
            "max_depth": 10_000,
            "max_features": ["unexpected"],
        },
    })
    assert result["params"]["n_estimators"] == 600
    assert result["params"]["max_depth"] == 30
    assert result["params"]["max_features"] is None
