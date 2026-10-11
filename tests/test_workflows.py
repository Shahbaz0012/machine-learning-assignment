import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_house_prices import train_and_evaluate as train_house_prices
from train_titanic import train_and_evaluate as train_titanic


def test_house_price_workflow_returns_metrics():
    rng = np.random.default_rng(42)
    area = rng.integers(800, 3000, size=40)
    data = pd.DataFrame({"GrLivArea": area, "OverallQual": rng.integers(1, 10, size=40), "SalePrice": area * 120 + rng.normal(0, 5000, size=40)})
    _, actual, predicted, metrics = train_house_prices(data)
    assert len(actual) == len(predicted) == 8
    assert metrics["numeric_features"] == 2
    assert np.isfinite(metrics["r2_score"])


def test_titanic_workflow_returns_accuracy():
    data = pd.DataFrame({
        "Pclass": [1, 2, 3, 1, 2, 3] * 8,
        "Sex": ["female", "male"] * 24,
        "Age": list(range(20, 68)),
        "SibSp": [0, 1] * 24,
        "Parch": [0, 0, 1] * 16,
        "Fare": np.linspace(8, 100, 48),
        "Embarked": ["S", "C", "Q"] * 16,
        "Survived": [1, 0] * 24,
    })
    _, metrics = train_titanic(data)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["test_rows"] == 10
