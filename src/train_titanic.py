"""Train and evaluate a Logistic Regression model for the Kaggle Titanic dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_INPUT = Path("data/titanic/train.csv")
DEFAULT_MODEL = Path("models/titanic_logistic_regression.joblib")
DEFAULT_METRICS = Path("reports/titanic_metrics.json")
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def build_model() -> Pipeline:
    """Return a preprocessing and logistic-regression pipeline."""
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("logistic", LogisticRegression(max_iter=1000))])


def train_and_evaluate(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Fit on a stratified split and return fitted model, predictions, and metrics."""
    missing = set(FEATURES + ["Survived"]) - set(data.columns)
    if missing:
        raise ValueError(f"The Titanic data is missing columns: {', '.join(sorted(missing))}")

    x = data[FEATURES]
    y = data["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to train.csv")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    model, metrics = train_and_evaluate(data, args.test_size, args.random_state)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Classification report:")
    print(pd.DataFrame(metrics["classification_report"]).transpose().to_string())
    print(f"Saved model: {args.model_output}")


if __name__ == "__main__":
    main()
