"""Train and evaluate a Linear Regression model for the Kaggle House Prices dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_INPUT = Path("data/house_prices/train.csv")
DEFAULT_MODEL = Path("models/house_price_linear_regression.joblib")
DEFAULT_METRICS = Path("reports/house_price_metrics.json")
DEFAULT_FIGURE = Path("reports/figures/house_price_predicted_vs_actual.png")


def build_model() -> Pipeline:
    """Return a numeric preprocessing and linear-regression pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regression", LinearRegression()),
        ]
    )


def train_and_evaluate(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Fit on a train split and return the fitted model, test targets, predictions, and metrics."""
    if "SalePrice" not in data.columns:
        raise ValueError("The House Prices data must include a 'SalePrice' column.")

    features = data.drop(columns=["SalePrice"]).select_dtypes(include="number")
    if features.empty:
        raise ValueError("No numeric predictor columns were found.")

    target = data["SalePrice"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "r2_score": float(r2_score(y_test, predictions)),
        "mean_absolute_error": float(mean_absolute_error(y_test, predictions)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "numeric_features": int(features.shape[1]),
    }
    return model, y_test, predictions, metrics


def save_plot(actual, predicted, output_path: Path) -> None:
    """Save a predicted-versus-actual scatter plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.65, color="steelblue", edgecolors="none")
    plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("House Prices: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to train.csv")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    model, actual, predicted, metrics = train_and_evaluate(data, args.test_size, args.random_state)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plot(actual, predicted, args.figure_output)

    print(f"R2 score: {metrics['r2_score']:.3f}")
    print(f"MAE: {metrics['mean_absolute_error']:.2f}")
    print(f"Saved model: {args.model_output}")
    print(f"Saved plot: {args.figure_output}")


if __name__ == "__main__":
    main()
