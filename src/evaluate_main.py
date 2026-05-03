import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.evaluate import (
    evaluate_rating_prediction,
    compute_coverage,
    analyze_sparsity,
    compute_baseline
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== EVALUATION START ===")

    # Load model
    model = joblib.load("models/model.pkl")

    # Load data
    df = pd.read_csv("data/processed/ratings_clean.csv")

    # Predictions
    y_true = df["rating"].values
    y_pred = model.predict_batch(df)

    # Metrics
    rating_metrics = evaluate_rating_prediction(y_true, y_pred)
    coverage = compute_coverage(model, df)
    sparsity = analyze_sparsity(df, 100)
    baseline = compute_baseline(y_true)

    report = {
        "rating": rating_metrics,
        "coverage": coverage,
        "sparsity": sparsity,
        "baseline": baseline
    }

    Path("evaluations").mkdir(exist_ok=True)

    with open("evaluations/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("✓ Evaluation Done!")


if __name__ == "__main__":
    main()