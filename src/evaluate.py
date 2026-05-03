import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_rating_prediction(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    errors = np.abs(y_true - y_pred)

    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mean_error": float(errors.mean()),
        "max_error": float(errors.max()),
        "min_error": float(errors.min())
    }


def compute_coverage(model, test_df):
    all_movies = test_df["movie_id"].unique()
    recommended = set()

    for movie in all_movies:
        users = test_df[test_df["movie_id"] == movie]["user_id"].unique()
        for u in users:
            pred = model.predict_rating(int(u), int(movie))
            if pred >= 3.0:
                recommended.add(int(movie))
                break

    coverage = len(recommended) / len(all_movies)

    logger.info(f"Coverage: {coverage:.2%}")

    return {
        "coverage_ratio": float(coverage),
        "recommended": len(recommended),
        "total": len(all_movies)
    }


def analyze_sparsity(df, n_movies):
    n_users = df["user_id"].nunique()
    n_ratings = len(df)

    max_possible = n_users * n_movies
    density = n_ratings / max_possible
    sparsity = 1 - density

    logger.info(f"Sparsity: {sparsity:.2%}")

    return {
        "users": n_users,
        "movies": n_movies,
        "ratings": n_ratings,
        "density": float(density),
        "sparsity": float(sparsity)
    }


def compute_baseline(y_true):
    baseline = np.full_like(y_true, np.mean(y_true))
    rmse = np.sqrt(mean_squared_error(y_true, baseline))

    logger.info(f"Baseline RMSE: {rmse:.4f}")

    return {"baseline_rmse": float(rmse)}