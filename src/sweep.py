import mlflow
import pandas as pd
import numpy as np
import joblib
import os

from src.mlflow_tracking import init_experiment, log_params, log_metrics
from src.train import KNNRecommendationModel


def run():
    print("🚀 Starting MLflow sweep...")

    # Initialize MLflow
    init_experiment()

    # Load data
    df = pd.read_csv("data/processed/ratings_clean.csv")
    features = joblib.load("models/rating_features.pkl")

    k_values = [3, 5, 10, 15, 20]

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    for k in k_values:
        print(f"\n🔹 Running K={k}")

        with mlflow.start_run(run_name=f"k={k}"):

            # Train model
            model = KNNRecommendationModel(k=k)
            model.fit(features, df)

            # Predictions
            y_true = df["rating"].values
            y_pred = model.predict_batch(df)

            # Metrics
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))

            # Log to MLflow
            log_params(k)
            log_metrics(rmse, mae)

            # Save model
            model_path = f"models/model_k{k}.pkl"
            model.save(model_path)

            # Log artifact
            mlflow.log_artifact(model_path)

            print(f"✅ K={k} | RMSE={rmse:.3f} | MAE={mae:.3f}")

    print("\n🎯 Sweep completed!")


# 🔥 IMPORTANT: This was missing
if __name__ == "__main__":
    run()