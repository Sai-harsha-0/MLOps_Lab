import pandas as pd
import numpy as np
import json
import joblib
from src.train import KNNRecommendationModel

print("="*60)
print("TRAINING MODEL")
print("="*60)

# Load data
ratings = pd.read_csv("data/processed/ratings_clean.csv")
features = joblib.load("models/rating_features.pkl")

# Split
train = ratings.sample(frac=0.8, random_state=42)
test = ratings.drop(train.index)

print(f"Train: {len(train)}, Test: {len(test)}")

# Train model
model = KNNRecommendationModel(k=5)
model.fit(features, train)

# Evaluate
preds = model.predict_batch(test)
y_true = test['rating'].values

rmse = np.sqrt(np.mean((y_true - preds) ** 2))
mae = np.mean(np.abs(y_true - preds))

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Save model
model.save("models/model.pkl")

# Save metadata
meta = {
    "k": 5,
    "rmse": float(rmse),
    "mae": float(mae)
}

with open("models/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Model saved")