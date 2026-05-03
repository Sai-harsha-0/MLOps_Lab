import pandas as pd
from src.train import KNNRecommendationModel

model = KNNRecommendationModel.load("models/model.pkl")

df = pd.read_csv("data/processed/ratings_clean.csv")

print("\nPredictions:\n")

for _, row in df.head(5).iterrows():
    pred = model.predict_rating(int(row['user_id']), int(row['movie_id']))
    print(f"User {row['user_id']} Movie {row['movie_id']} → Pred: {pred:.2f} | Actual: {row['rating']}")