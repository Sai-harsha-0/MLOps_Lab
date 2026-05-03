import os
import argparse
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

# =========================
# ENV VARIABLES
# =========================
STUDENT_NAME = os.getenv("STUDENT_NAME", "Future MLOps Engineer")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/rating_features.pkl")

# =========================
# FASTAPI INIT
# =========================
app = FastAPI(
    title="MovieLens Recommender API",
    description=f"Built by {STUDENT_NAME}",
    version="1.0"
)

# Load model + features once
try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    features = None


# =========================
# HEALTH ENDPOINT
# =========================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "student": STUDENT_NAME
    }


# =========================
# RECOMMEND ENDPOINT
# =========================
@app.get("/recommend")
def recommend(user_id: int, n: int = 5):

    try:
        if model is None or features is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if user_id not in list(features.user_ids):
            raise HTTPException(status_code=404, detail="User not found")

        predictions = []

        for movie_id in features.movie_ids:
            try:
                pred = model.predict_rating(int(user_id), int(movie_id))
                if pred is not None:
                    predictions.append((movie_id, pred))
            except Exception:
                continue

        if len(predictions) == 0:
            raise HTTPException(status_code=500, detail="No predictions generated")

        predictions.sort(key=lambda x: x[1], reverse=True)

        top_n = predictions[:n]

        return {
            "user_id": user_id,
            "recommendations": [
                {
                    "movie_id": int(mid),
                    "predicted_rating": float(score),
                    "rank": i + 1
                }
                for i, (mid, score) in enumerate(top_n)
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# =========================
# BATCH PREDICT
# =========================
@app.post("/predict_batch")
def predict_batch(data: dict):

    if model is None or features is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    try:
        for item in data.get("predictions", []):
            user_id = item.get("user_id")
            movie_id = item.get("movie_id")

            # validate inputs
            if user_id not in list(features.user_ids):
                continue
            if movie_id not in list(features.movie_ids):
                continue

            try:
                pred = model.predict_rating(int(user_id), int(movie_id))
                results.append({
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "predicted_rating": float(pred)
                })
            except Exception:
                continue

        return {
            "predictions": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# CLI SUPPORT (YOUR OLD CODE)
# =========================
def cli():
    parser = argparse.ArgumentParser(description="CLI mode")
    parser.add_argument("--message", type=str, default="Hello MLOps!")
    args = parser.parse_args()

    print("========================================")
    print(f"Message: {args.message}")
    print(f"Brought to you by: {STUDENT_NAME}")
    print("========================================")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    import sys

    # If arguments passed → CLI mode
    if len(sys.argv) > 1:
        cli()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)