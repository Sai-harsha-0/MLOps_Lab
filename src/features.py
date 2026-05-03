import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging
from typing import Tuple, List, Dict
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RatingFeatures:
    """Compute user similarity features."""

    def __init__(self):
        self.ratings_matrix = None
        self.similarity_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.fitted = False

    def fit(self, ratings_df: pd.DataFrame):
        if ratings_df.empty:
            raise ValueError("Empty dataframe")

        # Pivot table (user × movie)
        self.ratings_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0.0
        )

        self.user_ids = self.ratings_matrix.index.values
        self.movie_ids = self.ratings_matrix.columns.values

        logger.info(f"Matrix shape: {self.ratings_matrix.shape}")

        # Cosine similarity
        self.similarity_matrix = cosine_similarity(self.ratings_matrix)

        # Remove self similarity
        np.fill_diagonal(self.similarity_matrix, 0)

        self.fitted = True
        logger.info("✓ Features fitted")

        return self

    def get_similar_users(self, user_id: int, n: int = 5):
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        indices = np.where(self.user_ids == user_id)[0]

        if len(indices) == 0:
            return []

        user_idx = indices[0]
        similarities = self.similarity_matrix[user_idx]

        top_indices = np.argsort(similarities)[-n:][::-1]

        result = []
        for idx in top_indices:
            if similarities[idx] > 0:
                result.append((int(self.user_ids[idx]), float(similarities[idx])))

        return result

    def get_user_ratings_vector(self, user_id: int):
        if user_id not in self.user_ids:
            raise ValueError("User not found")

        idx = np.where(self.user_ids == user_id)[0][0]
        return self.ratings_matrix.iloc[idx].values

    def get_movie_rating_stats(self):
        vals = self.ratings_matrix[self.ratings_matrix > 0].values.flatten()

        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved to {path}")

    @staticmethod
    def load(path: str):
        return joblib.load(path)