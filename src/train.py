import joblib
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KNNRecommendationModel:

    def __init__(self, k=5, use_similarity_weights=False):
        self.k = k
        self.use_similarity_weights = use_similarity_weights
        self.features = None
        self.ratings_df = None
        self.default_rating = 3.0
        self.fitted = False

    def fit(self, features, ratings_df):
        self.features = features
        self.ratings_df = ratings_df
        self.fitted = True
        return self

    def predict_rating(self, user_id, movie_id):
        similar_users = self.features.get_similar_users(user_id, self.k)

        if not similar_users:
            return self.default_rating

        ratings = []
        sims = []

        for u, s in similar_users:
            r = self.ratings_df[
                (self.ratings_df['user_id'] == u) &
                (self.ratings_df['movie_id'] == movie_id)
            ]['rating'].values

            if len(r) > 0:
                ratings.append(r[0])
                sims.append(s)

        if len(ratings) == 0:
            return self.default_rating

        if self.use_similarity_weights:
            sims = np.array(sims)
            weights = sims / np.sum(sims)
            pred = np.sum(np.array(ratings) * weights)
        else:
            pred = np.mean(ratings)

        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, df):
        preds = []
        for _, row in df.iterrows():
            preds.append(self.predict_rating(int(row['user_id']), int(row['movie_id'])))
        return np.array(preds)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)