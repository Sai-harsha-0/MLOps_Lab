import pandas as pd
import sys
import logging
from pathlib import Path
from src.features import RatingFeatures

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_ratings_data(df: pd.DataFrame):
    required = {'user_id', 'movie_id', 'rating', 'timestamp'}

    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")

    if df.empty:
        raise ValueError("Empty dataset")

    if df.isnull().any().any():
        raise ValueError("Null values found")

    logger.info("✓ Data validation passed")
    return True


def prepare_features(ratings_path: str, output_dir: str = "models"):
    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 50)

    # Step 1: Load
    logger.info("Loading data...")
    df = pd.read_csv(ratings_path)
    logger.info(f"Loaded {len(df)} rows")

    # Step 2: Validate
    validate_ratings_data(df)

    # Step 3: Fit features
    logger.info("Computing similarity...")
    features = RatingFeatures()
    features.fit(df)

    # Step 4: Save
    Path(output_dir).mkdir(exist_ok=True)
    features.save(f"{output_dir}/rating_features.pkl")

    logger.info("✓ Feature engineering complete")

    return features


if __name__ == "__main__":
    ratings_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/ratings_clean.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "models"

    prepare_features(ratings_path, output_dir)