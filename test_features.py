import numpy as np
from src.features import RatingFeatures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load feature store
logger.info("Loading feature store...")
features = RatingFeatures.load('models/rating_features.pkl')

# Test 1: Similar users
logger.info("\n=== Similar Users ===")
for user_id in features.user_ids[:3]:
    similar = features.get_similar_users(int(user_id), n=3)
    print(f"\nUser {user_id} similar to:")
    for u, s in similar:
        print(f"  User {u} → {s:.3f}")

# Test 2: Shapes
logger.info("\n=== Matrix Info ===")
print("Ratings matrix:", features.ratings_matrix.shape)
print("Similarity matrix:", features.similarity_matrix.shape)

# Test 3: Sparsity
logger.info("\n=== Sparsity ===")
n_ratings = (features.ratings_matrix != 0).sum().sum()
max_possible = features.ratings_matrix.shape[0] * features.ratings_matrix.shape[1]

density = n_ratings / max_possible
print(f"Density: {density:.2%}")
print(f"Sparsity: {1-density:.2%}")

# Test 4: User stats
logger.info("\n=== User Stats ===")
for user_id in features.user_ids[:3]:
    vec = features.get_user_ratings_vector(int(user_id))
    count = np.sum(vec > 0)
    avg = np.mean(vec[vec > 0]) if count > 0 else 0
    print(f"User {user_id}: {count} ratings, avg = {avg:.2f}")

logger.info("\n✓ All tests completed")