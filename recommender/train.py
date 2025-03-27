"""
Script to train and test the hybrid recommender system.
"""

import logging
import os
from .hybrid_recommender import HybridRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_recommender(data_path: str, model_path: str) -> HybridRecommender:
    """
    Train the hybrid recommender system.
    
    Args:
        data_path: Path to the POIs dataset
        model_path: Path to save the trained models
        
    Returns:
        Trained HybridRecommender instance
    """
    # Initialize recommender
    recommender = HybridRecommender(n_clusters=8)
    
    # Load and preprocess data
    df = recommender.preprocess_data(data_path)
    logger.info(f"Loaded {len(df)} POIs")
    
    # Extract features and train models
    features = recommender.extract_features(df)
    logger.info(f"Extracted {features.shape[1]} features")
    
    recommender.train_clustering()
    logger.info("Trained clustering model")
    
    recommender.train_content_based()
    logger.info("Trained content-based filtering model")
    
    # Save models
    recommender.save_models(model_path)
    logger.info(f"Saved models to {model_path}")
    
    return recommender

def test_recommendations(recommender: HybridRecommender) -> None:
    """Test the recommender with sample queries."""
    # Test case 1: Museum lover in Amsterdam who likes to walk
    test_cases = [
        {
            "name": "Museum lover in Amsterdam",
            "preferences": {
                "type": "museum",
                "keywords": "art history culture",
                "mobility": {
                    "mode": "walking",
                    "max_distance": 2.0
                }
            },
            "location": (52.3676, 4.9041)  # Amsterdam coordinates
        },
        {
            "name": "Cyclist looking for parks",
            "preferences": {
                "type": "park",
                "keywords": "nature outdoor bicycle",
                "mobility": {
                    "mode": "cycling",
                    "max_distance": 5.0
                }
            },
            "location": (52.0907, 5.1214)  # Utrecht coordinates
        }
    ]
    
    for case in test_cases:
        logger.info(f"\nTesting: {case['name']}")
        recommendations = recommender.get_recommendations(
            case["preferences"],
            case["location"]
        )
        
        logger.info("Top recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"{i}. {rec['name']} ({rec['type']}) - "
                f"Score: {rec['score']:.3f}"
            )

def main():
    # Paths
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "Databases for training",
        "gtfs_nl",
        "pois_with_mobility.csv"
    )
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models",
        "hybrid_recommender.joblib"
    )
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Train recommender
    recommender = train_recommender(data_path, model_path)
    
    # Test recommendations
    test_recommendations(recommender)

if __name__ == "__main__":
    main()
