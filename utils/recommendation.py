import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from models.models import Destination, UserPreference, Review
from typing import List, Dict

class RecommendationEngine:
    def __init__(self):
        self.kmeans = None
        self.destination_features = None
        self.destination_ids = None

    def _prepare_destination_features(self, destinations: List[Destination]) -> np.ndarray:
        """
        Convert destination data into feature vectors for clustering and similarity calculations
        """
        features = []
        self.destination_ids = []
        
        for dest in destinations:
            # Create feature vector from destination attributes
            feature_vector = [
                dest.average_rating,
                len(dest.activities),
                len(dest.accessibility_features),
                # Add more features as needed
            ]
            
            # Extend feature vector with one-hot encoded activities
            activities_encoded = self._one_hot_encode(dest.activities, all_possible_activities)
            feature_vector.extend(activities_encoded)
            
            features.append(feature_vector)
            self.destination_ids.append(dest.id)
            
        return np.array(features)

    def train_clusters(self, destinations: List[Destination], n_clusters: int = 5):
        """
        Train K-means clustering on destination features
        """
        self.destination_features = self._prepare_destination_features(destinations)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(self.destination_features)

    def get_similar_destinations(self, destination_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Find similar destinations using cosine similarity
        """
        if self.destination_features is None:
            raise ValueError("Model not trained. Call train_clusters first.")

        # Get index of target destination
        target_idx = self.destination_ids.index(destination_id)
        target_features = self.destination_features[target_idx].reshape(1, -1)

        # Calculate similarity scores
        similarities = cosine_similarity(target_features, self.destination_features)[0]
        
        # Get indices of most similar destinations (excluding the target)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        # Return similar destination IDs and their similarity scores
        return [
            {
                'destination_id': self.destination_ids[idx],
                'similarity_score': float(similarities[idx])
            }
            for idx in similar_indices
        ]

    def get_personalized_recommendations(self, user_preferences: UserPreference, n_recommendations: int = 5) -> List[Dict]:
        """
        Generate personalized recommendations based on user preferences
        """
        if self.destination_features is None:
            raise ValueError("Model not trained. Call train_clusters first.")

        # Create user preference vector
        user_vector = self._create_user_preference_vector(user_preferences)
        
        # Calculate similarity between user preferences and destinations
        similarities = cosine_similarity(user_vector.reshape(1, -1), self.destination_features)[0]
        
        # Get indices of top recommendations
        recommendation_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        # Return recommended destination IDs and their scores
        return [
            {
                'destination_id': self.destination_ids[idx],
                'score': float(similarities[idx])
            }
            for idx in recommendation_indices
        ]

    @staticmethod
    def _one_hot_encode(items: List[str], all_possible_items: List[str]) -> List[int]:
        """
        Convert a list of items into one-hot encoded vector
        """
        return [1 if item in items else 0 for item in all_possible_items]

    @staticmethod
    def _create_user_preference_vector(preferences: UserPreference) -> np.ndarray:
        """
        Convert user preferences into a feature vector
        """
        # Create feature vector based on user preferences
        vector = []
        
        # Add budget preference (normalized)
        budget_mapping = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
        vector.append(budget_mapping.get(preferences.budget_range, 0.5))
        
        # Add activity preferences
        activities_encoded = RecommendationEngine._one_hot_encode(
            preferences.preferred_activities,
            all_possible_activities
        )
        vector.extend(activities_encoded)
        
        # Add more preference features as needed
        
        return np.array(vector)

# Define global variables
all_possible_activities = [
    'sightseeing', 'nature', 'adventure', 'culture', 'relaxation',
    'food', 'shopping', 'nightlife', 'history', 'sports'
]
