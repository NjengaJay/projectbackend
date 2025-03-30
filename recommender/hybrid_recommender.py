"""
Hybrid Recommender System combining K-Means clustering and content-based filtering
for tourist attraction recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import joblib
from typing import Dict, List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, n_clusters: int = 8):
        """
        Initialize the hybrid recommender system.
        
        Args:
            n_clusters: Number of clusters for K-Means (default: 8)
        """
        self.n_clusters = n_clusters
        
        # Store processed data
        self.pois_df = None
        self.feature_matrix = None
        self.tfidf_matrix = None
        self.cluster_labels = None
        
        # Store feature information
        self.categorical_columns = None
        self.mobility_columns = None
        self.feature_dimensions = None
        
        # Initialize models with explicit parameters
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10  # Explicitly set to avoid warning
        )
        self.feature_scaler = StandardScaler()
        self.location_scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        
    def load_models(self, model_path: str) -> None:
        """
        Load pre-trained models.
        
        Args:
            model_path: Path to model files
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model path {model_path} does not exist")
                return
                
            model_file = model_path
            if not os.path.exists(model_file):
                logger.warning(f"Model file {model_file} does not exist")
                return
                
            models = joblib.load(model_file)
            self.kmeans = models.get("kmeans")
            self.feature_scaler = models.get("feature_scaler")
            self.location_scaler = models.get("location_scaler")
            self.tfidf = models.get("tfidf")
            self.nn_model = models.get("nn_model")
            self.feature_matrix = models.get("feature_matrix")
            self.tfidf_matrix = models.get("tfidf_matrix")
            self.cluster_labels = models.get("cluster_labels")
            
        except Exception as e:
            logger.warning(f"Could not load models: {str(e)}")
            
    def preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Preprocess the POIs dataset.
        
        Args:
            data_path: Path to the POIs CSV file
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Loading and preprocessing POIs data...")
        
        try:
            # Load data
            logger.info(f"Attempting to load data from: {data_path}")
            logger.info(f"File exists: {os.path.exists(data_path)}")
            logger.info(f"Absolute path: {os.path.abspath(data_path)}")
            
            df = pd.read_csv(data_path)
            logger.info(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Available columns: {sorted(df.columns.tolist())}")
            
            # Handle missing values for core features
            core_features = ['opening_hours', 'wheelchair', 'tourism']
            for feature in core_features:
                if feature in df.columns:
                    df[feature] = df[feature].fillna('unknown')
                    logger.info(f"Found column {feature} with {df[feature].nunique()} unique values")
                else:
                    df[feature] = 'unknown'
                    logger.warning(f"Column {feature} not found in dataset, using default value")
            
            # Create feature text for TF-IDF
            df['feature_text'] = df.apply(lambda row: ' '.join(
                str(row[col]) for col in ['tourism', 'name', 'city']
                if col in df.columns and pd.notna(row.get(col, None))
            ), axis=1)
            
            logger.info(f"Sample feature text: {df['feature_text'].iloc[0]}")
            
            self.pois_df = df
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}", exc_info=True)
            raise
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and scale features for clustering.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Scaled feature matrix
        """
        logger.info("Extracting and scaling features...")
        features_list = []
        feature_dimensions = []
        
        # Scale location features if available
        if 'latitude' in df.columns and 'longitude' in df.columns:
            location_features = self.location_scaler.fit_transform(
                df[['latitude', 'longitude']]
            )
            features_list.append(location_features)
            feature_dimensions.append(('location', location_features.shape[1]))
        
        # Create dummy variables for available categorical features
        categorical_cols = [col for col in ['tourism', 'wheelchair', 'opening_hours'] 
                          if col in df.columns]
        if categorical_cols:
            categorical_features = pd.get_dummies(
                df[categorical_cols],
                prefix=categorical_cols
            )
            self.categorical_columns = categorical_features.columns.tolist()
            features_list.append(categorical_features.values)
            feature_dimensions.append(('categorical', categorical_features.shape[1]))
        
        # Add mobility features if available
        mobility_features = []
        for col in df.columns:
            if any(term in col.lower() for term in ['distance', 'duration', 'trips']):
                mobility_features.append(col)
        
        if mobility_features:
            logger.info(f"Using mobility features: {mobility_features}")
            self.mobility_columns = mobility_features
            mobility_data = df[mobility_features].fillna(0)
            mobility_scaled = StandardScaler().fit_transform(mobility_data)
            features_list.append(mobility_scaled)
            feature_dimensions.append(('mobility', mobility_scaled.shape[1]))
        
        # Combine all available features
        if not features_list:
            raise ValueError("No valid features found in the dataset")
        
        feature_matrix = np.hstack(features_list)
        self.feature_dimensions = feature_dimensions
        
        # Log feature dimensions
        logger.info("Feature matrix dimensions:")
        for name, dim in feature_dimensions:
            logger.info(f"- {name}: {dim} features")
        logger.info(f"Total features: {feature_matrix.shape[1]}")
        
        # Scale the combined features
        self.feature_matrix = self.feature_scaler.fit_transform(feature_matrix)
        return self.feature_matrix
        
    def train_clustering(self) -> None:
        """Train the clustering model."""
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Call extract_features first.")
        
        logger.info("Training clustering model...")
        self.cluster_labels = self.kmeans.fit_predict(self.feature_matrix)
        self.pois_df['cluster'] = self.cluster_labels
        
        # Log cluster distribution
        cluster_sizes = pd.Series(self.cluster_labels).value_counts()
        logger.info("Cluster distribution:")
        for cluster, size in cluster_sizes.items():
            logger.info(f"Cluster {cluster}: {size} POIs")
            
    def train_content_based(self) -> None:
        """Train the content-based filtering model."""
        if self.pois_df is None:
            raise ValueError("No data available. Call preprocess_data first.")
        
        logger.info("Training content-based filtering model...")
        self.tfidf_matrix = self.tfidf.fit_transform(self.pois_df['feature_text'])
        self.nn_model.fit(self.tfidf_matrix)
        
        # Log vocabulary size
        logger.info(f"TF-IDF vocabulary size: {len(self.tfidf.vocabulary_)}")
        
    def fit(self, data_path: str) -> None:
        """Fit all models in one go."""
        logger.info("Starting model training...")
        
        # Load and preprocess data
        df = self.preprocess_data(data_path)
        logger.info(f"Preprocessed {len(df)} POIs")
        
        # Extract and scale features
        features = self.extract_features(df)
        logger.info(f"Extracted {features.shape[1]} features")
        
        # Train models
        self.train_clustering()
        self.train_content_based()
        
        logger.info("Model training complete")
        
    def get_recommendations(
        self, 
        user_preferences: Dict,
        current_location: Tuple[float, float],
        n_recommendations: int = 10
    ) -> List[Dict]:
        """
        Get POI recommendations based on user preferences and location.
        
        Args:
            user_preferences: Dictionary containing user preferences
            current_location: Tuple of (latitude, longitude)
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended POIs
        """
        logger.info("Getting recommendations...")
        logger.info(f"User preferences: {user_preferences}")
        logger.info(f"Current location: {current_location}")
        
        if self.pois_df is None or self.feature_matrix is None:
            raise ValueError("Models not trained. Call fit() first.")
        
        try:
            # Get relevant cluster
            target_cluster = self._get_location_based_cluster(current_location)
            logger.info(f"Selected cluster: {target_cluster}")
            
            # Filter POIs by cluster
            cluster_pois = self.pois_df[
                self.pois_df['cluster'] == target_cluster
            ].copy()
            
            logger.info(f"Found {len(cluster_pois)} POIs in cluster")
            
            # Apply mobility filtering if preferences provided
            if 'mobility' in user_preferences:
                filtered_pois = self._apply_mobility_filter(
                    user_preferences['mobility']
                )
                cluster_pois = cluster_pois[
                    cluster_pois.index.isin(filtered_pois.index)
                ]
                logger.info(f"After mobility filtering: {len(cluster_pois)} POIs")
            
            if len(cluster_pois) == 0:
                logger.warning("No POIs found after filtering")
                return []
            
            # Get content-based recommendations
            recommendations = self._get_content_based_recommendations(
                cluster_pois,
                user_preferences,
                current_location,
                n_recommendations
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
        
    def _apply_mobility_filter(self, mobility_prefs: Dict) -> pd.DataFrame:
        """Apply mobility-based filtering to POIs."""
        filtered_df = self.pois_df.copy()
        
        mode = mobility_prefs.get('mode', 'walking')
        max_distance = mobility_prefs.get('max_distance', 5.0)
        
        # Find relevant distance columns based on transport mode
        distance_cols = [col for col in filtered_df.columns 
                        if 'distance' in col.lower()]
        
        if distance_cols:
            # Use the minimum distance across all available transport modes
            filtered_df['min_distance'] = filtered_df[distance_cols].min(axis=1)
            filtered_df = filtered_df[
                filtered_df['min_distance'] <= max_distance
            ]
        
        # Apply mode-specific filtering if relevant columns exist
        mode_cols = {
            'walking': 'walking',
            'cycling': 'cycling',
            'public_transport': ['bus', 'tram', 'train']
        }
        
        if mode in mode_cols:
            mode_terms = mode_cols[mode]
            if isinstance(mode_terms, str):
                mode_terms = [mode_terms]
                
            # Find columns related to the transport mode
            mode_cols = []
            for term in mode_terms:
                mode_cols.extend([
                    col for col in filtered_df.columns
                    if term in col.lower()
                ])
            
            if mode_cols:
                logger.info(f"Filtering by transport mode columns: {mode_cols}")
                # Keep POIs that have some activity for the chosen mode
                filtered_df = filtered_df[
                    filtered_df[mode_cols].sum(axis=1) > 0
                ]
        
        return filtered_df
        
    def _get_location_based_cluster(
        self, 
        location: Tuple[float, float]
    ) -> int:
        """Get the most appropriate cluster based on location."""
        # Create feature matrix for the location
        data = {
            'location': location,
            'tourism': 'unknown',
            'wheelchair': 'unknown',
            'opening_hours': 'unknown'
        }
        feature_matrix = self._create_feature_matrix(data)
        
        # Predict cluster
        return self.kmeans.predict(feature_matrix)[0]
        
    def _get_type_based_cluster(self, poi_type: str) -> int:
        """Get the most common cluster for a POI type."""
        type_clusters = self.pois_df[
            self.pois_df['tourism'] == poi_type  
        ]['cluster'].value_counts()
        
        return type_clusters.index[0] if not type_clusters.empty else 0
        
    def _get_content_based_recommendations(
        self,
        pois_df: pd.DataFrame,
        user_preferences: Dict,
        current_location: Optional[Tuple[float, float]],
        n_recommendations: int = 10
    ) -> List[Dict]:
        """Get content-based recommendations from a set of POIs."""
        if len(pois_df) == 0:
            logger.warning("No POIs available for content-based filtering")
            return []
            
        # Get content similarity scores
        if 'keywords' in user_preferences:
            content_scores = self._get_content_similarity(
                user_preferences['keywords'],
                pois_df
            )
        else:
            content_scores = np.ones(len(pois_df))
            
        # Get location similarity scores
        if current_location:
            location_scores = self._get_location_similarity(
                current_location,
                pois_df
            )
        else:
            location_scores = np.ones(len(pois_df))
            
        # Combine scores
        final_scores = 0.6 * content_scores + 0.4 * location_scores
        
        # Sort POIs by score
        pois_df = pois_df.copy()
        pois_df['score'] = final_scores
        recommendations = pois_df.nlargest(n_recommendations, 'score')
        
        # Format recommendations
        return [
            {
                'id': poi.name,
                'name': poi['name'],
                'type': poi['tourism'],
                'location': {
                    'latitude': poi['latitude'],
                    'longitude': poi['longitude']
                },
                'score': poi['score']
            }
            for _, poi in recommendations.iterrows()
        ]
        
    def _get_content_similarity(
        self,
        query: str,
        pois_df: pd.DataFrame
    ) -> np.ndarray:
        """Calculate content similarity scores."""
        # Transform query to TF-IDF space
        query_vector = self.tfidf.transform([query])
        
        # Get similarity scores
        if len(pois_df) > 0:
            poi_vectors = self.tfidf.transform(pois_df['feature_text'])
            similarities = cosine_similarity(query_vector, poi_vectors)[0]
        else:
            similarities = np.array([])
            
        return similarities
        
    def _get_location_similarity(
        self,
        current_location: Tuple[float, float],
        pois_df: pd.DataFrame
    ) -> np.ndarray:
        """Calculate location similarity scores."""
        if len(pois_df) == 0:
            return np.array([])
            
        # Calculate Haversine distances
        lat1, lon1 = current_location
        lat2 = pois_df['latitude'].values
        lon2 = pois_df['longitude'].values
        
        R = 6371  # Earth's radius in kilometers
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2) * np.sin(dlat/2) +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon/2) * np.sin(dlon/2))
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distances = R * c
        
        # Convert distances to similarity scores (inverse of distance)
        # Add small constant to avoid division by zero
        similarities = 1 / (distances + 0.1)
        
        # Normalize to [0,1]
        if len(similarities) > 0:
            similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        
        return similarities

    def _calculate_location_scores(
        self,
        pois: pd.DataFrame,
        current_location: Tuple[float, float]
    ) -> np.ndarray:
        """Calculate normalized distance-based scores."""
        distances = np.sqrt(
            (pois['latitude'] - current_location[0])**2 +
            (pois['longitude'] - current_location[1])**2
        )
        return 1 / (1 + distances)

    def _create_feature_matrix(self, data: Dict) -> np.ndarray:
        """Create a feature matrix for a single data point."""
        features_list = []
        feature_dimensions = []
        
        # Add location features
        if 'location' in data:
            location = np.array(data['location']).reshape(1, -1)
            location_scaled = self.location_scaler.transform(location)
            features_list.append(location_scaled)
            feature_dimensions.append(('location', location_scaled.shape[1]))
        
        # Add categorical features
        if self.categorical_columns:
            logger.info(f"Creating categorical features using columns: {self.categorical_columns}")
            # Create a DataFrame with one row
            df = pd.DataFrame([data], columns=['tourism', 'wheelchair', 'opening_hours'])
            df = df.fillna('unknown')
            
            # Create dummy variables using the same columns as training
            cat_features = pd.get_dummies(df)
            logger.info(f"Generated dummy columns: {cat_features.columns.tolist()}")
            
            # Ensure all training columns are present
            for col in self.categorical_columns:
                if col not in cat_features:
                    cat_features[col] = 0
                    
            # Reorder columns to match training
            cat_features = cat_features[self.categorical_columns]
            features_list.append(cat_features.values)
            feature_dimensions.append(('categorical', cat_features.shape[1]))
        
        # Add mobility features
        if self.mobility_columns:
            logger.info(f"Adding mobility features: {self.mobility_columns}")
            mobility_data = np.zeros((1, len(self.mobility_columns)))
            features_list.append(mobility_data)
            feature_dimensions.append(('mobility', len(self.mobility_columns)))
        
        # Combine features
        feature_matrix = np.hstack(features_list)
        
        # Log dimensions
        logger.info("Feature matrix dimensions:")
        for name, dim in feature_dimensions:
            logger.info(f"- {name}: {dim} features")
        logger.info(f"Total features: {feature_matrix.shape[1]}")
        
        # Scale features
        return self.feature_scaler.transform(feature_matrix)

    def save_models(self, path: str) -> None:
        """Save trained models and scalers."""
        logger.info(f"Saving models to {path}...")
        
        joblib.dump({
            'kmeans': self.kmeans,
            'feature_scaler': self.feature_scaler,
            'location_scaler': self.location_scaler,
            'tfidf': self.tfidf,
            'nn_model': self.nn_model,
            'feature_matrix': self.feature_matrix,
            'tfidf_matrix': self.tfidf_matrix,
            'cluster_labels': self.cluster_labels
        }, path)
        
    def load_models(self, path: str) -> None:
        """Load trained models and scalers."""
        logger.info(f"Loading models from {path}...")
        
        models = joblib.load(path)
        self.kmeans = models['kmeans']
        self.feature_scaler = models['feature_scaler']
        self.location_scaler = models['location_scaler']
        self.tfidf = models['tfidf']
        self.nn_model = models['nn_model']
        self.feature_matrix = models['feature_matrix']
        self.tfidf_matrix = models['tfidf_matrix']
        self.cluster_labels = models['cluster_labels']
