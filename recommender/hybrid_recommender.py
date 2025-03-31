"""
Hybrid Recommender System combining K-Means clustering and content-based filtering
for tourist attraction recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
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
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, batch_size=1000, random_state=42)
        self.feature_scaler = StandardScaler()
        self.location_scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        
        # Store processed data
        self.pois_df = None
        self.feature_matrix = None
        self.tfidf_matrix = None
        self.cluster_labels = None
        
        # Store feature information
        self.categorical_columns = None
        self.mobility_columns = None
        self.feature_dimensions = None
        
        # Initialize models
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, n_init=3, batch_size=1000, random_state=42)
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
            self.pois_df = models.get("pois_df")
            
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
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
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
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and scale features."""
        logger.info("Extracting and scaling features...")
        
        # Store the original DataFrame
        self.pois_df = df.copy()
        
        # Create feature text for TF-IDF using only available columns
        text_columns = ['tourism', 'name']  # Only use guaranteed columns
        df['feature_text'] = df.apply(lambda row: ' '.join(
            str(row[col]) for col in text_columns
            if pd.notna(row.get(col, None))
        ), axis=1)
        logger.info("Created feature text")
        
        # Fit and transform TF-IDF
        self.tfidf_matrix = self.tfidf.fit_transform(df['feature_text'])
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Use TF-IDF matrix as the feature matrix for clustering
        self.feature_matrix = self.tfidf_matrix.toarray()
        logger.info(f"Final feature matrix shape: {self.feature_matrix.shape}")
        
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
        """Train the recommender system on POIs data."""
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} POIs")
            
            # Add city column if not present
            if 'city' not in df.columns:
                logger.info("Adding city column with default value 'amsterdam'")
                df['city'] = 'amsterdam'  # Default to Amsterdam for now
            
            # Store the original DataFrame
            self.pois_df = df.copy()
            logger.info(f"Columns in DataFrame: {df.columns.tolist()}")
            
            # Extract features
            logger.info("Extracting features...")
            self.feature_matrix = self.extract_features(df)
            
            # Train clustering
            logger.info("Training clustering model...")
            self.cluster_labels = self.kmeans.fit_predict(self.feature_matrix)
            logger.info(f"Trained clustering model with {self.n_clusters} clusters")
            
            # Log cluster sizes
            for i in range(self.n_clusters):
                cluster_size = sum(self.cluster_labels == i)
                logger.info(f"Cluster {i}: {cluster_size} POIs")
                
        except Exception as e:
            logger.error(f"Error in fit: {str(e)}", exc_info=True)
            raise
        
    def get_pois_for_location(self, location: str) -> pd.DataFrame:
        """Get POIs for a specific location."""
        try:
            if self.pois_df is None:
                logger.error("POIs DataFrame is None")
                return pd.DataFrame()
                
            # Case-insensitive location matching
            location = location.lower()
            logger.info(f"Looking for POIs in location: {location}")
            logger.info(f"Available cities: {self.pois_df['city'].unique().tolist()}")
            
            # Try exact match first
            location_mask = self.pois_df['city'].str.lower() == location
            pois = self.pois_df[location_mask].copy()
            
            # If no exact match, try substring match
            if len(pois) == 0:
                logger.info("No exact match found, trying substring match")
                location_mask = self.pois_df['city'].str.lower().str.contains(location)
                pois = self.pois_df[location_mask].copy()
            
            logger.info(f"Location search results:")
            logger.info(f"- Total POIs: {len(self.pois_df)}")
            logger.info(f"- Location: {location}")
            logger.info(f"- Matching POIs: {len(pois)}")
            
            # Log some sample POIs if available
            if len(pois) > 0:
                logger.info("Sample POIs:")
                for _, poi in pois.head(3).iterrows():
                    logger.info(f"- {poi['name']} ({poi['tourism']})")
            
            return pois
            
        except Exception as e:
            logger.error(f"Error getting POIs for location: {str(e)}", exc_info=True)
            return pd.DataFrame()
            
    def get_recommendations(
        self,
        location: str,
        user_preferences: Dict,
        current_location: Optional[Tuple[float, float]] = None,
        n_recommendations: int = 10
    ) -> List[Dict]:
        """Get recommendations for a location."""
        try:
            logger.info(f"Getting recommendations for {location} with preferences {user_preferences}")
            
            # Get POIs for the location
            pois = self.get_pois_for_location(location)
            logger.info(f"Found {len(pois)} POIs for location {location}")
            
            if len(pois) == 0:
                logger.warning(f"No POIs found for location: {location}")
                return []
            
            # Get content-based recommendations
            recommendations = self._get_content_based_recommendations(
                pois,
                user_preferences,
                current_location,
                n_recommendations
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            return []
            
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
        current_location: Tuple[float, float],
        n_recommendations: int
    ) -> List[Dict]:
        """Get content-based recommendations."""
        try:
            logger.info("Getting content-based recommendations...")
            logger.info(f"Input POIs shape: {pois_df.shape}")
            
            # Create feature text for the query
            interests = user_preferences.get('interests', [])
            query_text = ' '.join(interests)
            logger.info(f"Query text: {query_text}")
            
            # Filter POIs by tourism type if museums are requested
            if 'museum' in interests:
                logger.info("Filtering for museums")
                museum_mask = pois_df['tourism'].str.lower().str.contains('museum', na=False)
                pois_df = pois_df[museum_mask]
                logger.info(f"Found {len(pois_df)} museums")
            
            # Apply accessibility filter if required
            if user_preferences.get('accessibility_required'):
                logger.info("Filtering for wheelchair accessibility")
                wheelchair_mask = pois_df['wheelchair'].str.lower() == 'yes'
                pois_df = pois_df[wheelchair_mask]
                logger.info(f"Found {len(pois_df)} wheelchair accessible POIs")
            
            if len(pois_df) == 0:
                logger.warning("No matching POIs found after filtering")
                return []
            
            # Transform query using TF-IDF
            query_vector = self.tfidf.transform([query_text])
            logger.info(f"Query vector shape: {query_vector.shape}")
            
            # Get TF-IDF vectors for the filtered POIs
            poi_indices = pois_df.index
            poi_vectors = self.tfidf_matrix[poi_indices]
            logger.info(f"POI vectors shape: {poi_vectors.shape}")
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, poi_vectors)
            logger.info(f"Similarities shape: {similarities.shape}")
            
            # Get top N indices
            top_n = min(n_recommendations, len(pois_df))
            top_indices = similarities[0].argsort()[-top_n:][::-1]
            logger.info(f"Found {len(top_indices)} matches")
            
            # Get recommendations
            recommendations = []
            for idx in top_indices:
                poi = pois_df.iloc[idx]
                recommendation = {
                    'name': poi['name'],
                    'type': poi['tourism'],
                    'rating': 4.0,  # Default rating
                    'location': {
                        'latitude': float(poi['latitude']),
                        'longitude': float(poi['longitude'])
                    },
                    'wheelchair_accessible': poi['wheelchair'].lower() == 'yes'
                }
                recommendations.append(recommendation)
            
            logger.info(f"Returning {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {str(e)}", exc_info=True)
            return []
        
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
            'cluster_labels': self.cluster_labels,
            'pois_df': self.pois_df
        }, path)
        
        logger.info("Models saved successfully")
        
    def load_models(self, path: str) -> None:
        """Load trained models and scalers."""
        try:
            logger.info(f"Loading models from {path}")
            models = joblib.load(path)
            logger.info("Successfully loaded models from disk")
            
            # Load each component
            self.kmeans = models.get("kmeans")
            logger.info(f"Loaded kmeans model: {type(self.kmeans)}")
            
            self.feature_scaler = models.get("feature_scaler")
            logger.info(f"Loaded feature_scaler: {type(self.feature_scaler)}")
            
            self.location_scaler = models.get("location_scaler")
            logger.info(f"Loaded location_scaler: {type(self.location_scaler)}")
            
            self.tfidf = models.get("tfidf")
            logger.info(f"Loaded tfidf: {type(self.tfidf)}")
            
            self.nn_model = models.get("nn_model")
            logger.info(f"Loaded nn_model: {type(self.nn_model)}")
            
            self.feature_matrix = models.get("feature_matrix")
            logger.info(f"Loaded feature_matrix shape: {self.feature_matrix.shape if self.feature_matrix is not None else None}")
            
            self.tfidf_matrix = models.get("tfidf_matrix")
            logger.info(f"Loaded tfidf_matrix shape: {self.tfidf_matrix.shape if self.tfidf_matrix is not None else None}")
            
            self.cluster_labels = models.get("cluster_labels")
            logger.info(f"Loaded cluster_labels shape: {self.cluster_labels.shape if self.cluster_labels is not None else None}")
            
            self.pois_df = models.get("pois_df")
            logger.info(f"Loaded pois_df shape: {self.pois_df.shape if self.pois_df is not None else None}")
            
            # Fit location scaler on POIs data if available
            if self.pois_df is not None and 'latitude' in self.pois_df.columns and 'longitude' in self.pois_df.columns:
                logger.info("Fitting location scaler on POIs data")
                self.location_scaler.fit(self.pois_df[['latitude', 'longitude']])
            
        except Exception as e:
            logger.error(f"Could not load models: {str(e)}", exc_info=True)
            raise