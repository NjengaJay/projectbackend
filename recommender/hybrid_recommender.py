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
        
    def load_models(self, path: str) -> bool:
        """
        Load pre-trained models.
        
        Args:
            path: Path to model files
            
        Returns:
            bool: True if models were loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Model path {path} does not exist")
                return False
                
            model_file = path
            if not os.path.exists(model_file):
                logger.warning(f"Model file {model_file} does not exist")
                return False
                
            logger.info(f"Loading models from {model_file}")
            models = joblib.load(model_file)
            
            # Load all components
            self.kmeans = models.get("kmeans")
            self.feature_scaler = models.get("feature_scaler")
            self.location_scaler = models.get("location_scaler")
            self.tfidf = models.get("tfidf")
            self.nn_model = models.get("nn_model")
            self.feature_matrix = models.get("feature_matrix")
            self.tfidf_matrix = models.get("tfidf_matrix")
            self.cluster_labels = models.get("cluster_labels")
            self.pois_df = models.get("pois_df")
            
            # Verify all components were loaded
            required_components = ["kmeans", "feature_scaler", "location_scaler", "tfidf", 
                                "nn_model", "feature_matrix", "tfidf_matrix", "cluster_labels", "pois_df"]
            for component in required_components:
                if models.get(component) is None:
                    logger.warning(f"Missing required component: {component}")
                    return False
                    
            # Log loaded components
            logger.info("Successfully loaded models from disk")
            logger.info(f"Loaded kmeans model: {type(self.kmeans)}")
            logger.info(f"Loaded feature_scaler: {type(self.feature_scaler)}")
            logger.info(f"Loaded location_scaler: {type(self.location_scaler)}")
            logger.info(f"Loaded tfidf: {type(self.tfidf)}")
            logger.info(f"Loaded nn_model: {type(self.nn_model)}")
            logger.info(f"Loaded feature_matrix shape: {self.feature_matrix.shape}")
            logger.info(f"Loaded tfidf_matrix shape: {self.tfidf_matrix.shape}")
            logger.info(f"Loaded cluster_labels shape: {self.cluster_labels.shape}")
            logger.info(f"Loaded pois_df shape: {self.pois_df.shape}")
            
            # Ensure feature_text column exists in pois_df
            if 'feature_text' not in self.pois_df.columns:
                self.pois_df['feature_text'] = self.pois_df.apply(lambda row: ' '.join(
                    str(row[col]) for col in ['tourism', 'name']
                    if pd.notna(row.get(col, None))
                ), axis=1)
            
            logger.info("About to return True")  # Debug print
            result = True  # Store in variable to help debugging
            logger.info(f"Result value: {result}")  # Debug print
            return result
            
        except Exception as e:
            logger.warning(f"Could not load models: {str(e)}")
            return False
            
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
                logger.error("POIs data not loaded")
                return pd.DataFrame()
                
            # Filter POIs by location
            location = location.lower()
            logger.info(f"Filtering POIs for location: {location}")
            
            # First try exact match
            location_mask = self.pois_df['city'].str.lower() == location
            pois = self.pois_df[location_mask].copy()
            
            if len(pois) == 0:
                logger.warning(f"No exact matches for location: {location}")
                # Try partial match
                location_mask = self.pois_df['city'].str.lower().str.contains(location, na=False)
                pois = self.pois_df[location_mask].copy()
            
            if len(pois) == 0:
                logger.warning(f"No POIs found for location: {location}")
                return pd.DataFrame()
            
            # Define attraction types
            attraction_types = {
                'museum', 'gallery', 'artwork', 'attraction', 'viewpoint', 
                'theme_park', 'zoo', 'aquarium', 'park', 'garden',
                'historic_site', 'monument', 'castle', 'ruins',
                'entertainment', 'theater', 'cinema', 'arts_centre'
            }
            
            # Filter out non-attraction POIs
            attraction_mask = pois['tourism'].str.lower().isin(attraction_types)
            pois = pois[attraction_mask].copy()
            
            # Log some stats about the POIs
            logger.info(f"Found {len(pois)} attraction POIs for {location}")
            logger.debug(f"POI columns: {pois.columns.tolist()}")
            if len(pois) > 0:
                sample_poi = pois.iloc[0]
                logger.debug(f"Sample POI name: {sample_poi.get('name')}")
                logger.debug(f"Sample POI tourism: {sample_poi.get('tourism')}")
                logger.debug(f"Sample POI coords: ({sample_poi.get('latitude')}, {sample_poi.get('longitude')})")
            
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
        """
        Get recommendations for a location.
        
        Args:
            location: Location to get recommendations for
            user_preferences: Dictionary of user preferences
            current_location: Optional tuple of (latitude, longitude)
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended POIs
        """
        try:
            logger.info(f"Getting recommendations for location: {location}")
            logger.debug(f"User preferences: {user_preferences}")
            
            # Get POIs for the location
            pois_df = self.get_pois_for_location(location)
            if pois_df.empty:
                logger.warning(f"No POIs found for location: {location}")
                return []
            
            # Apply mobility-based filtering if preferences provided
            if user_preferences.get('mobility'):
                logger.info("Applying mobility filter")
                filtered_df = self._apply_mobility_filter(user_preferences['mobility'], pois_df)
                logger.info(f"After mobility filter: {len(filtered_df)} POIs")
                if not filtered_df.empty:
                    pois_df = filtered_df
            
            # Get content-based recommendations
            logger.info("Getting content-based recommendations")
            recommendations = self._get_content_based_recommendations(
                pois_df=pois_df,
                user_preferences=user_preferences,
                current_location=current_location,
                n_recommendations=n_recommendations
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            logger.debug(f"Sample recommendation: {recommendations[0] if recommendations else 'No recommendations'}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            return []

    def _apply_mobility_filter(self, mobility_prefs: Dict, pois_df: pd.DataFrame) -> pd.DataFrame:
        """Apply mobility-based filtering to POIs."""
        try:
            if pois_df.empty:
                return pd.DataFrame()
            
            filtered_df = pois_df.copy()
            logger.info("Starting mobility filter with DataFrame shape: %s", filtered_df.shape)
            
            mode = mobility_prefs.get('mode', 'walking')
            max_distance = mobility_prefs.get('max_distance', 5.0)
            logger.info(f"Filtering for mode: {mode}, max_distance: {max_distance}")
            
            # Find relevant distance columns based on transport mode
            distance_cols = [col for col in filtered_df.columns 
                           if 'distance' in col.lower() and mode in col.lower()]
            
            if distance_cols:
                logger.info(f"Found distance columns: {distance_cols}")
                # Use the minimum distance across relevant transport modes
                filtered_df['min_distance'] = filtered_df[distance_cols].min(axis=1)
                filtered_df = filtered_df[
                    filtered_df['min_distance'] <= max_distance
                ]
                logger.info(f"After distance filtering, shape: {filtered_df.shape}")
            
            # Apply mode-specific filtering if relevant columns exist
            mode_cols = {
                'walking': ['walking'],
                'cycling': ['cycling', 'bicycle'],
                'public_transport': ['bus', 'tram', 'metro', 'train']
            }
            
            if mode in mode_cols:
                mode_terms = mode_cols[mode]
                logger.info(f"Looking for columns with terms: {mode_terms}")
                
                # Find columns related to the transport mode
                mode_cols = []
                for term in mode_terms:
                    mode_cols.extend([
                        col for col in filtered_df.columns
                        if term in col.lower()
                    ])
                
                if mode_cols:
                    logger.info(f"Found mode-specific columns: {mode_cols}")
                    # Keep POIs that have some activity for the chosen mode
                    mode_mask = filtered_df[mode_cols].notna().any(axis=1)
                    filtered_df = filtered_df[mode_mask]
                    logger.info(f"After mode filtering, shape: {filtered_df.shape}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error in mobility filter: {str(e)}", exc_info=True)
            return pois_df

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
        n_recommendations: int
    ) -> List[Dict]:
        """Get content-based recommendations using TF-IDF and location similarity."""
        try:
            logger.info("Starting content-based recommendations")
            logger.debug(f"Input POIs shape: {pois_df.shape}")
            
            # Get interests from preferences
            interests = user_preferences.get('interests', ['tourist_spot'])
            logger.info(f"User interests: {interests}")
            
            # Create query from interests
            query = ' '.join(interests)
            logger.info(f"Generated query: {query}")
            
            # Get content similarity scores
            content_scores = self._get_content_similarity(query, pois_df)
            logger.info(f"Generated content scores shape: {content_scores.shape if hasattr(content_scores, 'shape') else len(content_scores)}")
            
            # Get location similarity if current_location provided
            location_scores = None
            if current_location:
                logger.info("Calculating location similarity")
                location_scores = self._get_location_similarity(current_location, pois_df)
            
            # Combine scores
            final_scores = content_scores
            if location_scores is not None:
                logger.info("Combining content and location scores")
                final_scores = 0.7 * content_scores + 0.3 * location_scores
            
            # Define type weights for diversity
            type_weights = {
                'museum': 0.7,  # Reduce museum weight
                'gallery': 0.8,
                'artwork': 0.9,
                'attraction': 1.0,  # Keep general attractions at full weight
                'viewpoint': 1.0,
                'theme_park': 1.0,
                'zoo': 1.0,
                'aquarium': 1.0,
                'park': 0.9,
                'garden': 0.9,
                'historic_site': 0.8,
                'monument': 0.9,
                'castle': 1.0,
                'ruins': 0.9,
                'entertainment': 1.0,
                'theater': 0.9,
                'cinema': 0.8,
                'arts_centre': 0.8
            }
            
            # Create a list of (index, score, poi) tuples for sorting
            scored_pois = []
            for i in range(len(pois_df)):
                poi = pois_df.iloc[i]
                score = final_scores[i]
                
                # Apply type weight
                poi_type = poi.get('tourism', 'attraction').lower()
                weight = type_weights.get(poi_type, 1.0)
                score *= weight
                
                scored_pois.append((i, score, poi))
            
            # Sort by score
            scored_pois.sort(key=lambda x: x[1], reverse=True)
            
            # Track type counts for diversity
            type_counts = {}
            max_per_type = max(2, n_recommendations // 3)  # Allow at most 1/3 of recommendations to be the same type
            
            # Get recommendations, ensuring diversity
            recommendations = []
            for idx, score, poi in scored_pois:
                if len(recommendations) >= n_recommendations:
                    break
                    
                name = poi.get('name')
                if not name or pd.isna(name):
                    logger.warning(f"Skipping POI at index {idx} due to missing name")
                    continue
                
                # Get POI type and check if we have too many of this type
                tourism = poi.get('tourism', 'attraction').lower()
                if tourism not in type_counts:
                    type_counts[tourism] = 0
                if type_counts[tourism] >= max_per_type:
                    logger.info(f"Skipping {name} because we already have {max_per_type} of type {tourism}")
                    continue
                type_counts[tourism] += 1
                
                latitude = poi.get('latitude')
                longitude = poi.get('longitude')
                if pd.isna(latitude) or pd.isna(longitude):
                    logger.warning(f"POI {name} has invalid coordinates: ({latitude}, {longitude})")
                    latitude = longitude = None
                
                recommendation = {
                    'name': name,
                    'type': tourism,
                    'latitude': latitude,
                    'longitude': longitude,
                    'score': float(score),
                    'distance': self._calculate_distance(
                        current_location,
                        (latitude, longitude)
                    ) if current_location and latitude and longitude else None
                }
                logger.debug(f"Created recommendation: {recommendation}")
                recommendations.append(recommendation)
            
            logger.info(f"Returning {len(recommendations)} recommendations with type distribution: {type_counts}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {str(e)}", exc_info=True)
            return []
        
    def _get_location_similarity(
        self,
        current_location: Tuple[float, float],
        pois_df: pd.DataFrame
    ) -> np.ndarray:
        """Calculate location similarity scores."""
        try:
            # Calculate distances
            distances = np.array([
                self._calculate_distance(current_location, (lat, lon))
                for lat, lon in zip(pois_df['latitude'], pois_df['longitude'])
            ])
            
            # Transform distances to similarity scores (closer = higher score)
            max_distance = max(distances) if len(distances) > 0 else 1.0
            similarity_scores = 1 - (distances / max_distance)
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"Error calculating location similarity: {str(e)}")
            return np.zeros(len(pois_df))
            
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