"""
Enhanced chatbot handler with sentiment analysis and recommendation integration
"""
import logging
from typing import Dict, Optional, Tuple, List, Any
from nlp.sentiment_analyzer import TripAdvisorSentimentAnalyzer
from nlp.nlp_data_models import NLPResponse
from recommender.hybrid_recommender import HybridRecommender
from .travel_assistant import TravelAssistant
from .utils.travel_planner import TravelPlanner
import spacy
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotHandler:
    def __init__(self, init_recommender: bool = True):
        """
        Initialize chatbot with sentiment analysis and recommender
        
        Args:
            init_recommender: If True, initialize the recommender system immediately.
                            If False, it will be initialized on first use.
        """
        self.sentiment_analyzer = TripAdvisorSentimentAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")
        self.recommender = None
        self.travel_planner = None
        self.travel_assistant = None
        
        if init_recommender:
            success = self.initialize_recommender()
            if not success:
                logger.error("Failed to initialize recommender during __init__")
            
    def initialize_recommender(self, data_path: Optional[str] = None) -> bool:
        """
        Initialize the recommender system with optional custom data path
        
        Args:
            data_path: Optional path to the POIs data file. If None, uses default path.
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.recommender is not None:
            logger.info("Recommender system already initialized")
            return True

        try:
            # Create recommender instance
            self.recommender = HybridRecommender()
            
            # Try to load optimized model first
            model_cache = os.path.join(os.path.dirname(__file__), "..", "..", "model_cache")
            optimized_model = os.path.join(model_cache, "recommender_models_optimized.joblib")
            
            logger.info(f"Attempting to load optimized model from: {optimized_model}")
            if os.path.exists(optimized_model):
                success = self.recommender.load_models(optimized_model)
                if success:
                    logger.info("Successfully loaded optimized recommender model")
                    # Initialize travel planner with loaded recommender
                    self.travel_planner = TravelPlanner(recommender=self.recommender)
                    self.travel_assistant = TravelAssistant(self.travel_planner)
                    return True
                else:
                    logger.error("Failed to load optimized model despite file existing")
                    self.recommender = None
            else:
                logger.warning(f"Optimized model not found at: {optimized_model}")
                
            logger.warning("Could not load optimized model, falling back to training")
                
            # Only if loading fails, try training
            if data_path is None:
                data_path = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "Databases for training",
                    "gtfs_nl",
                    "pois_with_mobility.csv"
                )
            
            self.recommender = HybridRecommender()  # Create fresh instance for training
            self.recommender.fit(data_path)
            # Initialize travel planner after training
            self.travel_planner = TravelPlanner(recommender=self.recommender)
            self.travel_assistant = TravelAssistant(self.travel_planner)
            logger.info("Successfully initialized recommender system through training")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize recommender: {str(e)}")
            self.recommender = None
            return False

    def process_message(self, message: str, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process incoming message using sentiment analysis and travel assistance"""
        try:
            # Ensure recommender and travel components are initialized
            if self.recommender is None:
                success = self.initialize_recommender()
                if not success:
                    return {
                        'response': {'text': "I apologize, but I'm having trouble initializing my recommendation system. Please try again later."},
                        'success': False,
                        'error': "Failed to initialize recommender"
                    }
                
            if self.travel_planner is None:
                self.travel_planner = TravelPlanner(recommender=self.recommender)
                self.travel_assistant = TravelAssistant(self.travel_planner)
            
            # Try travel assistant first
            travel_response = self.travel_assistant.process_query(message)
            
            # If travel assistant handled it successfully
            if travel_response.text or travel_response.attractions or travel_response.route:
                response_data = {
                    'text': travel_response.text or "I found some information for you!",
                    'attractions': travel_response.attractions,
                    'route': travel_response.route,
                    'cost': travel_response.cost,
                    'reviews': travel_response.reviews
                }
                
                if travel_response.error:
                    response_data['error'] = travel_response.error
                
                return {
                    'response': response_data,
                    'success': True
                }
            
            # Fallback to existing chatbot logic
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(message)
            doc = self.nlp(message.lower())
            
            if any(word.text in ["recommend", "suggestion", "suggest", "find"] for word in doc):
                # Extract location and preferences
                location, preferences = self._extract_recommendation_params(doc)
                
                if location:
                    # Get recommendations
                    recommendations = self.recommender.get_recommendations(
                        user_preferences=preferences,
                        current_location=location
                    )
                    
                    # Format response
                    response = self._format_recommendations(recommendations)
                else:
                    response = "I'd be happy to recommend places! Could you tell me which city you're interested in?"
            else:
                # Generate regular response
                response = self._generate_response(message, sentiment_score)
            
            return {
                'response': {'text': response},
                'sentiment': sentiment_score,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                'response': {'text': "I apologize, but I'm having trouble processing your request. Please try again."},
                'success': False,
                'error': str(e)
            }

    def _extract_recommendation_params(self, doc) -> Tuple[Optional[Tuple[float, float]], Dict]:
        """Extract location and preferences from the message"""
        # Default preferences
        preferences = {
            'mobility': {
                'mode': 'walking',
                'max_distance': 2.0
            }
        }
        
        # Extract location (using predefined city coordinates for now)
        city_coords = {
            'amsterdam': (52.3676, 4.9041),
            'rotterdam': (51.9225, 4.4792),
            'utrecht': (52.0907, 5.1214),
            'delft': (52.0116, 4.3571)
        }
        
        location = None
        for city in city_coords:
            if city in doc.text.lower():
                location = city_coords[city]
                break
        
        # Extract activity types
        activities = {
            'museum': ['museum', 'art', 'culture', 'history'],
            'park': ['park', 'nature', 'outdoor', 'walking'],
            'attraction': ['attraction', 'sightseeing', 'tourist']
        }
        
        for activity_type, keywords in activities.items():
            if any(keyword in doc.text.lower() for keyword in keywords):
                preferences['type'] = activity_type
                preferences['keywords'] = ' '.join(keywords)
                break
        
        # Extract mobility preferences
        transport_modes = {
            'walking': ['walk', 'walking', 'stroll'],
            'cycling': ['bike', 'bicycle', 'cycling'],
            'public_transport': ['bus', 'train', 'tram', 'metro', 'public transport']
        }
        
        for mode, keywords in transport_modes.items():
            if any(keyword in doc.text.lower() for keyword in keywords):
                preferences['mobility']['mode'] = mode
                break
        
        return location, preferences
        
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """Format recommendations into a readable response"""
        if not recommendations:
            return "I couldn't find any places matching your preferences. Could you try different criteria?"
            
        response = "Here are some places you might enjoy:\n\n"
        for i, rec in enumerate(recommendations[:5], 1):
            response += f"{i}. {rec['name']} ({rec['type']})\n"
            
        response += "\nWould you like more details about any of these places?"
        return response
        
    def _generate_response(self, message: str, sentiment_score: float) -> str:
        """Generate appropriate response based on sentiment"""
        if "help" in message.lower():
            return ("I can help you discover interesting places! Just ask me for recommendations "
                   "and tell me what city you're interested in. You can also specify what kind of "
                   "places you like (museums, parks, attractions) and how you prefer to travel "
                   "(walking, cycling, public transport).")
        elif sentiment_score < 0.3:
            return ("I notice you seem unsatisfied. I'd be happy to help you find better places "
                   "that match your interests. What kind of places do you usually enjoy?")
        else:
            return "Hello! I'm your travel assistant. I can recommend interesting places based on your preferences. Where would you like to explore?"
