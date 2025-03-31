import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

# Use relative imports for local packages
from ..nlp.text_processor import TextProcessor
from ..nlp.intent_classifier import IntentClassifier
from ..nlp.entity_recognizer import EntityRecognizer
from ..nlp.sentiment_analyzer import TripAdvisorSentimentAnalyzer
from .travel_planner import TravelPlanner, TravelPreferences, RoutePreference
import spacy

logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading language model for the spaCy POS tagger")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

@dataclass
class TravelResponse:
    """Class to structure travel assistant responses."""
    text: str
    intent: Optional[str] = None
    entities: Dict = field(default_factory=dict)
    route: Optional[Dict] = None
    reviews: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    attractions: List[Dict] = field(default_factory=list)
    sentiment: Optional[str] = None  # Added sentiment field for review queries
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary."""
        return {
            'text': self.text,
            'intent': self.intent,
            'entities': self.entities,
            'route': self.route,
            'reviews': self.reviews,
            'error': self.error,
            'attractions': self.attractions,
            'sentiment': self.sentiment
        }

class TravelAssistant:
    """Main class for processing travel-related queries and generating responses."""
    
    def __init__(self):
        """Initialize the travel assistant components."""
        self.text_processor = TextProcessor()  # Removed nlp parameter since it's not needed
        self.intent_classifier = IntentClassifier()
        self.entity_recognizer = EntityRecognizer(nlp)
        self.sentiment_analyzer = TripAdvisorSentimentAnalyzer()
        self.travel_planner = TravelPlanner()
        
    def process_message(self, message: str, preferences: Dict = None, user_id: str = None) -> TravelResponse:
        """
        Process an incoming message and generate a response.
        
        Args:
            message: The user's message
            preferences: Optional user preferences for travel planning
            user_id: Optional user ID for personalization
            
        Returns:
            TravelResponse object containing the response and related data
        """
        try:
            # Process text
            processed_text = self.text_processor.process(message)
            
            # Classify intent
            intent = self.intent_classifier.classify(processed_text)
            
            # Extract entities
            entities = self.entity_recognizer.extract_entities(processed_text)
            
            # Initialize response
            response = TravelResponse(
                text="I'm sorry, I couldn't understand your request.",
                intent=intent,
                entities=entities
            )
            
            # Handle different intents
            if intent == "greeting":
                response.text = "Hello! I'm your travel assistant. How can I help you plan your journey today?"
            elif intent == "route_query":
                route = self.travel_planner.plan_route(
                    entities.get("locations", []),
                    TravelPreferences(**preferences) if preferences else None
                )
                response.route = route
                response.text = self._format_route_response(route)
                
            elif intent == "review_query":
                reviews = self.travel_planner.get_reviews(entities.get("location"))
                sentiment = self.sentiment_analyzer.analyze_reviews(reviews)
                response.reviews = reviews
                response.sentiment = sentiment
                response.text = self._format_review_response(reviews, sentiment)
                
            elif intent == "attraction_query":
                attractions = self.travel_planner.get_attractions(
                    entities.get("location"),
                    preferences.get("interests") if preferences else None
                )
                response.attractions = attractions
                response.text = self._format_attraction_response(attractions)
            else:
                response.text = "I'm here to help with travel planning. You can ask me about routes, attractions, or reviews for different places!"
                
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return TravelResponse(
                text="I apologize, but I encountered an error while processing your request.",
                error=str(e)
            )
    
    def _format_route_response(self, route: Dict) -> str:
        """Format route data into a readable response."""
        if route.get("error"):
            return route["message"]
            
        if not route or route.get("status") != "success":
            return "I couldn't find a suitable route for your request."
            
        route_data = route["route"]
        return self._format_route_data(route_data)
    
    def _format_route_data(self, route_data: Dict) -> str:
        """Format the route data from the recommender system."""
        try:
            return route_data["description"]
        except (KeyError, TypeError):
            return "I found a route but couldn't format it properly. Please try again."
    
    def _format_review_response(self, reviews: List[Dict], sentiment: str) -> str:
        """Format review data into a readable response."""
        if not reviews:
            return "I couldn't find any reviews for that location."
        
        try:
            response = f"Here are some reviews (Overall sentiment: {sentiment}):\n\n"
            for review in reviews[:3]:  # Show top 3 reviews
                response += f"- {review['text']}\n"
                response += f"  Rating: {review['rating']} stars\n\n"
            return response.strip()
        except (KeyError, TypeError):
            return "I found some reviews but couldn't format them properly. Please try again."
    
    def _format_attraction_response(self, attractions: List[Dict]) -> str:
        """Format attraction data into a readable response."""
        if not attractions:
            return "I couldn't find any attractions for that location."
            
        try:
            response = "Here are some places you might enjoy:\n\n"
            for attraction in attractions[:5]:  # Show top 5 attractions
                response += f"{attraction['name']}\n"
                if 'description' in attraction:
                    response += f"- {attraction['description']}\n"
                if 'rating' in attraction:
                    response += f"- Rating: {attraction['rating']} stars\n"
                if 'distance' in attraction:
                    response += f"- Distance: {attraction['distance']:.1f} km\n"
                response += "\n"
            return response.strip()
        except (KeyError, TypeError):
            return "I found some attractions but couldn't format them properly. Please try again."

# Create a singleton instance
_travel_assistant = None

def get_travel_assistant() -> TravelAssistant:
    """Get or create the singleton TravelAssistant instance."""
    global _travel_assistant
    if _travel_assistant is None:
        _travel_assistant = TravelAssistant()
    return _travel_assistant

def generate_response(message: str, preferences: Dict = None, user_id: str = None) -> str:
    """
    Generate a response to a user message using the travel assistant.
    
    Args:
        message: The user's message
        preferences: Optional user preferences for travel planning
        user_id: Optional user ID for personalization
        
    Returns:
        A string containing the response message
    """
    assistant = get_travel_assistant()
    response = assistant.process_message(message, preferences, user_id)
    return response.text
