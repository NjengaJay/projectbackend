import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

# Use relative imports for local packages
from .nlp.text_processor import TextProcessor
from .nlp.intent_classifier import IntentClassifier
from .nlp.entity_recognizer import EntityRecognizer
from .nlp.sentiment_analyzer import TripAdvisorSentimentAnalyzer
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
            if intent == "route_query":
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
                
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return TravelResponse(
                text="I apologize, but I encountered an error while processing your request.",
                error=str(e)
            )
    
    def _format_route_response(self, route: Dict) -> str:
        """Format route data into a readable response."""
        if not route:
            return "I couldn't find a suitable route for your request."
        # Add your route formatting logic here
        return "Here's your route: [Route formatting to be implemented]"
    
    def _format_review_response(self, reviews: List[Dict], sentiment: str) -> str:
        """Format review data into a readable response."""
        if not reviews:
            return "I couldn't find any reviews for that location."
        # Add your review formatting logic here
        return f"Here are some reviews (Overall sentiment: {sentiment}): [Review formatting to be implemented]"
    
    def _format_attraction_response(self, attractions: List[Dict]) -> str:
        """Format attraction data into a readable response."""
        if not attractions:
            return "I couldn't find any attractions for that location."
        # Add your attraction formatting logic here
        return "Here are some attractions you might enjoy: [Attraction formatting to be implemented]"

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