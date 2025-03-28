"""
Travel Assistant module that integrates NLP and travel planning capabilities.
"""

import logging
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from .nlp.text_processor import TextProcessor
from .nlp.intent_classifier import IntentClassifier
from .nlp.entity_recognizer import EntityRecognizer
from .nlp.sentiment_analyzer import TripAdvisorSentimentAnalyzer
from .travel_planner import TravelPlanner, TravelPreferences, RoutePreference
import spacy

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

@dataclass
class TravelQuery:
    """A processed travel query with intent and entities."""
    text: str
    intent: str
    entities: Dict[str, Any]
    confidence: float = 0.0

@dataclass
class TravelResponse:
    """Response object for travel-related queries."""
    intent: Optional[str] = None
    entities: Dict = field(default_factory=dict)
    text: Optional[str] = None
    route: Optional[Dict] = None
    cost: Optional[float] = None
    reviews: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    attractions: List[Dict] = field(default_factory=list)
    sentiment: Optional[str] = None  # Added sentiment field for review queries
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary."""
        return {
            'intent': self.intent,
            'entities': self.entities,
            'text': self.text,
            'route': self.route,
            'cost': self.cost,
            'reviews': self.reviews,
            'error': self.error,
            'attractions': self.attractions,
            'sentiment': self.sentiment
        }

class TravelAssistant:
    """Assistant for handling travel-related queries."""
    
    def __init__(self, model_path=None):
        """Initialize the travel assistant."""
        # Initialize components
        self.travel_planner = TravelPlanner()
        self.intent_classifier = IntentClassifier()
        self.entity_recognizer = EntityRecognizer()
        self.sentiment_analyzer = TripAdvisorSentimentAnalyzer()
        
        # Set thresholds
        self.intent_threshold = 0.5
        self.sentiment_threshold = 0.5
        
        # Load models if path provided
        if model_path:
            model_path = Path(model_path)
            if (model_path / "entity_model").exists():
                self.entity_recognizer.load_model(str(model_path / "entity_model"))
            if (model_path / "sentiment_model").exists():
                self.sentiment_analyzer.load_model(str(model_path / "sentiment_model"))
        
        self.conversation_state = {}
        
        # Define greeting patterns
        self.greeting_patterns = [
            r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
            r'\bhow are you\b',
            r'\bnice to meet you\b'
        ]
        
    def is_greeting(self, query: str) -> bool:
        """Check if the query is a greeting."""
        query = query.lower()
        return any(re.search(pattern, query) for pattern in self.greeting_patterns)
        
    def get_greeting_response(self) -> TravelResponse:
        """Generate a greeting response."""
        response = TravelResponse()
        response.text = "Hello! I'm your Dutch travel assistant. I can help you find routes between cities, suggest attractions, and provide travel information in the Netherlands. How can I help you today?"
        return response

    def process_query(self, query: str) -> TravelResponse:
        """Process a user query and return a response."""
        # Create response object
        response = TravelResponse()
        
        try:
            # Check if it's a greeting
            if self.is_greeting(query):
                return self.get_greeting_response()
            
            # Classify intent
            intent, confidence = self.intent_classifier.classify(query)
            
            # Set the intent
            response.intent = intent if confidence > self.intent_threshold else "unknown"
            
            # Handle unknown intent
            if response.intent == "unknown":
                response.error = "I'm sorry, I don't understand your query. Could you please rephrase it?"
                return response
            
            # Extract entities based on intent
            entities = self.entity_recognizer.extract_entities(query)
            response.entities = entities
            
            # Validate Dutch locations
            dutch_cities = ['amsterdam', 'rotterdam', 'utrecht', 'den haag', 'eindhoven', 'groningen', 'tilburg', 'almere', 'breda', 'nijmegen']
            
            def validate_location(location: str) -> bool:
                """Check if location is a Dutch city."""
                if not location:
                    return False
                return location.lower() in dutch_cities
            
            # Process based on intent
            if response.intent == "find_route":
                origin = entities.get("origin")
                destination = entities.get("destination")
                
                if not origin or not destination:
                    response.error = "Missing required information: Please specify both origin and destination cities"
                    return response
                
                if not validate_location(origin):
                    response.error = f"Origin '{origin}' is not a valid Dutch city"
                    return response
                    
                if not validate_location(destination):
                    response.error = f"Destination '{destination}' is not a valid Dutch city"
                    return response
                
                # For now, return mock route data
                mock_route = {
                    "origin": origin,
                    "destination": destination,
                    "duration": "2 hours",
                    "distance": "150 km",
                    "steps": [
                        f"Take train from {origin}",
                        "Change at Utrecht Centraal",
                        f"Arrive at {destination}"
                    ]
                }
                response.route = mock_route
                response.text = f"Here's the route from {origin.title()} to {destination.title()}"
                
            elif response.intent == "get_cost":
                origin = entities.get("origin")
                destination = entities.get("destination")
                
                if not origin or not destination:
                    response.error = "Missing required information: Please specify both origin and destination cities"
                    return response
                
                if not validate_location(origin):
                    response.error = f"Origin '{origin}' is not a valid Dutch city"
                    return response
                    
                if not validate_location(destination):
                    response.error = f"Destination '{destination}' is not a valid Dutch city"
                    return response
                
                # For now, return mock route and cost data
                mock_route = {
                    "origin": origin,
                    "destination": destination,
                    "duration": "2 hours",
                    "distance": "150 km",
                    "steps": [
                        f"Take train from {origin}",
                        "Change at Utrecht Centraal",
                        f"Arrive at {destination}"
                    ]
                }
                response.route = mock_route
                response.cost = 25.0
                response.text = f"The cost from {origin.title()} to {destination.title()} is €25.00"
                
            elif response.intent == "find_attraction":
                location = entities.get("location")
                attraction_type = entities.get("attraction_type", "any")
                
                if not location:
                    response.error = "Missing required information: Please specify a city to find attractions in"
                    return response
                
                if not validate_location(location):
                    response.error = f"Location '{location}' is not a valid Dutch city"
                    return response
                
                # Mock attractions data for Dutch cities
                mock_attractions = [
                    {
                        "name": "Rijksmuseum",
                        "type": "museum",
                        "city": "Amsterdam",
                        "rating": 4.8,
                        "reviews": [
                            {"text": "World-class art collection!", "rating": 5},
                            {"text": "Beautiful building and exhibits", "rating": 5},
                            {"text": "Must-visit in Amsterdam", "rating": 4}
                        ]
                    },
                    {
                        "name": "Van Gogh Museum",
                        "type": "museum",
                        "city": "Amsterdam",
                        "rating": 4.7,
                        "reviews": [
                            {"text": "Amazing collection of Van Gogh", "rating": 5},
                            {"text": "Well organized museum", "rating": 4},
                            {"text": "Fascinating artwork", "rating": 5}
                        ]
                    }
                ]
                
                response.attractions = mock_attractions
                response.reviews = mock_attractions[0]["reviews"]
                response.text = f"Here are some {attraction_type} to visit in {location.title()}"
                
            elif response.intent == "get_review":
                attraction = entities.get("attraction")
                
                if not attraction:
                    response.error = "Missing required information: Please specify an attraction to get reviews for"
                    return response
                
                # Mock review data for Dutch attractions
                mock_reviews = [
                    {"text": "World-class art collection!", "rating": 5},
                    {"text": "Beautiful building and exhibits", "rating": 5},
                    {"text": "Must-visit in Amsterdam", "rating": 4}
                ]
                
                response.reviews = mock_reviews
                response.sentiment = "positive"  # Add sentiment
                response.text = f"Here are some reviews for {attraction}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            response.error = str(e)
            return response
