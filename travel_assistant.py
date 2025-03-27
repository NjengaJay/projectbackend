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
        
        # Initialize attractions data
        self.attractions = {
            "general": {
                "Museums & Art": [
                    {"name": "🏛️ Rijksmuseum - Amsterdam", 
                     "accessible": True, 
                     "features": ["Wheelchair ramps", "Elevators", "Accessible restrooms", "Audio guides"]},
                    {"name": "🎨 Van Gogh Museum - Amsterdam", 
                     "accessible": True, 
                     "features": ["Step-free access", "Elevators", "Accessible restrooms", "Wheelchair rental"]},
                    {"name": "🏺 NEMO Science Museum - Amsterdam", 
                     "accessible": True, 
                     "features": ["Ramps", "Elevators", "Accessible restrooms", "Touch exhibits"]},
                    {"name": "🎨 Mauritshuis - The Hague", 
                     "accessible": True, 
                     "features": ["Elevator access", "Wheelchair rental", "Accessible entrance"]},
                    {"name": "💡 Philips Museum - Eindhoven", 
                     "accessible": True, 
                     "features": ["Step-free access", "Elevators", "Accessible toilets"]},
                    {"name": "🎵 Museum Speelklok - Utrecht", 
                     "accessible": True, 
                     "features": ["Ground floor accessible", "Ramps", "Accessible toilets"]}
                ],
                "Modern Attractions": [
                    {"name": "🎡 Efteling Theme Park - Kaatsheuvel", 
                     "accessible": True, 
                     "features": ["Wheelchair rental", "Accessible rides", "Special assistance", "Priority access"]},
                    {"name": "🌺 Madurodam Miniature Park - The Hague", 
                     "accessible": True, 
                     "features": ["Flat terrain", "Wide paths", "Accessible facilities"]},
                    {"name": "📱 A'DAM Lookout - Amsterdam", 
                     "accessible": True, 
                     "features": ["Elevator access", "Accessible viewing areas", "Assistance available"]},
                    {"name": "🏗️ Market Hall - Rotterdam", 
                     "accessible": True, 
                     "features": ["Level access", "Wide aisles", "Elevators", "Accessible restrooms"]}
                ],
                "Nature & Gardens": [
                    {"name": "🌷 Keukenhof Gardens - Lisse", 
                     "accessible": True, 
                     "features": ["Paved paths", "Wheelchair rental", "Accessible facilities", "Rest areas"]},
                    {"name": "🌲 Vondelpark - Amsterdam", 
                     "accessible": True, 
                     "features": ["Paved paths", "Flat terrain", "Accessible facilities"]},
                    {"name": "🌳 Genneper Parks - Eindhoven", 
                     "accessible": True, 
                     "features": ["Accessible paths", "Rest areas", "Visitor center access"]}
                ],
                "Historical Sites": [
                    {"name": "🏠 Anne Frank House - Amsterdam", 
                     "accessible": "Partially", 
                     "features": ["Ground floor accessible", "Virtual tour available", "Museum cafe accessible"]},
                    {"name": "⛪ Royal Palace Amsterdam", 
                     "accessible": True, 
                     "features": ["Elevator access", "Wheelchair rental", "Accessible route"]},
                    {"name": "🏰 Peace Palace - The Hague", 
                     "accessible": True, 
                     "features": ["Ramps", "Elevator", "Guided tours available"]}
                ]
            }
        }
        
    def is_greeting(self, query: str) -> bool:
        """Check if the query is a greeting."""
        query = query.lower()
        return any(re.search(pattern, query) for pattern in self.greeting_patterns)
        
    def get_greeting_response(self) -> Dict:
        """Generate a greeting response."""
        return {
            "status": "success",
            "message": "Hello! I'm your travel assistant. I can help you find routes between locations, suggest travel options, and provide travel information. How can I help you today?",
            "query": None,
            "entities": {},
            "is_greeting": True
        }

    def _extract_cities(self, text: str) -> List[str]:
        """Extract city names from text."""
        cities = []
        text_lower = text.lower()
        
        # First, try to extract cities using "from" and "to" keywords
        from_index = text_lower.find(" from ")
        to_index = text_lower.find(" to ")
        
        if from_index != -1 and to_index != -1:
            # Extract text after "from" until the next preposition or end of string
            from_text = text_lower[from_index + 6:].split()[0]
            # Extract text after "to" until the next preposition or end of string
            to_text = text_lower[to_index + 4:].split()[0].rstrip('?.,!')
            
            # Check if extracted cities are valid
            if from_text in self.travel_planner.dutch_cities:
                cities.append(from_text)
            if to_text in self.travel_planner.dutch_cities:
                cities.append(to_text)
        
        # If we don't find cities using keywords, try direct matching
        if len(cities) < 2:
            cities = []
            for city in self.travel_planner.dutch_cities.keys():
                if city in text_lower:
                    cities.append(city)
        
        # If we still don't have enough cities, try NLP
        if len(cities) < 2:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:
                    city = ent.text.lower()
                    if city in self.travel_planner.dutch_cities and city not in cities:
                        cities.append(city)
        
        return cities[:2]  # Return at most 2 cities

    def process_query(self, query: str) -> TravelResponse:
        """Process a user query and return a response."""
        # Create response object
        response = TravelResponse()
        
        try:
            # Classify intent
            intent, confidence = self.intent_classifier.classify(query)
            
            # Map intent to standardized names
            intent_mapping = {
                'attraction_query': 'find_attraction',
                'cost_query': 'get_cost',
                'route_query': 'find_route',
                'review_query': 'get_review',
                'unknown': 'unknown'
            }
            response.intent = intent_mapping.get(intent, 'unknown')
            
            # Extract entities based on intent
            entities = self.intent_classifier.extract_entities(query)
            response.entities = entities
            
            # Validate Dutch locations
            dutch_cities = ['amsterdam', 'rotterdam', 'utrecht', 'den haag', 'eindhoven', 'groningen', 'tilburg', 'almere', 'breda', 'nijmegen']
            
            def validate_location(location: str) -> bool:
                """Check if location is a Dutch city."""
                if not location:
                    return False
                return location.lower() in dutch_cities
            
            # Process based on intent
            if intent == "route_query":
                origin = entities.get("origin")
                destination = entities.get("destination")
                
                if not origin or not destination:
                    response.error = "Missing origin or destination"
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
                
            elif intent == "cost_query":
                origin = entities.get("origin")
                destination = entities.get("destination")
                
                if not origin or not destination:
                    response.error = "Missing origin or destination"
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
                response.cost = {
                    "amount": 25,
                    "currency": "EUR",
                    "details": "Standard fare"
                }
                response.text = f"The cost from {origin.title()} to {destination.title()} is €25"
                
            elif intent == "review_query":
                # Extract subject (place/attraction) from query
                subject = entities.get("subject")
                if not subject:
                    response.error = "Missing place or attraction to review"
                    return response
                
                # Mock Dutch attractions for reviews
                dutch_attractions = {
                    'rijksmuseum': 'Amsterdam',
                    'van gogh museum': 'Amsterdam',
                    'anne frank house': 'Amsterdam',
                    'efteling': 'Kaatsheuvel',
                    'keukenhof': 'Lisse',
                    'mauritshuis': 'Den Haag',
                    'kinderdijk': 'Rotterdam'
                }
                
                if subject.lower() not in dutch_attractions:
                    response.error = f"'{subject}' is not a recognized Dutch attraction"
                    return response
                
                # For now return mock reviews with sentiment
                mock_reviews = [
                    {
                        "rating": 4.5,
                        "text": "Great Dutch cultural experience!",
                        "date": "2025-03-15",
                        "sentiment": "positive"
                    },
                    {
                        "rating": 4.0,
                        "text": "Beautiful place but can get crowded",
                        "date": "2025-03-10",
                        "sentiment": "mixed"
                    }
                ]
                response.reviews = mock_reviews
                
                # Set overall sentiment based on average review sentiment
                sentiments = [review.get("sentiment") for review in mock_reviews]
                if all(s == "positive" for s in sentiments):
                    response.sentiment = "positive"
                elif all(s == "negative" for s in sentiments):
                    response.sentiment = "negative"
                else:
                    response.sentiment = "mixed"
                
                # Add text response
                response.text = f"Here's what people think about {subject.title()}"
                
            elif intent == "attraction_query":
                # Extract location and attraction type from query
                location = entities.get("location")
                attraction_type = entities.get("attraction_type")
                
                # Check for required entities
                if not location:
                    response.error = "Missing location"
                    return response
                if not attraction_type:
                    response.error = "Missing attraction type"
                    return response
                    
                if not validate_location(location):
                    response.error = f"'{location}' is not a valid Dutch city"
                    return response
                
                # For now return mock Dutch attractions
                mock_attractions = {
                    'amsterdam': [
                        {
                            "name": "Rijksmuseum",
                            "type": "museum",
                            "rating": 4.8,
                            "location": "amsterdam",
                            "description": "Dutch national museum"
                        },
                        {
                            "name": "Van Gogh Museum",
                            "type": "museum",
                            "rating": 4.7,
                            "location": "amsterdam",
                            "description": "World's largest Van Gogh collection"
                        }
                    ],
                    'den haag': [
                        {
                            "name": "Mauritshuis",
                            "type": "museum",
                            "rating": 4.6,
                            "location": "den haag",
                            "description": "Home to Dutch Golden Age paintings"
                        }
                    ]
                }.get(location.lower(), [])
                
                # Filter by attraction type
                attractions = [a for a in mock_attractions if a["type"] == attraction_type]
                
                if not attractions:
                    response.error = f"No {attraction_type}s found in {location.title()}"
                    return response
                
                # Add mock reviews
                mock_reviews = [
                    {
                        "rating": 4.5,
                        "text": "Amazing Dutch art collection!",
                        "date": "2025-03-15",
                        "sentiment": "positive"
                    },
                    {
                        "rating": 4.0,
                        "text": "Great museum with Dutch masters",
                        "date": "2025-03-10",
                        "sentiment": "mixed"
                    }
                ]
                
                response.attractions = attractions
                response.reviews = mock_reviews
                response.text = f"Here are some {attraction_type}s in {location.title()}"
                
            elif intent == "unknown":
                response.error = "I'm not sure what you're asking. Try asking about routes, attractions, or costs in the Netherlands."
                
        except Exception as e:
            response.error = str(e)
            
        return response

    def _handle_route_query(self, entities: Dict) -> TravelResponse:
        """Handle queries about routes and directions."""
        try:
            # Extract start and end locations
            start_location = entities.get("start_location")
            end_location = entities.get("end_location")
            
            if not start_location or not end_location:
                return TravelResponse(
                    intent="route",
                    entities=entities,
                    route=[],
                    error="Missing start or end location"
                )
            
            # Get preferences
            preferences = entities.get("preferences", {})
            
            # Get route from planner
            route_info = self.travel_planner.get_route(
                start=start_location,
                end=end_location,
                preferences=preferences
            )
            
            if "error" in route_info:
                return TravelResponse(
                    intent="route",
                    entities=entities,
                    route=[],
                    error=route_info["error"]
                )
            
            return TravelResponse(
                intent="route",
                entities=entities,
                route=route_info["route"],
                text=route_info["description"],
                cost=route_info["total_cost"]
            )
            
        except Exception as e:
            logger.error(f"Error handling route query: {str(e)}", exc_info=True)
            return TravelResponse(
                intent="route",
                entities=entities,
                route=[],
                error=f"Error finding route: {str(e)}"
            )

    def _get_reviews_for_route(self, start: str, end: str, mode: str) -> List[Dict]:
        """Get relevant reviews for a route segment."""
        try:
            # Mock reviews for now - in production this would query a review database
            reviews_db = {
                'amsterdam-rotterdam': [
                    {
                        'rating': 4.5,
                        'text': "Great accessible route! The intercity train has excellent wheelchair facilities.",
                        'date': '2025-03-20',
                        'mode': 'intercity_train'
                    },
                    {
                        'rating': 4.0,
                        'text': "Staff was very helpful with boarding assistance.",
                        'date': '2025-03-19',
                        'mode': 'intercity_train'
                    }
                ]
            }
            
            # Create route key
            route_key = f"{start.lower()}-{end.lower()}"
            
            # Get reviews for route
            route_reviews = reviews_db.get(route_key, [])
            
            # Filter by transport mode if specified
            if mode:
                route_reviews = [r for r in route_reviews if r['mode'] == mode]
                
            return route_reviews
            
        except Exception as e:
            logger.error(f"Error getting reviews: {str(e)}", exc_info=True)
            return []
    
    def _handle_info_query(self, entities: Dict[str, Any]) -> TravelResponse:
        """Handle queries about attractions and places to visit."""
        try:
            # Extract location and attraction type
            location = None
            attraction_type = None
            
            for ent in entities:
                if "location" in ent:
                    location = ent
                elif "attraction_type" in ent:
                    attraction_type = ent
            
            if not location:
                return TravelResponse(
                    intent="info",
                    entities=entities,
                    route=[],
                    error="Missing location for attraction search"
                )
            
            # Get recommendations from planner
            reviews = self.travel_planner.get_recommendations(
                location=location,
                attraction_type=attraction_type
            )
            
            return TravelResponse(
                intent="info",
                entities=entities,
                reviews=reviews
            )
            
        except Exception as e:
            logger.error(f"Error handling info query: {str(e)}", exc_info=True)
            return TravelResponse(
                intent="info",
                entities=entities,
                route=[],
                error=f"Error finding attractions: {str(e)}"
            )

    def _get_mock_reviews(self, attractions):
        reviews = []
        for attraction in attractions:
            reviews.append({
                'rating': 4.8,
                'text': f"Amazing {attraction['name']}!",
                'date': '2025-03-20',
            })
            reviews.append({
                'rating': 4.5,
                'text': f"Great accessibility features in {attraction['name']}.",
                'date': '2025-03-19',
            })
        return reviews
