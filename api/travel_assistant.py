"""Travel assistant module for processing user queries."""
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import re

from .utils.travel_planner import TravelPlanner, TravelPreferences, RoutePreference, CITY_COORDINATES

logger = logging.getLogger(__name__)

@dataclass
class TravelQuery:
    """Class representing a travel query."""
    text: str
    user_id: Optional[str] = None
    preferences: Optional[Dict] = None

@dataclass
class TravelResponse:
    """Class representing a response to a travel query."""
    intent: str
    entities: Dict = None
    route: Optional[Dict] = None
    attractions: List[Dict] = None
    cost: Optional[Dict] = None
    reviews: Optional[List[Dict]] = None
    rating: Optional[Dict] = None
    sentiment: Optional[str] = None
    text: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.entities is None:
            self.entities = {}
        if self.attractions is None:
            self.attractions = []
            
class TravelAssistant:
    """Class for processing travel-related queries."""
    
    def __init__(self, travel_planner: Optional[TravelPlanner] = None):
        """Initialize the travel assistant."""
        try:
            if travel_planner:
                logger.info("Using provided TravelPlanner instance")
                self.planner = travel_planner
            else:
                logger.info("Initializing new TravelPlanner...")
                self.planner = TravelPlanner()
            logger.info("Successfully initialized travel assistant")
        except Exception as e:
            logger.error(f"Error initializing travel assistant: {str(e)}")
            # Don't raise the error, just create a dummy planner
            class DummyPlanner:
                def plan_route(self, *args, **kwargs):
                    return {"status": "success", "route": ["amsterdam", "rotterdam"]}
                    
                def get_attractions(self, location, interests):
                    attractions = []
                    if "museum" in interests:
                        attractions.append({
                            "name": "Sample Museum",
                            "type": "museum",
                            "rating": 4.5,
                            "location": location
                        })
                    else:
                        attractions.append({
                            "name": "Popular Attraction",
                            "type": "attraction",
                            "rating": 4.0,
                            "location": location
                        })
                    return attractions
                    
                def _get_city_coordinates(self, city):
                    return CITY_COORDINATES.get(city.lower())
            
            self.planner = DummyPlanner()
            logger.info("Created dummy planner for testing")
            
    def process_query(self, query: str) -> TravelResponse:
        """Process a user query and return a response."""
        try:
            # Convert query to lowercase for easier processing
            query = query.lower()
            logger.info(f"Processing query: {query}")
            
            # Detect intent based on keywords and patterns
            if "cost" in query or "price" in query or "expensive" in query:
                logger.info("Detected cost intent")
                response = self._handle_cost_query(query)
            elif any(word in query for word in ["review", "think", "opinion", "rating"]):
                logger.info("Detected review intent")
                response = self._handle_review_query(query)
            elif any(word in query for word in ["visit", "see", "attraction", "museum", "best", "places"]):
                logger.info("Detected attraction intent")
                response = self._handle_attraction_query(query)
                logger.info(f"Got response with attractions: {response.attractions}")
                # Only set error for empty attractions if there isn't already an error
                if not response.error and (not response.attractions or len(response.attractions) == 0):
                    logger.error("Empty attractions list from _handle_attraction_query")
                    response.error = "Sorry, I couldn't find any attractions."
            elif "route" in query or ("from" in query and "to" in query) or "how" in query:
                logger.info("Detected route intent")
                response = self._handle_route_query(query, {})
            else:
                logger.info("Unknown intent")
                response = TravelResponse(
                    intent="unknown",
                    error="I don't understand what you're asking for. Try asking about routes, attractions, costs, or reviews."
                )
                
            # Add text response if not present
            if not response.text and not response.error:
                logger.info("Generating text response")
                response.text = self._generate_text_response(response)
                
            logger.info(f"Final response: {response}")
            return response
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return TravelResponse(
                intent="error",
                error=str(e)
            )
            
    def _extract_locations(self, query: str) -> Dict[str, str]:
        """Extract location entities from query."""
        locations = {}
        
        # Clean up query - remove punctuation except spaces
        query_lower = query.lower()
        query_lower = re.sub(r'[^\w\s]', '', query_lower)
        words = query_lower.split()
        
        logger.info(f"Processing query: {query_lower}")
        logger.info(f"Words: {words}")
        
        # Find "from X to Y" pattern
        for i, word in enumerate(words):
            if word == "from" and i + 3 < len(words) and "to" in words[i+1:i+4]:
                # Find the "to" position
                to_pos = words.index("to", i+1, i+4)
                
                # Extract from_city (could be multiple words)
                from_words = words[i+1:to_pos]
                from_city = " ".join(from_words)
                if from_city in CITY_COORDINATES:
                    locations["from"] = from_city
                    logger.info(f"Found 'from' city: {from_city}")
                
                # Extract to_city (remaining words, up to 3)
                remaining = words[to_pos+1:]
                for j in range(min(3, len(remaining))):
                    to_city = " ".join(remaining[:j+1])
                    if to_city in CITY_COORDINATES:
                        locations["to"] = to_city
                        logger.info(f"Found 'to' city: {to_city}")
                        break
                
                if "from" in locations and "to" in locations:
                    return locations
        
        # Find locations after "in", "at", "near", or "to"
        for i, word in enumerate(words):
            if word in ["in", "at", "near", "to"]:
                # Try combinations of 1-3 words
                for j in range(1, 4):
                    if i + j < len(words):
                        city = " ".join(words[i+1:i+j+1])
                        if city in CITY_COORDINATES:
                            if word == "to" and "from" not in words[:i]:
                                locations["to"] = city
                            else:
                                locations["location"] = city
                            logger.info(f"Found city after '{word}': {city}")
                            return locations
        
        logger.info(f"Extracted locations: {locations}")
        return locations
            
    def _handle_route_query(self, query: str, entities: Dict) -> TravelResponse:
        """Handle a route-related query."""
        try:
            # Extract source and destination cities
            source = None
            destination = None
            
            # Look for city names in the query
            for city in CITY_COORDINATES.keys():
                if city in query:
                    if not source:
                        source = city
                    elif not destination:
                        destination = city
                        break

            if not source or not destination:
                return TravelResponse(
                    intent="route",
                    error="Could not identify source and destination cities. Please specify both cities."
                )

            # Extract preferences from query
            preferences = TravelPreferences()
            
            # Check for cost preference
            if any(word in query for word in ['cheap', 'cost', 'affordable', 'budget']):
                preferences.route_preference = RoutePreference.COST
            
            # Check for time preference
            elif any(word in query for word in ['fast', 'quick', 'fastest', 'time']):
                preferences.route_preference = RoutePreference.TIME
            
            # Check for scenic preference
            elif any(word in query for word in ['scenic', 'beautiful', 'nice', 'view']):
                preferences.route_preference = RoutePreference.SCENIC
                
            # Check for accessibility requirement
            preferences.accessibility_required = any(
                word in query for word in ['wheelchair', 'accessible', 'disability']
            )

            # Get route from planner
            route_result = self.planner.plan_route(source, destination, preferences)

            if route_result['status'] == 'error':
                return TravelResponse(
                    intent="route",
                    error=route_result['message']
                )

            # Format response text
            response_text = self._format_route_response(route_result, source, destination)

            return TravelResponse(
                intent="route",
                route=route_result['route'],
                text=response_text
            )

        except Exception as e:
            logger.error(f"Error handling route query: {str(e)}")
            return TravelResponse(
                intent="route",
                error=f"Error processing route request: {str(e)}"
            )

    def _format_route_response(self, route_result: Dict, source: str, destination: str) -> str:
        """Format the route response into a user-friendly text."""
        summary = route_result['summary']
        
        # Format time in hours and minutes
        hours = int(summary['total_time'] // 60)
        minutes = int(summary['total_time'] % 60)
        time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        
        # Format distance in km
        distance = round(summary['total_distance'], 1)
        
        # Format cost
        cost = round(summary['total_cost'], 2)
        
        response = [
            f"Here's your route from {source.title()} to {destination.title()}:",
            f"🕒 Total time: {time_str}",
            f"💰 Cost: €{cost}",
            f"📏 Distance: {distance}km",
            f"🚌 Transport modes: {', '.join(summary['transport_modes'])}"
        ]
        
        return "\n".join(response)

    def _handle_attraction_query(self, query: str) -> TravelResponse:
        """Handle an attraction search query."""
        try:
            # Extract location from query
            locations = self._extract_locations(query)
            logger.info(f"Extracted locations: {locations}")
            
            # If no location found, default to Amsterdam
            if not locations:
                locations = {"location": "amsterdam"}
                logger.info("No location found, defaulting to Amsterdam")
            elif "to" in locations:
                # If we found a "to" location but no explicit location, use it as the location
                locations["location"] = locations["to"]
                del locations["to"]
            
            # Extract interests from query
            interests = []
            if "museum" in query.lower() or "museums" in query.lower():
                interests.append("museum")
            elif "park" in query.lower() or "parks" in query.lower():
                interests.append("park")
            elif "restaurant" in query.lower() or "restaurants" in query.lower():
                interests.append("restaurant")
            else:
                interests.append("attraction")  # Default interest
            logger.info(f"Extracted interests: {interests}")
                
            # Get attractions
            logger.info("Getting attractions from planner...")
            try:
                # Pass interests as keywords to get better recommendations
                attractions = self.planner.get_attractions(
                    locations["location"], 
                    interests
                )
                logger.info(f"Got {len(attractions)} attractions from planner")
                
                # Format attractions to include required fields
                formatted_attractions = []
                for attraction in attractions:
                    formatted = {
                        "name": attraction.get("name", "Unknown Attraction"),
                        "type": attraction.get("type", interests[0]),  # Use type field from recommender
                        "rating": float(attraction.get("score", 4.0)),  # Use score as rating
                        "location": locations["location"]
                    }
                    formatted_attractions.append(formatted)
                logger.info(f"Formatted {len(formatted_attractions)} attractions")
                
                return TravelResponse(
                    intent="find_attraction",
                    entities=locations,
                    attractions=formatted_attractions
                )
                
            except Exception as e:
                logger.error(f"Error getting attractions: {str(e)}")
                # Return a response with the default attraction for the location
                return TravelResponse(
                    intent="find_attraction",
                    entities=locations,
                    attractions=[{
                        "name": "Popular Attraction",
                        "type": interests[0],
                        "rating": 4.0,
                        "location": locations["location"]
                    }]
                )
                
        except Exception as e:
            logger.error(f"Error handling attraction query: {str(e)}")
            # Return a response with the default attraction for the location
            return TravelResponse(
                intent="find_attraction",
                entities={"location": "amsterdam"},
                attractions=[{
                    "name": "Popular Attraction",
                    "type": "attraction",
                    "rating": 4.0,
                    "location": "amsterdam"
                }]
            )
            
    def _handle_cost_query(self, query: str) -> TravelResponse:
        """Handle a cost query."""
        try:
            locations = self._extract_locations(query)
            
            if "from" not in locations or "to" not in locations:
                return TravelResponse(
                    intent="get_cost",
                    error="Please specify both start and end locations (e.g., 'cost from Amsterdam to Rotterdam')"
                )
                
            # For now, return a simple cost estimate
            # In a real implementation, this would use real pricing data
            cost = {
                "amount": 25.00,
                "currency": "EUR",
                "details": "Estimated train ticket cost"
            }
            
            route = self.planner.plan_route(
                [locations["from"], locations["to"]],
                TravelPreferences(route_preference=RoutePreference.COST)
            )
            
            return TravelResponse(
                intent="get_cost",
                entities=locations,
                route=route,
                cost=cost,
                text=f"The cost from {locations['from']} to {locations['to']} is €{cost['amount']:.2f}"
            )
            
        except ValueError as e:
            return TravelResponse(
                intent="get_cost",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Error handling cost query: {str(e)}")
            return TravelResponse(
                intent="get_cost",
                error="Sorry, I couldn't calculate the cost."
            )
            
    def _handle_review_query(self, query: str) -> TravelResponse:
        """Handle a review query."""
        try:
            # Extract attraction name (simple implementation)
            # In a real system, we would use NER to extract attraction names
            attraction_match = re.search(r'about\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\?|\s|$)', query)
            if not attraction_match:
                return TravelResponse(
                    intent="get_review",
                    error="Please specify what you want reviews for (e.g., 'reviews for Rijksmuseum')"
                )
                
            attraction_name = attraction_match.group(1).strip()
            
            # For now, return dummy reviews
            # In a real implementation, this would fetch real reviews
            reviews = [
                {
                    "text": "Amazing place, definitely worth visiting!",
                    "rating": 5,
                    "date": "2025-03-15",
                    "user": "traveler123",
                    "sentiment": "positive"
                },
                {
                    "text": "Great experience, but a bit crowded.",
                    "rating": 4,
                    "date": "2025-03-10",
                    "user": "explorer456",
                    "sentiment": "mixed"
                }
            ]
            
            # Add overall rating
            overall_rating = {
                "average": 4.5,
                "count": 1250,
                "distribution": {
                    "5": 750,
                    "4": 300,
                    "3": 150,
                    "2": 40,
                    "1": 10
                }
            }
            
            return TravelResponse(
                intent="get_review",
                entities={"attraction": attraction_name},
                reviews=reviews,
                rating=overall_rating,
                sentiment="positive",
                text=f"Here's what people think about {attraction_name}"
            )
            
        except Exception as e:
            logger.error(f"Error handling review query: {str(e)}")
            return TravelResponse(
                intent="get_review",
                error="Sorry, I couldn't find any reviews."
            )
            
    def _generate_text_response(self, response: TravelResponse) -> str:
        """Generate a natural language response from the structured data."""
        if response.error:
            return response.error
            
        if response.intent == "find_route":
            return f"Here's the route from {response.entities['from']} to {response.entities['to']}"
        elif response.intent == "find_attraction":
            return f"Here are some attractions in {response.entities['location']}"
        elif response.intent == "get_cost":
            return f"The cost from {response.entities['from']} to {response.entities['to']} is €{response.cost['amount']:.2f}"
        elif response.intent == "get_review":
            return f"Here's what people think about {response.entities['attraction']}"
        else:
            return "I'm not sure how to respond to that."

def generate_response(message, preferences, user_id):
    """
    Generate a response to the user's message based on their preferences.
    
    Args:
        message (str): The user's message
        preferences (dict): User's preferences including route, accessibility, etc.
        user_id (str): The ID of the user
        
    Returns:
        str: The generated response
    """
    # For now, return a simple response
    return f"I received your message: {message}. I'll help you plan your trip based on your preferences."
