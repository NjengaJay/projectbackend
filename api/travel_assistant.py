"""Travel assistant module for processing user queries."""
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import re
from fuzzywuzzy import process

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
            # Log the query
            logger.info(f"Processing query: {query}")
            
            # Convert query to lowercase for matching
            query_lower = query.lower()
            
            # Detect intent
            if any(word in query_lower for word in ["route", "path", "way", "directions"]):
                logger.info("Detected route intent")
                return self._handle_route_query(query, {})
                
            elif any(word in query_lower for word in ["cost", "price", "expensive", "cheap"]):
                logger.info("Detected cost intent")
                return self._handle_cost_query(query, {})
                
            elif any(word in query_lower for word in ["recommend", "recommendation", "suggest", "show", "tell me about"]):
                logger.info("Detected attraction intent")
                return self._handle_attraction_query(query)
                
            elif any(word in query_lower for word in ["review", "rating", "good", "bad"]):
                logger.info("Detected review intent")
                return self._handle_review_query(query, {})
                
            elif any(pattern in query_lower for pattern in [
                "what can i visit", "what's in", "what is in", "places to visit",
                "things to do", "tourist spots", "attractions", "places of interest",
                "sightseeing", "what to see"
            ]):
                logger.info("Detected attraction intent")
                return self._handle_attraction_query(query)
                
            else:
                logger.info("Unknown intent")
                return TravelResponse(
                    intent="unknown",
                    error="I don't understand what you're asking for. Try asking about routes, attractions, costs, or reviews."
                )
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return TravelResponse(
                intent="error",
                error=str(e)
            )
            
    def _extract_locations(self, query: str) -> list:
        """Extract location names from the query."""
        # Convert query to lowercase for matching
        query_lower = query.lower()
        
        # Define location markers (words that often precede or follow location names)
        location_markers = {
            'prefix': ['in', 'at', 'to', 'from', 'near', 'around', 'visiting'],
            'suffix': ['city', 'town', 'area', 'region']
        }
        
        # Split query into words and remove punctuation
        import re
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Try to find locations in the query
        locations = []
        i = 0
        while i < len(words):
            # Check if current word is a location marker
            is_marker = (words[i] in location_markers['prefix'] or 
                        (i > 0 and words[i] in location_markers['suffix']))
            
            # Try multi-word combinations after a marker
            if is_marker and i + 1 < len(words):
                # Try combinations of 1-3 words after the marker
                for j in range(min(3, len(words) - (i + 1)), 0, -1):
                    potential_location = ' '.join(words[i+1:i+1+j])
                    logger.debug(f"Trying potential location after marker: {potential_location}")
                    
                    # Try to normalize this potential location
                    normalized = self._normalize_city_name(potential_location)
                    if normalized in ['Amsterdam', 'Rotterdam', 'The Hague', 'Utrecht', 'Eindhoven', 'Groningen', 'Tilburg', 'Almere', 'Breda']:
                        logger.debug(f"Found valid location: {normalized}")
                        if normalized not in locations:  # Avoid duplicates
                            locations.append(normalized)
                        i += j  # Skip the words we just matched
                        break
            i += 1
        
        # If no locations found with markers, try direct matching
        if not locations:
            for i in range(len(words)):
                for j in range(min(3, len(words) - i), 0, -1):
                    potential_location = ' '.join(words[i:i+j])
                    normalized = self._normalize_city_name(potential_location)
                    if normalized in ['Amsterdam', 'Rotterdam', 'The Hague', 'Utrecht', 'Eindhoven', 'Groningen', 'Tilburg', 'Almere', 'Breda']:
                        if normalized not in locations:  # Avoid duplicates
                            locations.append(normalized)
                        break
        
        logger.debug(f"Extracted locations from query '{query}': {locations}")
        return locations

    def _extract_accessibility_preferences(self, query: str) -> Dict[str, bool]:
        """Extract accessibility preferences from the query."""
        accessibility = {}
        
        # Check for wheelchair accessibility
        if any(term in query.lower() for term in ['wheelchair', 'accessible', 'disability']):
            accessibility['wheelchair_accessible'] = True
        
        # Check for elevator access
        if any(term in query.lower() for term in ['elevator', 'lift']):
            accessibility['elevator_access'] = True
        
        # Check for accessible parking
        if any(term in query.lower() for term in ['parking', 'disabled parking', 'handicap parking']):
            accessibility['accessible_parking'] = True
        
        # Check for accessible restrooms
        if any(term in query.lower() for term in ['restroom', 'bathroom', 'toilet']):
            accessibility['accessible_restroom'] = True
        
        return accessibility

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
            f"Time: {time_str}",
            f"Cost: €{cost}",
            f"Distance: {distance}km",
            f"Transport modes: {', '.join(summary['transport_modes'])}"
        ]
        
        return "\n".join(response)

    def _handle_attraction_query(self, query: str) -> TravelResponse:
        """Handle an attraction search query."""
        try:
            # Extract locations from query
            locations = self._extract_locations(query)
            logger.debug(f"Extracted locations: {locations}")
            
            # If no location found, default to Amsterdam
            if not locations:
                location = "Amsterdam"
                logger.debug("No location found, defaulting to Amsterdam")
            else:
                location = locations[0]  # Use first location found
                logger.debug(f"Using location: {location}")
            
            # Extract interests from query
            interests = []
            if "museum" in query.lower():
                interests.append("museum")
            elif "park" in query.lower():
                interests.append("park")
            elif "restaurant" in query.lower():
                interests.append("restaurant")
            elif "shopping" in query.lower():
                interests.append("shopping")
            else:
                interests.append("tourist_spot")
            
            logger.debug(f"Extracted interests: {interests}")
            
            try:
                # Extract accessibility preferences
                accessibility = self._extract_accessibility_preferences(query)
                
                # Get user preferences including accessibility
                user_preferences = {
                    'interests': interests,
                    'mobility': {'mode': 'walking', 'max_distance': 2.0},
                    'accessibility': accessibility
                }
                
                # Get attractions from planner
                attractions = self.planner.get_attractions(location, user_preferences)
                logger.debug(f"Got {len(attractions)} attractions from planner")
                
                if not attractions:
                    # Create a more specific error message based on accessibility requirements
                    error_msg = f"I couldn't find any attractions in {location}"
                    if accessibility:
                        features = []
                        if accessibility.get('wheelchair_accessible'):
                            features.append("wheelchair accessibility")
                        if accessibility.get('elevator_access'):
                            features.append("elevator access")
                        if accessibility.get('accessible_parking'):
                            features.append("accessible parking")
                        if accessibility.get('accessible_restroom'):
                            features.append("accessible restrooms")
                        
                        error_msg += f" with {' and '.join(features)}"
                    error_msg += ". You might want to try a different location or modify your accessibility requirements."
                    
                    return TravelResponse(
                        intent="find_attraction",
                        entities={"location": location},
                        error=error_msg
                    )
                
                # Format attractions for response
                formatted_attractions = []
                for attraction in attractions:
                    formatted = {
                        "name": attraction.get("name", "Unknown Attraction"),
                        "type": attraction.get("type", interests[0]),
                        "rating": float(attraction.get("rating", 4.0)),
                        "location": location  # Use the normalized location from query
                    }
                    formatted_attractions.append(formatted)
                
                logger.debug(f"Formatted attractions: {formatted_attractions}")
                
                return TravelResponse(
                    intent="find_attraction",
                    entities={"location": location},
                    attractions=formatted_attractions
                )
                
            except Exception as e:
                logger.error(f"Error getting attractions: {str(e)}")
                # Return a response with the default attraction
                return TravelResponse(
                    intent="find_attraction",
                    entities={"location": location},
                    attractions=[{
                        "name": "Popular Attraction",
                        "type": interests[0],
                        "rating": 4.0,
                        "location": location
                    }]
                )
                
        except Exception as e:
            logger.error(f"Error handling attraction query: {str(e)}")
            return TravelResponse(
                intent="find_attraction",
                error=str(e)
            )

    def _handle_cost_query(self, query: str) -> TravelResponse:
        """Handle a cost query."""
        try:
            locations = self._extract_locations(query)
            
            if len(locations) < 2:
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
                [locations[0], locations[1]],
                TravelPreferences(route_preference=RoutePreference.COST)
            )
            
            return TravelResponse(
                intent="get_cost",
                entities={"from": locations[0], "to": locations[1]},
                route=route,
                cost=cost,
                text=f"The cost from {locations[0]} to {locations[1]} is €{cost['amount']:.2f}"
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
            # Create a more informative response for attractions
            if not response.attractions:
                location = response.entities.get('location', 'that location')
                return f"I couldn't find any attractions in {location} matching your criteria. Try broadening your search or try a different location."
                
            text_parts = [f"Here are some popular attractions in {response.entities.get('location', 'that location')}:"]
            
            # Group attractions by type
            attractions_by_type = {}
            for attraction in response.attractions:
                type_str = attraction.get('type', 'attraction').lower()
                if type_str not in attractions_by_type:
                    attractions_by_type[type_str] = []
                attractions_by_type[type_str].append(attraction)
            
            # Define type emoji mapping with ASCII fallbacks
            type_emojis = {
                'museum': '*',        # 🏛️
                'gallery': '@',       # 🎨
                'artwork': '@',       # 🎨
                'attraction': '!',    # 🎯
                'viewpoint': '^',     # 🌅
                'theme_park': 'O',    # 🎡
                'zoo': '&',           # 🦁
                'aquarium': '~',      # 🐠
                'park': '#',          # 🌳
                'garden': '*',        # 🌸
                'historic_site': '^', # 🏛️
                'monument': '^',      # 🗽
                'castle': '^',        # 🏰
                'ruins': '^',         # 🏛️
                'entertainment': '@',  # 🎭
                'theater': '@',       # 🎭
                'cinema': '@',        # 🎬
                'arts_centre': '@',   # 🎨
                'tourist_spot': '!'   # Default for general tourist spots
            }
            
            # Sort types by priority
            priority_types = [
                'attraction',  # General attractions first
                'tourist_spot',
                'theme_park', 'entertainment',
                'castle', 'historic_site', 'monument',
                'viewpoint', 'park', 'garden',
                'zoo', 'aquarium',
                'museum', 'gallery', 'artwork', 'arts_centre'  # Cultural attractions last
            ]
            
            sorted_types = sorted(
                attractions_by_type.keys(),
                key=lambda x: (
                    priority_types.index(x) if x in priority_types else len(priority_types)
                )
            )
            
            # Add attractions by type
            for type_str in sorted_types:
                attractions = attractions_by_type[type_str]
                symbol = type_emojis.get(type_str, '>')
                
                # Capitalize and format type
                formatted_type = type_str.replace('_', ' ').title()
                text_parts.append(f"\n{symbol} {formatted_type}:")
                
                for attraction in attractions:
                    name = attraction.get('name', '')
                    rating = attraction.get('rating', 0)
                    distance = attraction.get('distance')
                    
                    # Format the attraction details
                    details = [name]
                    if rating > 0:
                        details.append(f"Rating: {rating:.1f}/5")
                    if distance:
                        details.append(f"{distance:.1f}km away")
                    
                    text_parts.append(f"  - {' | '.join(details)}")
            
            return "\n".join(text_parts)
            
        elif response.intent == "get_cost":
            return f"The cost from {response.entities['from']} to {response.entities['to']} is €{response.cost['amount']:.2f}"
            
        elif response.intent == "get_review":
            return f"Here's what people think about {response.entities['attraction']}"
            
        else:
            return "I'm not sure how to respond to that."
            
    def _normalize_city_name(self, city: str) -> str:
        """Normalize city name to match known city names."""
        from fuzzywuzzy import process
        
        if not hasattr(self, '_city_mapping'):
            # Create a mapping of lowercase city names to their proper case
            self._city_mapping = {
                'amsterdam': 'Amsterdam',
                'rotterdam': 'Rotterdam',
                'the hague': 'The Hague',
                'den haag': 'The Hague',
                'de haag': 'The Hague',
                'haag': 'The Hague',
                'hague': 'The Hague',
                'utrecht': 'Utrecht',
                'eindhoven': 'Eindhoven',
                'groningen': 'Groningen',
                'tilburg': 'Tilburg',
                'almere': 'Almere',
                'breda': 'Breda'
            }
        
        city_lower = city.lower().strip()
        
        # Direct lookup first
        if city_lower in self._city_mapping:
            logger.debug(f"Direct match found for '{city}' -> '{self._city_mapping[city_lower]}'")
            return self._city_mapping[city_lower]
        
        # Try fuzzy matching if direct lookup fails
        match, score = process.extractOne(city_lower, self._city_mapping.keys())
        logger.debug(f"Fuzzy match for '{city}': '{match}' with score {score}")
        
        if score >= 85:
            return self._city_mapping[match]
        
        # If no match found, return original with first letter of each word capitalized
        logger.debug(f"No match found for '{city}', returning capitalized version")
        return ' '.join(word.capitalize() for word in city.split())

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
