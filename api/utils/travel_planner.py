"""Travel Planner module for handling route planning and POI recommendations."""
import logging
import os
import sys
import pandas as pd
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Use relative import for recommender
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recommender.hybrid_recommender import HybridRecommender

logger = logging.getLogger(__name__)

# City coordinates mapping
CITY_COORDINATES = {
    'amsterdam': (52.3676, 4.9041),
    'rotterdam': (51.9225, 4.4792),
    'the hague': (52.0705, 4.3007),
    'utrecht': (52.0907, 5.1214),
    'eindhoven': (51.4416, 5.4697),
    'groningen': (53.2194, 6.5665),
    'tilburg': (51.5719, 5.0672),
    'almere': (52.3508, 5.2647),
    'breda': (51.5719, 4.7683),
    'nijmegen': (51.8426, 5.8546),
    'enschede': (52.2215, 6.8937),
    'apeldoorn': (52.2112, 5.9699),
    'haarlem': (52.3874, 4.6462),
    'arnhem': (51.9851, 5.8987),
    'zaanstad': (52.4537, 4.8137),
    'amersfoort': (52.1561, 5.3878),
    'haarlemmermeer': (52.3080, 4.6897),
    'den bosch': (51.6978, 5.3037),
    'zwolle': (52.5168, 6.0830),
    'zoetermeer': (52.0574, 4.4944),
    'leiden': (52.1601, 4.4970),
    'maastricht': (50.8514, 5.6910),
    'dordrecht': (51.8133, 4.6697),
    'ede': (52.0402, 5.6659),
    'alphen aan den rijn': (52.1343, 4.6640),
    'westland': (51.9930, 4.2831),
    'alkmaar': (52.6324, 4.7534),
    'emmen': (52.7792, 6.9061),
    'delft': (52.0116, 4.3571),
    'venlo': (51.3704, 6.1720),
    'deventer': (52.2660, 6.1552),
    'sittard-geleen': (51.0017, 5.8716),
    'helmond': (51.4793, 5.6570),
    'oss': (51.7654, 5.5196),
    'amstelveen': (52.3114, 4.8725)
}

class RoutePreference(str, Enum):
    TIME = "time"
    COST = "cost"
    SCENIC = "scenic"

@dataclass
class TravelPreferences:
    route_preference: RoutePreference = RoutePreference.TIME
    accessibility_required: bool = False
    scenic_priority: float = 0.5

class DummyPlanner:
    """Lightweight planner implementation without graph dependencies."""
    def plan_route(self, *args, **kwargs):
        return {
            "status": "info", 
            "message": "Using simplified route planner. Full routing capabilities coming soon.",
            "estimated_time": "2 hours",  # Dummy estimate
            "route": [
                {"type": "train", "from": args[0], "to": args[1], "duration": "2 hours"}
            ]
        }
        
    def get_attractions(self, location, interests):
        attractions = []
        if "museum" in interests:
            attractions.append({
                "name": "Sample Museum",
                "type": "museum",
                "rating": 4.5,
                "location": location,
                "description": "A fascinating museum in " + location
            })
        if "park" in interests:
            attractions.append({
                "name": "City Park",
                "type": "park",
                "rating": 4.3,
                "location": location,
                "description": "Beautiful park in " + location
            })
        if "restaurant" in interests:
            attractions.append({
                "name": "Local Restaurant",
                "type": "restaurant",
                "rating": 4.4,
                "location": location,
                "description": "Popular local cuisine in " + location
            })
        return attractions
                
    def _get_city_coordinates(self, city):
        return CITY_COORDINATES.get(city.lower())

class TravelPlanner:
    """Travel planner class for handling route planning and POI recommendations."""
    
    def __init__(self, recommender=None):
        """Initialize the travel planner with basic components."""
        try:
            logger.info("Initializing components...")
            
            # Use provided recommender or create new one
            if recommender is not None:
                self.recommender = recommender
                logger.info("Using provided recommender instance")
            else:
                logger.warning("No recommender provided, using basic planner only")
                self.recommender = None
                
            # Initialize the dummy planner
            self.planner = DummyPlanner()
                
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}", exc_info=True)
            logger.error(f"Current working directory: {os.getcwd()}")
            self.planner = DummyPlanner()
            self.recommender = None

    def _get_city_coordinates(self, city: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a city name."""
        return self.planner._get_city_coordinates(city)
        
    def plan_route(self, start_city: str, end_city: str, preferences: Optional[TravelPreferences] = None) -> Dict:
        """Plan a route between two cities with given preferences."""
        try:
            if not preferences:
                preferences = TravelPreferences()
                
            return self.planner.plan_route(start_city, end_city, preferences)
            
        except Exception as e:
            logger.error(f"Error finding route: {str(e)}")
            return {
                "status": "error",
                "message": f"Error finding route: {str(e)}"
            }
        
    def get_attractions(
        self,
        location: str,
        user_preferences: Dict
    ) -> List[Dict]:
        """Get attractions for a location."""
        try:
            logger.info(f"Getting attractions for location: {location}, interests: {user_preferences.get('interests', [])}")
            
            # Format user preferences
            formatted_preferences = {
                'interests': user_preferences.get('interests', ['tourist_spot']),
                'mobility': user_preferences.get('mobility', {'mode': 'walking', 'max_distance': 2.0}),
                'accessibility': user_preferences.get('accessibility', {})
            }
            logger.info(f"Created user preferences: {formatted_preferences}")
            
            # Call recommender.get_recommendations
            logger.info("Calling recommender.get_recommendations...")
            recommendations = self.recommender.get_recommendations(
                location=location,
                user_preferences=formatted_preferences
            )
            logger.info(f"Got {len(recommendations)} recommendations")
            
            # If no recommendations found, return empty list
            if not recommendations:
                logger.info("No recommendations found, returning empty list")
                return []
            
            # Format attractions for response, skipping any with unknown names
            formatted_attractions = []
            for rec in recommendations:
                name = rec.get('name', '').strip()
                if not name or name.lower() == 'unknown':
                    logger.warning(f"Skipping recommendation due to invalid name: {name}")
                    continue
                
                attraction = {
                    'name': name,
                    'type': rec.get('type', 'attraction').lower(),
                    'rating': rec.get('rating', 0),
                    'location': location
                }
                
                # Add distance if available
                if 'distance' in rec and rec['distance'] is not None:
                    attraction['distance'] = rec['distance']
                
                formatted_attractions.append(attraction)
            
            logger.info(f"Returning {len(formatted_attractions)} formatted attractions")
            return formatted_attractions
            
        except Exception as e:
            logger.error(f"Error getting attractions: {str(e)}", exc_info=True)
            return []
