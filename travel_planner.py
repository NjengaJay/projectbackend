"""
Travel planner module for route planning and recommendations
"""

import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import math

from .recommender.hybrid_recommender import HybridRecommender

logger = logging.getLogger(__name__)

class RoutePreference(Enum):
    """Route preference options"""
    TIME = "time"
    COST = "cost"
    SCENIC = "scenic"

@dataclass
class TravelPreferences:
    """Container for travel preferences"""
    route_preference: RoutePreference
    accessibility_required: bool = False
    max_cost: Optional[float] = None
    scenic_priority: float = 0.0
    time_weight: float = 0.5
    cost_weight: float = 0.5

@dataclass
class RouteSegment:
    """Container for route segment information"""
    from_location: str
    to_location: str
    transport_mode: str
    duration: float
    cost: float
    accessibility_features: List[str]
    scenic_rating: float
    points_of_interest: List[Dict] = None

    def __post_init__(self):
        if self.points_of_interest is None:
            self.points_of_interest = []

def find_fastest_route(source: Tuple[float, float], target: Tuple[float, float], graph: nx.Graph) -> Dict:
    """Find the fastest route between two points"""
    logger.info(f"Finding fastest route from {source} to {target}")
    try:
        # Find nearest nodes to source and target
        source_node = _find_nearest_node(graph, source)
        target_node = _find_nearest_node(graph, target)
        
        # Use NetworkX to find shortest path by time
        path = nx.shortest_path(graph, source=source_node, target=target_node, weight='duration')
        
        # Extract route data
        route_data = {
            "steps": []
        }
        
        for i in range(len(path) - 1):
            edge_data = graph[path[i]][path[i+1]]
            route_data["steps"].append({
                "from_location": edge_data["from_name"],
                "to_location": edge_data["to_name"],
                "mode": edge_data["mode"],
                "duration": edge_data["duration"],
                "cost": edge_data["cost"],
                "accessibility_features": edge_data.get("accessibility_features", []),
                "scenic_rating": edge_data.get("scenic_rating", 0.0)
            })
            
        return route_data
        
    except Exception as e:
        logger.error(f"Error finding fastest route: {str(e)}", exc_info=True)
        raise

def _find_nearest_node(graph: nx.Graph, coordinates: Tuple[float, float]) -> Tuple[float, float]:
    """Find the nearest node in the graph to the given coordinates"""
    min_dist = float('inf')
    nearest_node = None
    
    for node in graph.nodes():
        dist = ((node[0] - coordinates[0])**2 + (node[1] - coordinates[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
            
    return nearest_node

def find_optimal_route(
    source: Tuple[float, float],
    target: Tuple[float, float],
    graph: nx.Graph,
    preferences: TravelPreferences
) -> Dict:
    """Find optimal route based on preferences"""
    logger.info(f"Finding optimal route from {source} to {target} with preferences: {preferences}")
    try:
        # Find nearest nodes to source and target
        source_node = _find_nearest_node(graph, source)
        target_node = _find_nearest_node(graph, target)
        
        # Define edge weight function based on preferences
        def edge_weight(u, v, edge_data):
            weight = 0.0
            
            # Time component
            time_factor = edge_data['duration'] / 3600  # Convert to hours
            weight += preferences.time_weight * time_factor
            
            # Cost component
            if preferences.max_cost and edge_data['cost'] > preferences.max_cost:
                return float('inf')
            cost_factor = edge_data['cost'] / 100  # Normalize by 100
            weight += preferences.cost_weight * cost_factor
            
            # Scenic component
            if preferences.route_preference == RoutePreference.SCENIC:
                scenic_factor = 1 - edge_data.get('scenic_rating', 0)
                weight += preferences.scenic_priority * scenic_factor
            
            # Accessibility component
            if preferences.accessibility_required and not edge_data.get('accessibility_features'):
                return float('inf')
                
            return weight
            
        # Find shortest path using custom weight function
        path = nx.shortest_path(graph, source=source_node, target=target_node, weight=edge_weight)
        
        # Extract route data
        route_data = {
            "steps": []
        }
        
        for i in range(len(path) - 1):
            edge_data = graph[path[i]][path[i+1]]
            route_data["steps"].append({
                "from_location": edge_data["from_name"],
                "to_location": edge_data["to_name"],
                "mode": edge_data["mode"],
                "duration": edge_data["duration"],
                "cost": edge_data["cost"],
                "accessibility_features": edge_data.get("accessibility_features", []),
                "scenic_rating": edge_data.get("scenic_rating", 0.0)
            })
            
        return route_data
        
    except Exception as e:
        logger.error(f"Error finding optimal route: {str(e)}", exc_info=True)
        raise

class TravelPlanner:
    """Plans travel routes and provides recommendations"""
    
    def __init__(self):
        """Initialize the travel planner with Dutch cities and transport modes."""
        self.dutch_cities = {
            "amsterdam": {
                "name": "Amsterdam",
                "lat": 52.3676,
                "lon": 4.9041
            },
            "rotterdam": {
                "name": "Rotterdam",
                "lat": 51.9244,
                "lon": 4.4777
            },
            "utrecht": {
                "name": "Utrecht",
                "lat": 52.0907,
                "lon": 5.1214
            },
            "den haag": {
                "name": "Den Haag",
                "lat": 52.0705,
                "lon": 4.3007
            },
            "eindhoven": {
                "name": "Eindhoven",
                "lat": 51.4416,
                "lon": 5.4697
            }
        }
        
        self.transport_modes = {
            "intercity_train": {
                "name": "Intercity Train",
                "speed": 140,  # km/h
                "cost_per_km": 0.19,  # euros
                "accessibility": {
                    "wheelchair_accessible": True,
                    "step_free_access": True,
                    "accessible_toilets": True,
                    "priority_seating": True
                }
            },
            "regional_train": {
                "name": "Regional Train",
                "speed": 100,  # km/h
                "cost_per_km": 0.15,  # euros
                "accessibility": {
                    "wheelchair_accessible": True,
                    "step_free_access": True,
                    "accessible_toilets": False,
                    "priority_seating": True
                }
            },
            "high_speed_train": {
                "name": "High-Speed Train",
                "speed": 250,  # km/h
                "cost_per_km": 0.35,  # euros
                "accessibility": {
                    "wheelchair_accessible": True,
                    "step_free_access": True,
                    "accessible_toilets": True,
                    "priority_seating": True
                }
            },
            "bus": {
                "name": "Bus",
                "speed": 60,  # km/h
                "cost_per_km": 0.12,  # euros
                "accessibility": {
                    "wheelchair_accessible": True,
                    "step_free_access": False,
                    "accessible_toilets": False,
                    "priority_seating": True
                }
            }
        }
            
        # Initialize route graph
        self.route_graph = nx.Graph()
            
        # Add nodes and edges to graph with scenic attributes
        for city, data in self.dutch_cities.items():
            self.route_graph.add_node(city, **data)
            
        # Add connections with scenic weights
        for (city1, city2) in [("amsterdam", "rotterdam"), ("rotterdam", "utrecht"), ("utrecht", "den haag"), ("den haag", "eindhoven")]:
            self.route_graph.add_edge(
                city1, 
                city2, 
                scenic_score=8.5,
                recommended_mode="intercity_train",
                highlights=["Dutch countryside", "Windmills"]
            )
            
        logger.info("TravelPlanner initialized with scenic route data")
            
    def _calculate_distance(self, start: str, end: str) -> float:
        """Calculate distance between two cities."""
        start_data = self.dutch_cities[start.lower()]
        end_data = self.dutch_cities[end.lower()]
        
        # Calculate distance using Haversine formula
        R = 6371  # Earth's radius in kilometers
        
        lat1 = math.radians(start_data["lat"])
        lon1 = math.radians(start_data["lon"])
        lat2 = math.radians(end_data["lat"])
        lon2 = math.radians(end_data["lon"])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c  # Distance in kilometers

    def get_route(self, start: str, end: str, preferences: Dict = None) -> Dict:
        """Get a route between two cities."""
        logger.info(f"Getting route from {start} to {end} with preferences: {preferences}")
        try:
            # Validate locations
            if not self._validate_location(start) or not self._validate_location(end):
                return self._handle_invalid_locations(start, end)

            # Get route steps
            mode = "intercity_train"  # Default to intercity train
            steps = []
            
            # Calculate distance and duration
            distance = self._calculate_distance(start, end)
            mode_data = self.transport_modes[mode]
            duration = (distance / mode_data["speed"]) * 60  # Convert to minutes
            cost = distance * mode_data["cost_per_km"]
            
            # Get accessibility features
            accessibility_features = [
                feature.replace("_", " ").title()  # Format feature names nicely
                for feature, enabled in mode_data["accessibility"].items()
                if enabled
            ]

            # Create route step
            step = {
                "from_location": start,
                "to_location": end,
                "mode": mode_data["name"],
                "duration": round(duration),
                "distance": round(distance, 1),
                "cost": round(cost, 2),
                "accessibility_features": accessibility_features,
                "description": (
                    f"Take the {mode_data['name']} from {start.title()} to {end.title()}. "
                    f"This route is equipped with: {', '.join(accessibility_features)}."
                )
            }
            
            steps.append(step)
            
            # Calculate totals
            total_duration = sum(step["duration"] for step in steps)
            total_cost = sum(step["cost"] for step in steps)
            
            # Create response message
            message = (
                f"Take the {mode_data['name']} from {start.title()} to {end.title()}. "
                f"Duration: {round(duration)} minutes. Distance: {round(distance, 1)} km. "
                f"Cost: €{round(cost, 2)}. Available features: {', '.join(accessibility_features)}"
            )
            
            return {
                "status": "success",
                "message": message,
                "steps": steps,
                "total_duration": total_duration,
                "total_cost": total_cost
            }
            
        except Exception as e:
            logger.error(f"Error getting route: {str(e)}", exc_info=True)
            return {"error": "An error occurred while planning your route. Please try again."}

    def _calculate_accessibility_score(self, start: str, end: str, mode: str) -> Tuple[float, Dict]:
        """Calculate accessibility score for a route."""
        # Get accessibility features for stations and transport
        start_station = self.dutch_cities[start].get('accessibility', {})
        end_station = self.dutch_cities[end].get('accessibility', {})
        transport = self.transport_modes[mode].get('accessibility', {})
        
        # Calculate score based on available features
        score = 0.0
        features = {
            'wheelchair_accessible': transport.get('wheelchair_accessible', False),
            'step_free_access': transport.get('step_free_access', False),
            'accessible_toilets': transport.get('accessible_toilets', False),
            'assistance_available': transport.get('assistance_available', False)
        }
        
        # Add points for each feature
        if features['wheelchair_accessible']:
            score += 3.0
        if features['step_free_access']:
            score += 2.5
        if features['accessible_toilets']:
            score += 2.0
        if features['assistance_available']:
            score += 2.5
            
        # Add points for station features
        for station in [start_station, end_station]:
            if station.get('elevator_access'):
                score += 1.0
            if station.get('wheelchair_friendly_stations'):
                score += 1.0
            if station.get('assistance_desk'):
                score += 1.0
            if station.get('accessible_parking'):
                score += 1.0
                
        # Normalize score to 0-10 range
        max_score = 15.0  # Maximum possible score
        score = min((score / max_score) * 10, 10)
        
        return score, features

    def _calculate_scenic_score(self, city1: str, city2: str, mode: str) -> float:
        """
        Calculate scenic score for a route segment based on:
        - City scenic scores
        - Transport mode scenic score
        - Predefined scenic route score
        - Presence of water routes if applicable
        - Historic centers and landmarks
        
        Returns:
            float: Scenic score from 0-10
        """
        try:
            base_score = 0
            
            # Get city data
            city1_data = self.dutch_cities.get(city1, {})
            city2_data = self.dutch_cities.get(city2, {})
            
            # Get transport mode data
            mode_data = self.transport_modes.get(mode, {})
            
            # Get predefined scenic route data
            route_data = self.route_graph.get_edge_data(city1, city2)
            
            # Calculate score components
            city_score = (city1_data.get('scenic_score', 0) + city2_data.get('scenic_score', 0)) / 2
            transport_score = mode_data.get('scenic_score', 5)
            route_score = route_data.get('scenic_score', 6)
            
            # Bonus for water routes if transport mode supports it
            water_bonus = 1 if (
                mode_data.get('requires_water') and 
                city1_data.get('water_routes') and 
                city2_data.get('water_routes')
            ) else 0
            
            # Bonus for historic centers
            historic_bonus = 0.5 if (
                city1_data.get('historic_center') or 
                city2_data.get('historic_center')
            ) else 0
            
            # Calculate weighted average
            base_score = (
                city_score * 0.3 +
                transport_score * 0.3 +
                route_score * 0.4
            )
            
            # Add bonuses
            final_score = min(10, base_score + water_bonus + historic_bonus)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating scenic score: {str(e)}", exc_info=True)
            return 5.0  # Default middle score
            
    def _find_scenic_route(self, start: str, end: str) -> List[Dict]:
        """
        Find the most scenic route between two points using:
        1. Predefined scenic routes where available
        2. Scenic scores for cities and transport modes
        3. Special features like water routes and historic centers
        
        Returns:
            List of route segments with transport modes and descriptions
        """
        try:
            route_segments = []
            current_city = start
            
            # Check if there's a direct scenic route
            if (start, end) in self.route_graph.edges():
                route_data = self.route_graph.get_edge_data(start, end)
                mode = route_data['recommended_mode']
                mode_data = self.transport_modes[mode]
                
                distance = self._calculate_distance(start, end)
                duration = (distance / mode_data['speed']) * 60
        
                return [{
                    "segment": 1,
                    "mode": mode,
                    "from": start,
                    "to": end,
                    "duration": duration,
                    "cost": distance * mode_data['cost_per_km'],
                    "distance": distance,
                    "scenic_score": route_data['scenic_score'],
                    "scenic_points": route_data['highlights'],
                    "description": f"Scenic {mode.replace('_', ' ')} journey with views of {', '.join(route_data['highlights'])}"
                }]
            
            # Otherwise, find scenic route through intermediate cities
            scenic_path = []
            
            # Use NetworkX to find path maximizing scenic scores
            def scenic_weight(u, v, d):
                return 1 / (self._calculate_scenic_score(u, v, d.get('recommended_mode', 'local_train')) + 0.1)
            
            try:
                scenic_path = nx.shortest_path(
                    self.route_graph, 
                    start, 
                    end, 
                    weight=scenic_weight
                )
            except nx.NetworkXNoPath:
                raise ValueError(f"No route found between {start} and {end}")
            
            # Convert path to route segments
            for i in range(len(scenic_path) - 1):
                from_city = scenic_path[i]
                to_city = scenic_path[i + 1]
                
                # Get route data
                route_data = self.route_graph.get_edge_data(from_city, to_city)
                mode = route_data.get('recommended_mode', 'local_train')
                mode_data = self.transport_modes[mode]
                
                distance = self._calculate_distance(from_city, to_city)
                duration = (distance / mode_data['speed']) * 60  # Convert to minutes
                cost = distance * mode_data['cost_per_km']
                
                segment = {
                    "segment": i + 1,
                    "mode": mode,
                    "from": from_city,
                    "to": to_city,
                    "duration": duration,
                    "cost": cost,
                    "distance": distance,
                    "scenic_score": route_data.get('scenic_score', 
                        self._calculate_scenic_score(from_city, to_city, mode)),
                    "scenic_points": route_data.get('highlights', 
                        [f"Views of {self.dutch_cities[to_city]['name']}"]),
                    "description": route_data.get('description', 
                        f"Scenic {mode.replace('_', ' ')} journey to {to_city}")
                }
                
                route_segments.append(segment)
            
            return route_segments
            
        except Exception as e:
            logger.error(f"Error finding scenic route: {str(e)}", exc_info=True)
            return []

    def _get_route_steps(self, start: str, end: str, mode: str, preferences: Dict) -> List[Dict]:
        """Generate detailed steps for a route."""
        try:
            logger.info(f"Generating route steps from {start} to {end} with mode {mode}")
            
            # Get city data
            start_city = self.dutch_cities[start]
            end_city = self.dutch_cities[end]
            
            # Get transport mode data
            transport = self.transport_modes[mode]
            logger.info(f"Selected transport mode: {transport}")
            
            # Calculate distance using Haversine formula
            from math import radians, sin, cos, sqrt, atan2
            
            lat1, lon1 = radians(start_city["lat"]), radians(start_city["lon"])
            lat2, lon2 = radians(end_city["lat"]), radians(end_city["lon"])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = 6371 * c  # Earth's radius * c
            
            # Calculate duration and cost
            duration = (distance / transport["speed"]) * 60  # Convert to minutes
            cost = distance * transport["cost_per_km"]
            
            # Format accessibility features
            accessibility_features = []
            feature_names = {
                "wheelchair_accessible": "Wheelchair Accessible",
                "step_free_access": "Step-Free Access",
                "accessible_toilets": "Accessible Toilets",
                "priority_seating": "Priority Seating"
            }
            
            for key, display_name in feature_names.items():
                if transport["accessibility"].get(key, False):
                    accessibility_features.append(display_name)
            
            # Create route step
            route_step = {
                "from": start_city["name"],
                "to": end_city["name"],
                "transport_mode": transport["name"],
                "duration": round(duration, 1),
                "distance": round(distance, 1),
                "cost": round(cost, 2),
                "accessibility_features": accessibility_features,
                "description": (
                    f"Take the {transport['name']} from {start_city['name']} to {end_city['name']}. "
                    f"Duration: {round(duration)} minutes. Distance: {round(distance, 1)} km. Cost: €{round(cost, 2)}. "
                    f"Available features: {', '.join(accessibility_features)}."
                )
            }
            
            logger.info(f"Generated route step: {route_step}")
            return [route_step]
            
        except Exception as e:
            logger.error(f"Error generating route steps: {str(e)}", exc_info=True)
            raise

    def _select_transport_mode(self, preferences: Dict) -> str:
        """Select the best transport mode based on preferences."""
        if preferences.get("accessible"):
            return "intercity_train"  # Trains have the best accessibility features
        elif preferences.get("scenic"):
            return "regional_train"  # Regional trains offer scenic routes
        elif preferences.get("fast"):
            return "high_speed_train"  # High-speed trains are fastest
        elif preferences.get("cheap"):
            return "bus"  # Buses are typically cheapest
        else:
            return "intercity_train"  # Default to intercity train

    def _get_scenic_route(self, start: str, end: str, mode: Optional[str] = None, preferences: Optional[Dict] = None) -> Dict:
        """Get scenic route between two points."""
        route_segments = self._find_scenic_route(start, end)
        
        if not route_segments:
            return {"error": "Could not find a scenic route between these cities"}
        
        # Calculate totals
        total_duration = sum(segment["duration"] for segment in route_segments)
        total_cost = sum(segment["cost"] for segment in route_segments)
        total_distance = sum(segment["distance"] for segment in route_segments)
        avg_scenic_score = sum(segment["scenic_score"] for segment in route_segments) / len(route_segments)
        
        # Get direct route for comparison
        direct_distance = self._calculate_distance(start, end)
        direct_duration = (direct_distance / self.transport_modes['intercity_train']['speed']) * 60
        
        description = (
            f"Scenic route with average scenic score of {avg_scenic_score:.1f}/10. "
            f"This route takes {total_duration:.0f} minutes compared to {direct_duration:.0f} minutes for the direct train, "
            f"but offers spectacular views and experiences through multiple modes of transport."
        )
        
        return {
            "route": route_segments,
            "description": description,
            "total_duration": total_duration,
            "total_cost": total_cost,
            "total_distance": total_distance,
            "scenic_score": avg_scenic_score
        }
            
    def _find_nearest_dutch_city(self, location: str) -> Optional[str]:
        """
        Find the nearest Dutch city to a given location using mock coordinates.
        In a real implementation, this would use actual geocoding and distance calculations.
        
        Args:
            location: Location to find nearest Dutch city for
            
        Returns:
            Name of nearest Dutch city or None if location is invalid
        """
        # Mock coordinates for some foreign cities
        foreign_cities = {
            'berlin': {'lat': 52.5200, 'lon': 13.4050},
            'paris': {'lat': 48.8566, 'lon': 2.3522},
            'london': {'lat': 51.5074, 'lon': -0.1278},
            'brussels': {'lat': 50.8503, 'lon': 4.3517},
            'frankfurt': {'lat': 50.1109, 'lon': 8.6821},
            'copenhagen': {'lat': 55.6761, 'lon': 12.5683}
        }
        
        # Get coordinates for foreign city
        location = location.lower()
        if location not in foreign_cities:
            return None
            
        foreign_coords = foreign_cities[location]
        
        # Find nearest Dutch city by calculating distances
        nearest_city = None
        min_distance = float('inf')
        
        for dutch_city, coords in self.dutch_cities.items():
            # Simple Euclidean distance - in real implementation use Haversine formula
            distance = ((coords['lat'] - foreign_coords['lat']) ** 2 + 
                       (coords['lon'] - foreign_coords['lon']) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_city = dutch_city
                
        return nearest_city

    def _validate_location(self, location: str) -> Optional[Dict[str, float]]:
        """
        Validate if location is in Netherlands and return coordinates
        
        Args:
            location: Location name to validate
            
        Returns:
            Dictionary with lat/lon if valid Dutch location, None otherwise
        """
        # Convert to lowercase for comparison
        location = location.lower()
        
        # Check if location is in Dutch cities
        if location in self.dutch_cities:
            return self.dutch_cities[location]
            
        return None

    def get_recommendations(self, location: str, attraction_type: Optional[str] = None) -> List[str]:
        """
        Get recommendations for attractions
        
        Args:
            location: Location to get recommendations for
            attraction_type: Optional type of attraction
            
        Returns:
            List of recommendations
        """
        try:
            # Mock recommendations
            recommendations = [
                f"The {attraction_type or 'attractions'} in {location} are amazing! Must visit.",
                f"Really enjoyed exploring the {attraction_type or 'places'} in {location}.",
                f"Great experience visiting {location}'s {attraction_type or 'tourist spots'}.",
                f"Wonderful {attraction_type or 'destination'}! {location} exceeded expectations."
            ]
            
            return recommendations[:3]  # Return top 3
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            return []

    def _get_route_function(self, preferences: TravelPreferences):
        """Get appropriate route finding function based on preferences"""
        if preferences.route_preference == RoutePreference.TIME:
            return find_fastest_route
        else:
            return find_optimal_route
            
    def plan_journey(
        self,
        start_location: str,
        end_location: str,
        preferences: TravelPreferences,
        include_recommendations: bool = False
    ) -> List[RouteSegment]:
        """
        Plan a journey based on preferences
        
        Args:
            start_location: Starting location name
            end_location: Ending location name
            preferences: Travel preferences
            include_recommendations: Whether to include POI recommendations
            
        Returns:
            List of route segments
        """
        logger.info(f"Planning journey from {start_location} to {end_location}")
        try:
            # Validate locations
            start_coordinates = self._validate_location(start_location)
            end_coordinates = self._validate_location(end_location)
            
            # Get appropriate route function
            route_func = self._get_route_function(preferences)
            
            # Find route
            if preferences.route_preference == RoutePreference.TIME:
                route_data = route_func(start_coordinates, end_coordinates, self.route_graph)
            else:
                route_data = route_func(
                    start_coordinates,
                    end_coordinates,
                    self.route_graph,
                    preferences=preferences
                )
            
            # Convert to route segments
            segments = []
            for step in route_data["steps"]:
                segment = RouteSegment(
                    from_location=step["from_location"],
                    to_location=step["to_location"],
                    transport_mode=step["mode"],
                    duration=step["duration"],
                    cost=step["cost"],
                    accessibility_features=step["accessibility_features"],
                    scenic_rating=step["scenic_rating"]
                )
                
                # Add recommendations if requested
                if include_recommendations:
                    segment.points_of_interest = self.recommender.get_recommendations(
                        location=(
                            (start_coordinates['lat'] + end_coordinates['lat']) / 2,
                            (start_coordinates['lon'] + end_coordinates['lon']) / 2
                        ),
                        radius=5000  # 5km radius
                    )
                    
                segments.append(segment)
                
            return segments
            
        except Exception as e:
            logger.error(f"Error planning journey: {str(e)}", exc_info=True)
            raise

    def _handle_invalid_locations(self, start: str, end: str) -> Dict:
        """Handle invalid location errors with suggestions."""
        invalid_locations = []
        suggestions = {}
        
        # Check if locations are in Netherlands
        if not self._validate_location(start):
            invalid_locations.append(start)
        if not self._validate_location(end):
            invalid_locations.append(end)
            
        if invalid_locations:
            # Create a helpful message with available Dutch cities
            available_cities = ", ".join([city.title() for city in self.dutch_cities.keys()])
            error_msg = (
                f"I apologize, but I can only provide routes within the Netherlands. "
                f"The following location(s) are not in my database: {', '.join(invalid_locations)}.\n\n"
                f"I can help you plan routes between these Dutch cities: {available_cities}.\n\n"
                f"For example, try asking: 'How do I get from Amsterdam to Rotterdam?'"
            )
            return {"error": error_msg}
            
        return None
