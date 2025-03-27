from typing import Dict, List, Tuple
import heapq
from math import radians, sin, cos, sqrt, atan2

class RouteOptimizer:
    def __init__(self):
        self.destinations = {}  # Graph representation
        self.accessibility_weights = {
            'wheelchair_friendly': 0.7,
            'step_free': 0.8,
            'accessible_transport': 0.6,
            'accessible_parking': 0.7,
            'accessible_restrooms': 0.8
        }

    def add_destination(self, id: int, lat: float, lng: float, accessibility_features: List[str]):
        """
        Add a destination to the graph with its coordinates and accessibility features
        """
        self.destinations[id] = {
            'coordinates': (lat, lng),
            'accessibility': accessibility_features,
            'connections': {}  # Will store connected destinations and weights
        }

    def connect_destinations(self):
        """
        Create connections between all destinations with weights based on distance and accessibility
        """
        destination_ids = list(self.destinations.keys())
        
        for i in range(len(destination_ids)):
            for j in range(i + 1, len(destination_ids)):
                id1, id2 = destination_ids[i], destination_ids[j]
                
                # Calculate distance-based weight
                distance = self._calculate_distance(
                    self.destinations[id1]['coordinates'],
                    self.destinations[id2]['coordinates']
                )
                
                # Calculate accessibility-based weight
                accessibility_score = self._calculate_accessibility_score(
                    self.destinations[id1]['accessibility'],
                    self.destinations[id2]['accessibility']
                )
                
                # Combined weight (lower is better)
                weight = distance * (2 - accessibility_score)
                
                # Add bidirectional connections
                self.destinations[id1]['connections'][id2] = weight
                self.destinations[id2]['connections'][id1] = weight

    def find_optimal_route(self, start_id: int, destinations: List[int]) -> Tuple[List[int], float]:
        """
        Find the optimal route through specified destinations using Dijkstra's algorithm
        
        Args:
            start_id: Starting destination ID
            destinations: List of destination IDs to visit
            
        Returns:
            Tuple of (optimal route, total distance)
        """
        if start_id not in self.destinations:
            raise ValueError("Start destination not found")
            
        # Initialize variables for Dijkstra's algorithm
        distances = {dest: float('infinity') for dest in self.destinations}
        distances[start_id] = 0
        pq = [(0, start_id)]
        previous = {dest: None for dest in self.destinations}
        visited = set()
        
        # Run Dijkstra's algorithm
        while pq and len(visited) < len(destinations):
            current_distance, current_id = heapq.heappop(pq)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            
            # Check all connections from current destination
            for next_id, weight in self.destinations[current_id]['connections'].items():
                if next_id in visited:
                    continue
                    
                distance = current_distance + weight
                
                if distance < distances[next_id]:
                    distances[next_id] = distance
                    previous[next_id] = current_id
                    heapq.heappush(pq, (distance, next_id))
        
        # Reconstruct the optimal route
        route = []
        current = destinations[-1]  # End at the last destination
        
        while current is not None:
            route.append(current)
            current = previous[current]
            
        route.reverse()
        
        return route, distances[destinations[-1]]

    @staticmethod
    def _calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate the Haversine distance between two coordinates
        """
        lat1, lon1 = map(radians, coord1)
        lat2, lon2 = map(radians, coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # Radius of Earth in kilometers
        r = 6371
        
        return c * r

    def _calculate_accessibility_score(self, features1: List[str], features2: List[str]) -> float:
        """
        Calculate accessibility score between two destinations based on their features
        """
        common_features = set(features1) & set(features2)
        score = sum(self.accessibility_weights.get(feature, 0) for feature in common_features)
        
        # Normalize score to [0, 1]
        max_possible_score = sum(self.accessibility_weights.values())
        return score / max_possible_score if max_possible_score > 0 else 0

    def optimize_multi_destination_trip(self, destinations: List[int], 
                                     accessibility_preference: str = 'standard') -> Dict:
        """
        Optimize a multi-destination trip considering accessibility preferences
        """
        if len(destinations) < 2:
            raise ValueError("Need at least 2 destinations for route optimization")
            
        # Adjust weights based on accessibility preference
        if accessibility_preference == 'high':
            for feature in self.accessibility_weights:
                self.accessibility_weights[feature] *= 1.5
        elif accessibility_preference == 'essential':
            for feature in self.accessibility_weights:
                self.accessibility_weights[feature] *= 2.0
                
        # Find optimal route
        optimal_route, total_distance = self.find_optimal_route(
            destinations[0],  # Start with first destination
            destinations
        )
        
        # Calculate detailed route information
        route_details = []
        for i in range(len(optimal_route) - 1):
            current_id = optimal_route[i]
            next_id = optimal_route[i + 1]
            
            route_details.append({
                'from_id': current_id,
                'to_id': next_id,
                'distance': self._calculate_distance(
                    self.destinations[current_id]['coordinates'],
                    self.destinations[next_id]['coordinates']
                ),
                'accessibility_score': self._calculate_accessibility_score(
                    self.destinations[current_id]['accessibility'],
                    self.destinations[next_id]['accessibility']
                )
            })
            
        return {
            'optimal_route': optimal_route,
            'total_distance': total_distance,
            'route_details': route_details
        }
