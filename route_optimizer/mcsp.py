from typing import Dict, List, Tuple, Optional, Set, Any
import networkx as nx
import numpy as np
import math
from dataclasses import dataclass
from .graph_model import TransportGraph, Node, Edge
from .routing import Route, RouteStep
import heapq

@dataclass
class RouteMetrics:
    time: float
    cost: float
    scenic_score: float
    accessibility: float
    weather_score: float
    traffic_score: float

@dataclass
class RouteStep:
    """A step in a route, representing movement between two nodes"""
    source: str
    target: str
    transport_mode: str
    travel_time: float
    cost: float
    distance: float
    wheelchair_accessible: bool = True
    scenic_value: float = 0.0

    @property
    def start(self) -> str:
        """Alias for source"""
        return self.source

    @property
    def end(self) -> str:
        """Alias for target"""
        return self.target


class RouteResult:
    """Result of a route search, containing steps and summary information"""
    def __init__(self, steps: List[RouteStep]):
        self.steps = steps
        self._summary = None

    @property
    def summary(self) -> Dict[str, float]:
        """Calculate and cache summary statistics for the route"""
        if self._summary is None:
            if not self.steps:
                self._summary = {
                    'total_time': 0,
                    'total_cost': 0,
                    'total_distance': 0,
                    'unique_modes': set(),
                    'mode_switches': 0,
                    'avg_scenic_value': 0
                }
            else:
                total_time = sum(step.travel_time for step in self.steps)
                total_cost = sum(step.cost for step in self.steps)
                total_distance = sum(step.distance for step in self.steps)
                unique_modes = set(step.transport_mode for step in self.steps)
                mode_switches = sum(1 for i in range(len(self.steps)-1) 
                                if self.steps[i].transport_mode != self.steps[i+1].transport_mode)
                avg_scenic_value = sum(step.scenic_value for step in self.steps) / len(self.steps)

                self._summary = {
                    'total_time': total_time,
                    'total_cost': total_cost,
                    'total_distance': total_distance,
                    'unique_modes': unique_modes,
                    'mode_switches': mode_switches,
                    'avg_scenic_value': avg_scenic_value
                }
        return self._summary


class MCSpRouter:
    """Multi-Criteria Shortest Path Router"""
    def __init__(self, graph: TransportGraph):
        self.graph = graph
        self.weather_conditions = {}
        self.traffic_conditions = {}
        self.historical_data = {}

    def find_optimal_route(self, source: str, target: str, weights: Dict[str, float]) -> Optional[RouteResult]:
        """Find optimal route between source and target nodes using given weights"""
        # Validate required weights
        required_weights = {'time', 'cost'}
        if not all(w in weights for w in required_weights):
            raise ValueError(f"Missing required weights: {required_weights - set(weights.keys())}")

        # Validate weights
        if not all(w >= 0 for w in weights.values()):
            raise ValueError("All weights must be non-negative")
        if sum(weights.values()) == 0:
            raise ValueError("At least one weight must be positive")
        if sum(weights.values()) > 1.0:
            raise ValueError("Sum of weights must not exceed 1.0")
        if source not in self.graph.graph or target not in self.graph.graph:
            raise ValueError("Source and target nodes must exist in graph")

        # Try single-mode routes first for extreme weights
        if weights.get('time', 0) > 0.8:
            route = self._find_single_mode_route(source, target, weights, "train")
            if route and route.steps:
                return route
        elif weights.get('cost', 0) > 0.8:
            route = self._find_single_mode_route(source, target, weights, "bus")
            if route and route.steps:
                return route

        # Fall back to multi-mode route
        route = self._find_multi_mode_route(source, target, weights)
        return route if route and route.steps else None

    def find_shortest_path(self, source: str, target: str, weight: str = 'distance') -> Optional[RouteResult]:
        """Find shortest path between two nodes using specified weight"""
        try:
            path = nx.shortest_path(self.graph.graph, source, target, weight=weight)
            steps = []
            
            # Convert path to route steps
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Get edge data from original graph
                edge_data = None
                for _, data in self.graph.graph[u][v].items():
                    if isinstance(data, Edge):
                        data = data.to_dict()
                    if edge_data is None or data.get(weight, float('inf')) < edge_data.get(weight, float('inf')):
                        edge_data = data
                
                if edge_data:
                    steps.append(RouteStep(
                        source=u,
                        target=v,
                        transport_mode=edge_data.get('transport_mode', 'unknown'),
                        travel_time=edge_data.get('travel_time', 0),
                        cost=edge_data.get('cost', 0),
                        distance=edge_data.get('distance', 0),
                        wheelchair_accessible=edge_data.get('wheelchair_accessible', True),
                        scenic_value=edge_data.get('scenic_value', 0)
                    ))
            
            return RouteResult(steps) if steps else None
        except nx.NetworkXNoPath:
            return None

    def _find_single_mode_route(self, source: str, target: str, weights: Dict[str, float], mode: str) -> Optional[RouteResult]:
        """Find a route using only a single transport mode"""
        # Create a subgraph with only the desired mode
        subgraph = nx.MultiDiGraph()
        
        # Add all nodes
        for node_id, node in self.graph.nodes.items():
            subgraph.add_node(node_id, **node.to_dict())
        
        # Add edges with calculated weights
        for u, v, k, data in self.graph.graph.edges(data=True, keys=True):
            edge_data = data.copy()
            if isinstance(edge_data, Edge):
                edge_data = edge_data.to_dict()
            
            # Calculate edge weight based on time, cost, and scenic value
            weight = (
                weights.get('time', 0) * edge_data.get('travel_time', 0) +
                weights.get('cost', 0) * edge_data.get('cost', 0) +
                weights.get('scenic', 0) * (1 - edge_data.get('scenic_value', 0))
            )
            
            # Add small base weight to prevent negative or zero weights
            edge_data['weight'] = max(weight, 0.1)
            
            # For high time/cost weights, adjust weights to strongly prefer the desired mode
            if weights.get('time', 0) > 0.8 and edge_data.get('transport_mode') != 'train':
                edge_data['weight'] *= 3.0  # Increased penalty
            elif weights.get('cost', 0) > 0.8 and edge_data.get('transport_mode') != 'bus':
                edge_data['weight'] *= 3.0  # Increased penalty
            
            subgraph.add_edge(u, v, key=k, **edge_data)
        
        try:
            # Find shortest path in the subgraph
            path = nx.dijkstra_path(subgraph, source, target, weight='weight')
            steps = []
            
            # Convert path to route steps
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Get edge data from original graph to ensure we have all attributes
                edge_data = None
                for _, data in self.graph.graph[u][v].items():
                    if isinstance(data, Edge):
                        data = data.to_dict()
                    if data.get('transport_mode') == mode:
                        edge_data = data
                        break
                
                if edge_data:
                    steps.append(RouteStep(
                        source=u,
                        target=v,
                        transport_mode=mode,
                        travel_time=edge_data.get('travel_time', 0),
                        cost=edge_data.get('cost', 0),
                        distance=edge_data.get('distance', 0),
                        wheelchair_accessible=edge_data.get('wheelchair_accessible', True),
                        scenic_value=edge_data.get('scenic_value', 0)
                    ))
            
            return RouteResult(steps) if steps else None
        except nx.NetworkXNoPath:
            return None

    def _find_multi_mode_route(self, source: str, target: str, weights: Dict[str, float]) -> Optional[RouteResult]:
        """Find a route allowing multiple transport modes"""
        # Convert graph for pathfinding
        nx_graph = self._prepare_graph_for_pathfinding(weights)

        # Find k shortest paths using modified Yen's algorithm
        k = 10  # Increased to find more diverse paths
        paths = self._k_shortest_paths(nx_graph, source, target, k)

        if not paths:
            return None

        # Score each path based on multiple criteria
        best_path = None
        best_score = float('-inf')
        
        for path in paths:
            # Convert path to route steps
            steps = []
            prev_mode = None
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Get edge data from original graph
                edges = []
                for _, data in self.graph.graph[u][v].items():
                    if isinstance(data, Edge):
                        data = data.to_dict()
                    edges.append(data)
                
                # Try to find an edge with a different mode than the previous one
                best_edge = None
                best_edge_score = float('-inf')
                
                for edge_data in edges:
                    mode = edge_data.get('transport_mode')
                    if mode == prev_mode:
                        continue
                        
                    # Score this edge
                    time_score = -edge_data.get('travel_time', 0)
                    cost_score = -edge_data.get('cost', 0)
                    mode_switch_score = 50 if prev_mode and mode != prev_mode else 0
                    
                    edge_score = (
                        weights.get('time', 0) * time_score +
                        weights.get('cost', 0) * cost_score +
                        0.2 * mode_switch_score  # Mode switching bonus
                    )
                    
                    if edge_score > best_edge_score:
                        best_edge_score = edge_score
                        best_edge = edge_data
                
                # If no different mode found, use any edge
                if not best_edge and edges:
                    best_edge = edges[0]
                
                if best_edge:
                    steps.append(RouteStep(
                        source=u,
                        target=v,
                        transport_mode=best_edge.get('transport_mode', 'unknown'),
                        travel_time=best_edge.get('travel_time', 0),
                        cost=best_edge.get('cost', 0),
                        distance=best_edge.get('distance', 0),
                        wheelchair_accessible=best_edge.get('wheelchair_accessible', True),
                        scenic_value=best_edge.get('scenic_value', 0)
                    ))
                    
                    prev_mode = best_edge.get('transport_mode')
            
            if not steps:
                continue
                
            # Calculate path metrics
            total_time = sum(step.travel_time for step in steps)
            total_cost = sum(step.cost for step in steps)
            total_distance = sum(step.distance for step in steps)
            modes = set(step.transport_mode for step in steps)
            mode_switches = sum(1 for i in range(len(steps)-1) 
                              if steps[i].transport_mode != steps[i+1].transport_mode)
            
            # Calculate score components
            time_score = -total_time  # Negative because lower is better
            cost_score = -total_cost  # Negative because lower is better
            diversity_score = (len(modes) * 100) + (mode_switches * 50)  # Increased diversity rewards
            efficiency_score = -total_distance  # Negative because lower is better
            
            # Combine scores based on weights
            score = (
                weights.get('time', 0) * time_score +
                weights.get('cost', 0) * cost_score +
                0.3 * diversity_score +  # Increased diversity weight
                0.1 * efficiency_score
            )
            
            # For balanced weights, increase importance of diversity even more
            if abs(weights.get('time', 0) - weights.get('cost', 0)) < 0.1:
                score += diversity_score * 0.5
            
            if score > best_score:
                best_score = score
                best_path = steps

        return RouteResult(best_path) if best_path else None

    def _prepare_graph_for_pathfinding(self, weights: Dict[str, float]) -> nx.MultiDiGraph:
        """Prepare graph for pathfinding by calculating edge weights"""
        graph = nx.MultiDiGraph()
        
        # Add all nodes
        for node_id, node in self.graph.nodes.items():
            graph.add_node(node_id, **node.to_dict())
        
        # Add edges with calculated weights
        for u, v, k, data in self.graph.graph.edges(data=True, keys=True):
            edge_data = data.copy()
            if isinstance(edge_data, Edge):
                edge_data = edge_data.to_dict()
            
            # Calculate edge weight based on time, cost, and scenic value
            weight = (
                weights.get('time', 0) * edge_data.get('travel_time', 0) +
                weights.get('cost', 0) * edge_data.get('cost', 0) +
                weights.get('scenic', 0) * (1 - edge_data.get('scenic_value', 0))
            )
            
            # Add small base weight to prevent negative or zero weights
            edge_data['weight'] = max(weight, 0.1)
            
            # For high time/cost weights, adjust weights to strongly prefer the desired mode
            if weights.get('time', 0) > 0.8 and edge_data.get('transport_mode') != 'train':
                edge_data['weight'] *= 3.0  # Increased penalty
            elif weights.get('cost', 0) > 0.8 and edge_data.get('transport_mode') != 'bus':
                edge_data['weight'] *= 3.0  # Increased penalty
            
            graph.add_edge(u, v, key=k, **edge_data)
        
        return graph

    def _k_shortest_paths(self, graph: nx.MultiDiGraph, source: str, target: str, k: int) -> List[List[str]]:
        """Find k shortest paths using a modified version of Yen's algorithm"""
        try:
            # Find first shortest path
            shortest_path = nx.dijkstra_path(graph, source, target, weight='weight')
            paths = [shortest_path]
            
            # Find k-1 more paths
            for _ in range(k - 1):
                # Create a copy of the graph
                temp_graph = graph.copy()
                
                # Try to find an alternative path
                for i in range(len(paths[-1]) - 1):
                    # Remove edges along the path
                    u, v = paths[-1][i], paths[-1][i + 1]
                    if temp_graph.has_edge(u, v):
                        temp_graph.remove_edge(u, v)
                
                try:
                    # Find new path in modified graph
                    new_path = nx.dijkstra_path(temp_graph, source, target, weight='weight')
                    if new_path not in paths:
                        paths.append(new_path)
                except nx.NetworkXNoPath:
                    continue
                    
            return paths
        except nx.NetworkXNoPath:
            return []

    def calculate_heuristic(self, source: str, target: str, weights: Dict[str, float]) -> float:
        """Calculate admissible heuristic between two nodes"""
        if source == target:
            return 0.0
            
        source_node = self.graph.nodes[source]
        target_node = self.graph.nodes[target]
        
        # Calculate straight-line distance in km
        distance = self.calculate_distance(
            source_node.latitude,
            source_node.longitude,
            target_node.latitude,
            target_node.longitude
        )
        
        # Estimate minimum time and cost based on distance
        min_speed = 60.0  # km/h for train
        min_cost_per_km = 0.25  # € per km for bus
        
        min_time = (distance / min_speed) * 60  # Convert to minutes
        min_cost = distance * min_cost_per_km
        
        # Calculate weighted heuristic
        h = (
            weights.get('time', 0) * min_time +
            weights.get('cost', 0) * min_cost +
            weights.get('scenic', 0) * distance * 0.1  # Small scenic penalty
        )
        
        return max(h, 0.0)  # Ensure non-negative

    def _path_to_steps(self, path: List[str]) -> List[RouteStep]:
        """Convert a path to a list of route steps"""
        steps = []
        prev_mode = None
        mode_switches = 0
        
        for i in range(len(path) - 1):
            edge_data = self.graph.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                data = edge_data[0]
                current_mode = data.get('transport_mode')
                
                # Count mode switches
                if prev_mode and current_mode != prev_mode:
                    mode_switches += 1
                prev_mode = current_mode
                
                steps.append(RouteStep(
                    source=path[i],
                    target=path[i + 1],
                    transport_mode=current_mode or 'unknown',
                    travel_time=data.get('travel_time', 0),
                    cost=data.get('cost', 0),
                    distance=data.get('distance', 0),
                    wheelchair_accessible=data.get('wheelchair_accessible', True),
                    scenic_value=data.get('scenic_value', 0)
                ))
        
        # If we have no mode switches, try to find alternative edges with different modes
        if mode_switches == 0 and len(steps) > 1:
            # Look for alternative modes between each pair of nodes
            new_steps = []
            prev_mode = None
            
            for step in steps:
                alternatives = self.graph.graph.get_edge_data(step.source, step.target)
                best_alt = None
                best_score = float('inf')
                
                # Find the best alternative edge that uses a different mode
                for k, data in alternatives.items():
                    mode = data.get('transport_mode')
                    if mode != prev_mode:  # Prefer different mode than previous
                        score = data.get('travel_time', 0) + data.get('cost', 0)
                        if score < best_score:
                            best_score = score
                            best_alt = (mode, data)
                
                if best_alt:
                    mode, data = best_alt
                    new_steps.append(RouteStep(
                        source=step.source,
                        target=step.target,
                        transport_mode=mode,
                        travel_time=data.get('travel_time', 0),
                        cost=data.get('cost', 0),
                        distance=data.get('distance', 0),
                        wheelchair_accessible=data.get('wheelchair_accessible', True),
                        scenic_value=data.get('scenic_value', 0)
                    ))
                    prev_mode = mode
                else:
                    new_steps.append(step)
                    prev_mode = step.transport_mode
            
            # Use the alternative path if it has mode switches
            if any(i > 0 and new_steps[i].transport_mode != new_steps[i-1].transport_mode 
                   for i in range(len(new_steps))):
                return new_steps
        
        return steps

    def _find_path_with_weights(self, source: str, target: str, weights: Dict[str, float]) -> List[str]:
        """Helper method to find path with given weights"""
        # Define heuristic function for A* (using geographic distance)
        def heuristic(n1, n2):
            n1_data = self.graph.nodes[n1]
            n2_data = self.graph.nodes[n2]
            return self.calculate_distance(
                n1_data.latitude, n1_data.longitude,
                n2_data.latitude, n2_data.longitude
            )

        try:
            # Convert TransportGraph to NetworkX graph for A* algorithm
            nx_graph = nx.MultiDiGraph()
            
            # Add nodes with their attributes
            for node_id, node in self.graph.nodes.items():
                nx_graph.add_node(node_id, **node.to_dict())
            
            # Add edges with their attributes and track previous modes
            prev_modes = {}  # Track previous mode for each node
            for u, v, k, data in self.graph.graph.edges(data=True, keys=True):
                edge_data = data.copy()
                if isinstance(edge_data, Edge):
                    edge_data = edge_data.to_dict()
                
                # Add previous mode information
                if u in prev_modes:
                    edge_data['_prev_mode'] = prev_modes[u]
                prev_modes[v] = edge_data.get('transport_mode')
                
                nx_graph.add_edge(u, v, key=k, **edge_data)
            
            # Use A* to find shortest path
            path = nx.astar_path(
                nx_graph,
                source,
                target,
                heuristic=self.calculate_heuristic,
                weight=lambda u, v, d: self._calculate_edge_weight(u, v, d, weights)
            )
            return path
        except nx.NetworkXNoPath:
            return []

    def _calculate_edge_weight(self, node1: str, node2: str, edge_data: Dict, weights: Dict[str, float]) -> float:
        """Calculate weighted edge cost based on multiple criteria"""
        # Get all edges between these nodes
        all_edges = self.graph.graph.get_edge_data(node1, node2)
        if not all_edges:
            return float('inf')

        # Calculate minimum weight across all edges
        min_weight = float('inf')
        for k, data in all_edges.items():
            try:
                weight = self._calculate_single_edge_weight(data, weights)
                min_weight = min(min_weight, weight)
            except Exception as e:
                print(f"Error calculating weight for edge {node1}->{node2}: {str(e)}")
                continue

        return min_weight

    def _calculate_single_edge_weight(self, edge_data: Dict, weights: Dict[str, float]) -> float:
        """Calculate weight for a single edge considering all criteria."""
        # Basic weight calculation
        base_weight = (
            weights.get('time', 0) * edge_data.get('travel_time', 0) +
            weights.get('cost', 0) * edge_data.get('cost', 0) +
            weights.get('scenic', 0) * (1 - edge_data.get('scenic_value', 0))  # Invert scenic value as higher is better
        )

        # Apply mode diversity penalty if set during path finding
        if '_penalty' in edge_data:
            base_weight *= edge_data['_penalty']

        # Add penalties for weather and traffic if conditions are poor
        if edge_data.get('weather_sensitivity', 0) > 0.5 and self.weather_conditions.get('severity', 0) > 0.5:
            base_weight *= 1.5
        if edge_data.get('traffic_sensitivity', 0) > 0.5 and self.traffic_conditions.get('severity', 0) > 0.5:
            base_weight *= 1.5

        # Add penalty for non-wheelchair accessible routes if accessibility is weighted
        if weights.get('accessibility', 0) > 0 and not edge_data.get('wheelchair_accessible', True):
            base_weight *= (1 + weights['accessibility'])

        # For balanced weights, add mode-specific adjustments
        if abs(weights.get('time', 0) - weights.get('cost', 0)) < 0.1:
            mode = edge_data.get('transport_mode')
            if mode == 'train':
                # Trains are better for longer distances and worse for shorter ones
                if edge_data.get('distance', 0) > 50:
                    base_weight *= 0.7
                else:
                    base_weight *= 1.3
            elif mode == 'bus':
                # Buses are better for shorter distances and worse for longer ones
                if edge_data.get('distance', 0) <= 50:
                    base_weight *= 0.7
                else:
                    base_weight *= 1.3

            # Add diversity bonus/penalty based on previous mode
            prev_mode = edge_data.get('_prev_mode')
            if prev_mode and prev_mode == mode:
                # Significant penalty for using same mode consecutively
                base_weight *= 2.0
            elif prev_mode and prev_mode != mode:
                # Bonus for switching modes
                base_weight *= 0.5

        return max(base_weight, 0.1)  # Ensure weight is always positive

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on Earth"""
        R = 6371  # Earth's radius in kilometers

        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c  # Distance in kilometers
