from typing import List, Dict, Tuple, Optional
import networkx as nx
from datetime import datetime, timedelta
from .graph_model import TransportGraph, Node, Edge

class RouteStep:
    """A step in a route between two points"""
    def __init__(
        self,
        start: str,
        end: str,
        transport_mode: str,
        travel_time: float,
        cost: float,
        distance: float,
        scenic_value: float = 0.0,
        wheelchair_accessible: bool = False
    ):
        """Initialize a route step"""
        self.start = start
        self.end = end
        self.transport_mode = transport_mode
        self.travel_time = float(travel_time)
        self.cost = float(cost)
        self.distance = float(distance)
        self.scenic_value = float(scenic_value)
        self.wheelchair_accessible = bool(wheelchair_accessible)
    
    def to_dict(self) -> Dict:
        """Convert route step to dictionary"""
        return {
            'start': self.start,
            'end': self.end,
            'transport_mode': self.transport_mode,
            'travel_time': self.travel_time,
            'cost': self.cost,
            'distance': self.distance,
            'scenic_value': self.scenic_value,
            'wheelchair_accessible': self.wheelchair_accessible
        }

class Route:
    """A route between two points consisting of multiple steps"""
    def __init__(self, steps: List[RouteStep]):
        """Initialize a route with a list of steps"""
        self.steps = steps
        
        # Calculate route summary
        self.summary = {
            'total_time': sum(step.travel_time for step in steps),
            'total_cost': sum(step.cost for step in steps),
            'total_distance': sum(step.distance for step in steps),
            'transport_modes': list(set(step.transport_mode for step in steps)),
            'num_transfers': len(steps) - 1
        }
        
        # Add direct access to metrics
        self.total_time = self.summary['total_time']
        self.total_cost = self.summary['total_cost']
        self.total_distance = self.summary['total_distance']
        self.transport_modes = self.summary['transport_modes']
        self.num_transfers = self.summary['num_transfers']
    
    def to_dict(self) -> Dict:
        """Convert route to dictionary"""
        return {
            'steps': [step.to_dict() for step in self.steps],
            'summary': self.summary
        }
    
    def __str__(self) -> str:
        """String representation of route"""
        return f"Route with {len(self.steps)} steps: {' -> '.join(step.transport_mode for step in self.steps)}"

def find_fastest_route(
    graph: TransportGraph,
    source: str,
    target: str,
    departure_time: Optional[datetime] = None,
    accessibility_required: bool = False
) -> Optional[Route]:
    """
    Find the fastest route between two locations using Dijkstra's algorithm.
    
    Args:
        graph: TransportGraph instance
        source: Source node ID
        target: Target node ID
        departure_time: Optional departure time for time-dependent routing
        accessibility_required: Whether the route must be wheelchair accessible
    
    Returns:
        Route object containing the optimal path and its details
    """
    try:
        # Create a weight function that considers travel time and accessibility
        def weight_function(u: str, v: str, edge_data: Dict) -> float:
            if accessibility_required and not edge_data.get('wheelchair_accessible', False):
                return float('inf')
            return edge_data.get('travel_time', float('inf'))
        
        # Find the shortest path using Dijkstra's algorithm
        path = nx.dijkstra_path(
            graph.graph,
            source,
            target,
            weight=weight_function
        )
        
        # Convert path to route steps
        steps = []
        for i in range(len(path) - 1):
            current_node = graph.get_node_info(path[i])
            next_node = graph.get_node_info(path[i + 1])
            
            # Get edge data (there might be multiple edges between nodes)
            edge_data = None
            min_time = float('inf')
            
            for _, _, data in graph.graph.edges(data=True):
                if data['start_node'] == path[i] and data['end_node'] == path[i + 1]:
                    if data['travel_time'] < min_time:
                        edge_data = data
                        min_time = data['travel_time']
            
            if edge_data and current_node and next_node:
                step = RouteStep(
                    start=path[i],
                    end=path[i + 1],
                    transport_mode=edge_data['transport_mode'],
                    travel_time=edge_data['travel_time'],
                    cost=edge_data['cost'],
                    distance=edge_data['distance'],
                    scenic_value=edge_data.get('scenic_value', 0.0),
                    wheelchair_accessible=edge_data.get('wheelchair_accessible', False)
                )
                steps.append(step)
        
        return Route(steps)
    
    except nx.NetworkXNoPath:
        return None
    except Exception as e:
        print(f"Error finding route: {str(e)}")
        return None

def get_route_description(route: Route) -> str:
    """
    Generate a human-readable description of the route.
    """
    if not route or not route.steps:
        return "No route found."
    
    description = []
    current_time = 0
    
    for i, step in enumerate(route.steps, 1):
        time_str = f"{int(current_time)}:{int((current_time % 1) * 60):02d}"
        next_time = current_time + step.travel_time
        next_time_str = f"{int(next_time)}:{int((next_time % 1) * 60):02d}"
        
        mode_desc = {
            'walk': 'Walk',
            'bus': 'Take bus',
            'train': 'Take train',
            'tram': 'Take tram',
            'metro': 'Take metro',
            'transfer': 'Transfer',
            'road': 'Drive'
        }.get(step.transport_mode, step.transport_mode)
        
        description.append(
            f"{i}. {time_str} - {next_time_str}: {mode_desc} from {step.start} "
            f"to {step.end} ({step.travel_time:.1f} min)"
        )
        
        current_time = next_time
    
    summary = (
        f"\nTotal Journey:"
        f"\n- Time: {route.summary['total_time']:.1f} minutes"
        f"\n- Cost: €{route.summary['total_cost']:.2f}"
        f"\n- Distance: {route.summary['total_distance']:.1f} km"
        f"\n- Transfers: {route.summary['num_transfers']}"
    )
    
    return "\n".join(description) + summary

def calculate_heuristic(
    graph: TransportGraph,
    current: str,
    target: str,
    time_weight: float = 0.7,
    cost_weight: float = 0.3
) -> float:
    """
    Calculate A* heuristic based on straight-line distance and estimated cost.
    
    Args:
        graph: TransportGraph instance
        current: Current node ID
        target: Target node ID
        time_weight: Weight for time component (default: 0.7)
        cost_weight: Weight for cost component (default: 0.3)
    
    Returns:
        Estimated cost to reach target
    """
    current_node = graph.get_node_info(current)
    target_node = graph.get_node_info(target)
    
    if not current_node or not target_node:
        return float('inf')
    
    # Calculate straight-line distance using Haversine formula
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1 = map(radians, [current_node.latitude, current_node.longitude])
    lat2, lon2 = map(radians, [target_node.latitude, target_node.longitude])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    # Estimate time based on average speed (assume 60 km/h)
    estimated_time = (distance / 60) * 60  # Convert to minutes
    
    # Estimate cost based on distance (assume €0.5 per km)
    estimated_cost = distance * 0.5
    
    # Combine time and cost estimates using weights
    return (time_weight * estimated_time) + (cost_weight * estimated_cost)

def find_optimal_route(
    graph: TransportGraph,
    source: str,
    target: str,
    time_weight: float = 0.7,
    cost_weight: float = 0.3,
    departure_time: Optional[datetime] = None,
    accessibility_required: bool = False
) -> Optional[Route]:
    """
    Find the optimal route using A* algorithm with customizable weights for time and cost.
    """
    print(f"\nFinding optimal route from {source} to {target}")
    print(f"Weights: time={time_weight}, cost={cost_weight}")
    
    # Validate input nodes
    if source not in graph.nodes or target not in graph.nodes:
        print(f"Error: Source {source} or target {target} not in graph")
        print(f"Available nodes: {list(graph.nodes)}")
        return None
    
    try:
        # Validate weights
        if not (0 <= time_weight <= 1 and 0 <= cost_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if abs(time_weight + cost_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1")
        
        # Create weight function that combines time and cost
        def weight_function(u: str, v: str, edge_data: Dict) -> float:
            if accessibility_required and not edge_data.get('wheelchair_accessible', False):
                return float('inf')
            
            time_cost = edge_data.get('travel_time', float('inf'))
            monetary_cost = edge_data.get('cost', float('inf'))
            
            if time_cost == float('inf') or monetary_cost == float('inf'):
                print(f"Warning: Invalid edge data between {u} and {v}: {edge_data}")
                return float('inf')
            
            # Normalize costs (assuming max values)
            normalized_time = time_cost / 180.0  # Assume max 3 hours
            normalized_cost = monetary_cost / 100.0  # Assume max 100 euros
            
            return time_weight * normalized_time + cost_weight * normalized_cost
        
        # Find path using A* algorithm
        path = nx.astar_path(
            graph.graph,
            source,
            target,
            heuristic=lambda n1, n2: calculate_heuristic(
                graph, n1, target, time_weight, cost_weight
            ),
            weight=weight_function
        )
        print(f"Found path: {path}")
        
        if not path or len(path) < 2:
            print(f"No valid path found between {source} and {target}")
            return None
            
        # Convert path to route steps
        steps = []
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]
            
            # Get nodes
            from_node = graph.nodes[from_id]
            to_node = graph.nodes[to_id]
            
            # Get edge data
            edge_data = graph.graph.get_edge_data(from_id, to_id)
            if not edge_data:
                print(f"No edge data found between {from_id} and {to_id}")
                continue
                
            try:
                # Create route step with all required attributes
                step = RouteStep(
                    start=from_id,
                    end=to_id,
                    transport_mode=edge_data['transport_mode'],
                    travel_time=float(edge_data['travel_time']),
                    cost=float(edge_data['cost']),
                    distance=float(edge_data['distance']),
                    scenic_value=float(edge_data.get('scenic_value', 0.0)),
                    wheelchair_accessible=bool(edge_data.get('wheelchair_accessible', False))
                )
                steps.append(step)
            except (KeyError, ValueError) as e:
                print(f"Error creating route step: {str(e)}")
                print(f"Edge data: {edge_data}")
                continue
        
        if not steps:
            print("No valid steps could be created")
            return None
            
        return Route(steps)
        
    except Exception as e:
        print(f"Error finding route: {str(e)}")
        return None

def get_route_metrics(route: Route) -> Dict:
    """Calculate metrics for a route"""
    # Initialize metrics dictionary with default values
    metrics = {
        'total_time': 0.0,
        'total_cost': 0.0,
        'total_distance': 0.0,
        'avg_speed': 0.0,
        'cost_per_km': 0.0,
        'transport_modes': {},
        'mode_distribution': {},
        'num_transfers': 0
    }
    
    # Handle empty route
    if not route or not route.steps:
        return metrics
    
    # Calculate basic totals
    metrics['total_time'] = sum(step.travel_time for step in route.steps)
    metrics['total_cost'] = sum(step.cost for step in route.steps)
    metrics['total_distance'] = sum(step.distance for step in route.steps)
    
    # Calculate derived metrics
    if metrics['total_time'] > 0:
        metrics['avg_speed'] = (metrics['total_distance'] / (metrics['total_time'] / 60))  # km/h
    if metrics['total_distance'] > 0:
        metrics['cost_per_km'] = metrics['total_cost'] / metrics['total_distance']
    
    # Calculate transport mode distribution
    mode_times = {}
    for step in route.steps:
        mode = step.transport_mode
        if mode not in mode_times:
            mode_times[mode] = 0
            metrics['transport_modes'][mode] = 0
        mode_times[mode] += step.travel_time
        metrics['transport_modes'][mode] += 1
    
    # Calculate mode distribution percentages based on time
    if metrics['total_time'] > 0:
        metrics['mode_distribution'] = {
            mode: round((time / metrics['total_time']) * 100, 2)  # Round to 2 decimal places
            for mode, time in mode_times.items()
        }
    elif metrics['total_distance'] == 0:
        # Special case for zero-distance routes
        metrics['mode_distribution'] = {'walk': 100.0}
    
    # Calculate number of transfers
    if len(route.steps) > 1:
        prev_mode = route.steps[0].transport_mode
        for step in route.steps[1:]:
            if step.transport_mode != prev_mode:
                metrics['num_transfers'] += 1
            prev_mode = step.transport_mode
    
    return metrics
