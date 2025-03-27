import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import math

@dataclass
class Node:
    """Class representing a node in the transport graph"""
    id: str
    name: str
    latitude: float
    longitude: float
    node_type: str = "stop"  # 'stop' or 'poi'
    wheelchair_accessible: bool = False
    
    def __post_init__(self):
        """Validate node attributes after initialization"""
        if not self.id:
            raise ValueError("Node ID is required")
        if not self.name:
            raise ValueError("Node name is required")
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")
        if self.node_type not in ['stop', 'poi']:
            raise ValueError(f"Invalid node type: {self.node_type}")
        if not isinstance(self.wheelchair_accessible, bool):
            self.wheelchair_accessible = bool(self.wheelchair_accessible)

    def to_dict(self):
        """Convert node to dictionary for graph storage"""
        return {
            'id': self.id,
            'name': self.name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'node_type': self.node_type,
            'wheelchair_accessible': self.wheelchair_accessible
        }

class Edge:
    """Edge in the transport graph"""
    def __init__(self, source: str, target: str, transport_mode: str, travel_time: float,
                 cost: float, distance: float, wheelchair_accessible: bool = False,
                 scenic_value: float = 0.0):
        self.source = source
        self.target = target
        self.transport_mode = transport_mode
        self.travel_time = float(travel_time)
        self.cost = float(cost)
        self.distance = float(distance)
        self.wheelchair_accessible = bool(wheelchair_accessible)
        self.scenic_value = float(scenic_value)
        self.__post_init__()

    def __post_init__(self):
        """Validate edge attributes"""
        if not self.source:
            raise ValueError("Edge source is required")
        if not self.target:
            raise ValueError("Edge target is required")
        if not self.transport_mode:
            raise ValueError("Transport mode is required")
        if self.transport_mode not in ['train', 'bus', 'walk']:
            raise ValueError(f"Invalid transport mode: {self.transport_mode}")

        # Validate numeric attributes
        for attr_name, attr_value in [
            ('travel_time', self.travel_time),
            ('cost', self.cost),
            ('distance', self.distance),
            ('scenic_value', self.scenic_value)
        ]:
            if not isinstance(attr_value, (int, float)):
                raise ValueError(f"{attr_name} must be a number, got {type(attr_value).__name__}")
            if attr_value < 0:
                raise ValueError(f"{attr_name} must be non-negative")
                
        # Validate scenic_value is between 0 and 1
        if not 0 <= self.scenic_value <= 1:
            raise ValueError(f"scenic_value must be between 0 and 1, got {self.scenic_value}")

        # Validate wheelchair_accessible is boolean
        if not isinstance(self.wheelchair_accessible, bool):
            raise ValueError("wheelchair_accessible must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary format"""
        return {
            'source': self.source,
            'target': self.target,
            'transport_mode': self.transport_mode,
            'travel_time': self.travel_time,
            'cost': self.cost,
            'distance': self.distance,
            'wheelchair_accessible': self.wheelchair_accessible,
            'scenic_value': self.scenic_value
        }

class TransportGraph:
    def __init__(self):
        """Initialize an empty transport graph"""
        self.graph = nx.MultiDiGraph()  # Use MultiDiGraph to allow multiple edges between nodes
        self.nodes = {}  # Store Node objects for easy access
        
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        print(f"\nAdding node to graph: {node.id}")
        print(f"Node details: {node.to_dict()}")
        
        try:
            # Add node to NetworkX graph
            self.graph.add_node(
                node.id,
                name=node.name,
                latitude=node.latitude,
                longitude=node.longitude,
                node_type=node.node_type,
                wheelchair_accessible=node.wheelchair_accessible
            )
            # Store Node object in our dictionary
            self.nodes[node.id] = node
            print(f"Node {node.id} added successfully")
            
        except Exception as e:
            print(f"Error adding node {node.id}: {str(e)}")
            raise
    
    def add_edge(
        self,
        start_node: str,
        end_node: str,
        transport_mode: str,
        travel_time: float,
        cost: float,
        distance: float,
        wheelchair_accessible: bool = True,
        scenic_value: float = 0.0,
        bidirectional: bool = True
    ) -> bool:
        """Add an edge to the graph"""
        try:
            # Check if nodes exist
            if start_node not in self.nodes or end_node not in self.nodes:
                print(f"Error: One or both nodes not found: {start_node}, {end_node}")
                return False
            
            # Check if edge with this transport mode already exists
            existing_edges = self.graph.get_edge_data(start_node, end_node)
            if existing_edges:
                for edge in existing_edges.values():
                    if edge.get('transport_mode') == transport_mode:
                        print(f"Edge with transport mode {transport_mode} already exists between {start_node} and {end_node}")
                        return False
            
            # Create edge data
            edge_data = {
                'start_node': start_node,
                'end_node': end_node,
                'transport_mode': transport_mode,
                'travel_time': float(travel_time),
                'cost': float(cost),
                'distance': float(distance),
                'wheelchair_accessible': bool(wheelchair_accessible),
                'scenic_value': float(scenic_value)
            }
            
            # Add forward edge
            self.graph.add_edge(start_node, end_node, **edge_data)
            print(f"Forward edge added successfully")
            
            # Add reverse edge if bidirectional
            if bidirectional:
                self.graph.add_edge(end_node, start_node, **edge_data)
                print(f"Reverse edge added successfully")
                
            return True
            
        except Exception as e:
            print(f"Error adding edge: {str(e)}")
            return False
    
    def get_node_info(self, node_id: str) -> Optional[Node]:
        """Get node information by ID"""
        try:
            return self.nodes.get(node_id)
        except Exception as e:
            print(f"Error getting node info for {node_id}: {str(e)}")
            return None
    
    def get_edge_info(self, source: str, target: str) -> List[Dict]:
        """Get all edges between two nodes"""
        edge_data = self.graph.get_edge_data(source, target)
        if not edge_data:
            return []
        
        edges = []
        for key, data in edge_data.items():
            edges.append({
                'source': source,
                'target': target,
                'transport_mode': data['transport_mode'],
                'travel_time': data['travel_time'],
                'cost': data['cost'],
                'distance': data['distance'],
                'wheelchair_accessible': data.get('wheelchair_accessible', False),
                'scenic_value': data.get('scenic_value', 0.0)
            })
        return edges
    
    def load_data(self) -> None:
        """Load all required datasets and build the graph"""
        self._load_pois()
        self._load_stops()
        self._load_road_network()
        self._load_transport_routes()
        self._load_transfers()
        self._add_walking_connections()
    
    def _load_pois(self) -> None:
        """Load tourist POIs and add them as nodes"""
        pois_df = pd.read_csv(self.data_dir / "pois_with_scenic_scores.csv")
        
        for _, row in pois_df.iterrows():
            node = Node(
                id=f"poi_{row['name']}_{row['latitude']}_{row['longitude']}",
                name=row['name'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                node_type='poi',
                scenic_score=row['scenic_score'],
                wheelchair_accessible=row['wheelchair'] == 'yes'
            )
            self.nodes[node.id] = node
            self.graph.add_node(
                node.id,
                node_type='poi',
                name=node.name,
                pos=(node.longitude, node.latitude),
                scenic_score=node.scenic_score,
                wheelchair_accessible=node.wheelchair_accessible
            )

    def _load_stops(self) -> None:
        """Load public transport stops and add them as nodes"""
        stops_df = pd.read_csv(self.data_dir / "stops_cleaned_filled.csv")
        
        for _, row in stops_df.iterrows():
            node = Node(
                id=f"stop_{row['stop_id']}",
                name=row['stop_name'],
                latitude=row['stop_lat'],
                longitude=row['stop_lon'],
                node_type='stop',
                wheelchair_accessible=row.get('wheelchair_boarding', 0) == 1
            )
            self.nodes[node.id] = node
            self.graph.add_node(
                node.id,
                node_type='stop',
                name=node.name,
                pos=(node.longitude, node.latitude),
                wheelchair_accessible=node.wheelchair_accessible
            )

    def _load_road_network(self) -> None:
        """Load road network and add edges between nodes"""
        roads_df = pd.read_csv(self.data_dir / "filtered_roads_cleaned.csv")
        
        for _, row in roads_df.iterrows():
            source = f"stop_{row['from_stop_id']}"
            target = f"stop_{row['to_stop_id']}"
            
            if source in self.nodes and target in self.nodes:
                edge = Edge(
                    source=source,
                    target=target,
                    transport_mode='road',
                    travel_time=row['travel_time'],
                    cost=row['cost'],
                    distance=row['distance'],
                    wheelchair_accessible=True,  # Assuming roads are wheelchair accessible by default
                    scenic_value=0.0
                )
                self._add_edge(edge)

    def _load_transport_routes(self) -> None:
        """Load public transport routes and add edges"""
        routes_df = pd.read_csv(self.data_dir / "netherlands-transport-cleaned.csv")
        times_df = pd.read_csv(self.data_dir / "stop_times_cleaned.csv")
        
        # Merge route and time information
        route_times = pd.merge(routes_df, times_df, on='trip_id')
        
        for _, row in route_times.iterrows():
            source = f"stop_{row['stop_id']}"
            next_stop = route_times[
                (route_times['trip_id'] == row['trip_id']) & 
                (route_times['stop_sequence'] == row['stop_sequence'] + 1)
            ]
            
            if not next_stop.empty and source in self.nodes:
                target = f"stop_{next_stop.iloc[0]['stop_id']}"
                if target in self.nodes:
                    edge = Edge(
                        source=source,
                        target=target,
                        transport_mode=row['route_type'],
                        travel_time=next_stop.iloc[0]['arrival_time'] - row['departure_time'],
                        cost=row.get('fare', 0.0),
                        distance=0.0,  # Calculate using coordinates if needed
                        wheelchair_accessible=row.get('wheelchair_accessible', 0) == 1,
                        scenic_value=0.0
                    )
                    self._add_edge(edge)

    def _load_transfers(self) -> None:
        """Load transfer connections between stops"""
        transfers_df = pd.read_csv(self.data_dir / "cleaned_transfers.csv")
        
        for _, row in transfers_df.iterrows():
            source = f"stop_{row['from_stop_id']}"
            target = f"stop_{row['to_stop_id']}"
            
            if source in self.nodes and target in self.nodes:
                edge = Edge(
                    source=source,
                    target=target,
                    transport_mode='transfer',
                    travel_time=row['min_transfer_time'],
                    cost=0.0,  # Transfers typically don't have additional cost
                    distance=0.0,  # Calculate using coordinates if needed
                    wheelchair_accessible=True,  # Update based on actual accessibility data
                    scenic_value=0.0
                )
                self._add_edge(edge)

    def _add_walking_connections(self, max_walking_distance: float = 1.0) -> None:
        """Add walking connections between nearby nodes (within max_walking_distance km)"""
        from itertools import combinations
        
        # Average walking speed in km/h
        walking_speed = 5.0
        
        for node1, node2 in combinations(self.nodes.values(), 2):
            distance = self.haversine_distance(
                node1.latitude, node1.longitude,
                node2.latitude, node2.longitude
            )
            
            if distance <= max_walking_distance:
                # Calculate walking time in minutes
                time = (distance / walking_speed) * 60
                
                edge = Edge(
                    source=node1.id,
                    target=node2.id,
                    transport_mode='walk',
                    travel_time=time,
                    cost=0.0,
                    distance=distance,
                    wheelchair_accessible=node1.wheelchair_accessible and node2.wheelchair_accessible,
                    scenic_value=0.0
                )
                self._add_edge(edge)
                
                # Add reverse edge for walking back
                reverse_edge = Edge(
                    source=node2.id,
                    target=node1.id,
                    transport_mode='walk',
                    travel_time=time,
                    cost=0.0,
                    distance=distance,
                    wheelchair_accessible=node1.wheelchair_accessible and node2.wheelchair_accessible,
                    scenic_value=0.0
                )
                self._add_edge(reverse_edge)

    def _add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph with all its attributes"""
        # Check if edge with same transport mode already exists
        existing_edges = self.graph.get_edge_data(edge.source, edge.target)
        if existing_edges:
            for key, data in existing_edges.items():
                if data['transport_mode'] == edge.transport_mode:
                    print(f"Edge with transport mode {edge.transport_mode} already exists between {edge.source} and {edge.target}")
                    return

        # Add edge with its attributes
        self.graph.add_edge(
            edge.source,
            edge.target,
            transport_mode=edge.transport_mode,
            travel_time=edge.travel_time,
            cost=edge.cost,
            distance=edge.distance,
            wheelchair_accessible=edge.wheelchair_accessible,
            scenic_value=edge.scenic_value
        )
        print(f"Forward edge added successfully")

    def get_graph_stats(self):
        """Get statistics about the graph"""
        stats = {
            'num_nodes': len(self.nodes),
            'num_edges': self.graph.number_of_edges(),
            'num_stops': sum(1 for node in self.nodes.values() if node.node_type == 'stop'),
            'num_pois': sum(1 for node in self.nodes.values() if node.node_type == 'poi'),
            'num_wheelchair_accessible_nodes': sum(1 for node in self.nodes.values() if node.wheelchair_accessible),
            'num_wheelchair_accessible_edges': sum(1 for _, _, data in self.graph.edges(data=True) if data.get('wheelchair_accessible', False)),
            'transport_modes': set(data['transport_mode'] for _, _, data in self.graph.edges(data=True)),
            'node_types': set(node.node_type for node in self.nodes.values()),
            'total_distance': sum(data['distance'] for _, _, data in self.graph.edges(data=True)),
            'avg_travel_time': sum(data['travel_time'] for _, _, data in self.graph.edges(data=True)) / self.graph.number_of_edges() if self.graph.number_of_edges() > 0 else 0,
            'avg_distance': sum(data['distance'] for _, _, data in self.graph.edges(data=True)) / self.graph.number_of_edges() if self.graph.number_of_edges() > 0 else 0
        }
        return stats

    def find_shortest_path(self, source: str, target: str, weight: str = 'distance') -> List[str]:
        """Find shortest path between two nodes using specified weight"""
        if source not in self.nodes:
            raise nx.NetworkXNoPath(f"Source node not found: {source}")
        if target not in self.nodes:
            raise nx.NetworkXNoPath(f"Target node not found: {target}")
            
        try:
            path = nx.shortest_path(self.graph, source, target, weight=weight)
            return path
        except nx.NetworkXNoPath:
            raise
        except Exception as e:
            raise ValueError(f"Error finding path: {str(e)}")

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on Earth"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        r = 6371
        
        return c * r
