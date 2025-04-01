"""Graph model for transport network."""
import networkx as nx
import pandas as pd
import joblib
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Node:
    """Class representing a node in the transport graph"""
    id: str
    name: str
    latitude: float
    longitude: float
    node_type: str = "stop"
    wheelchair_accessible: bool = False

    def __post_init__(self):
        """Validate node attributes after initialization"""
        if not isinstance(self.id, str):
            raise ValueError("Node ID must be a string")
        if not isinstance(self.name, str):
            raise ValueError("Node name must be a string")
        if not isinstance(self.latitude, (int, float)):
            raise ValueError("Latitude must be a number")
        if not isinstance(self.longitude, (int, float)):
            raise ValueError("Longitude must be a number")
        if not isinstance(self.node_type, str):
            raise ValueError("Node type must be a string")
            
        # Convert wheelchair_accessible to bool
        if isinstance(self.wheelchair_accessible, str):
            self.wheelchair_accessible = self.wheelchair_accessible.lower() in ('true', '1', 'yes', 't')
        else:
            self.wheelchair_accessible = bool(self.wheelchair_accessible)

    def to_dict(self) -> Dict:
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
        if self.transport_mode not in ['train', 'bus', 'walk', 'road', 'tram', 'metro']:
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
    """Graph representation of the transport network"""
    
    def __init__(self, use_cache: bool = True):
        """Initialize the transport graph with optional caching"""
        self.graph = nx.MultiDiGraph()
        self.nodes = {}
        self.use_cache = use_cache
        
        # Set up directories
        self.data_dir = Path(__file__).resolve().parent.parent.parent / "Databases for training" / "gtfs_nl"
        if not self.data_dir.exists():
            raise RuntimeError(f"Data directory not found: {self.data_dir}")
            
        # Ensure filtered data directory exists
        filtered_data_dir = self.data_dir / "filtered_data"
        if not filtered_data_dir.exists():
            raise RuntimeError(
                f"Filtered data directory not found: {filtered_data_dir}\n"
                "Please run scripts/create_filtered_dataset.py first"
            )
            
        self.cache_dir = Path(__file__).resolve().parent.parent.parent / "model_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.graph_cache_file = self.cache_dir / "transport_graph.joblib"
        
        # Try loading from cache first, build from scratch if needed
        if self.use_cache and self._load_from_cache():
            logger.info(f"Successfully loaded graph from cache: {self.graph_cache_file}")
            logger.info(f"Graph stats: {self.get_graph_stats()}")
        else:
            logger.info("Building graph from filtered data...")
            self.load_data()
            if self.use_cache:
                self._save_to_cache()
                logger.info(f"Graph stats after build: {self.get_graph_stats()}")
                
    def _load_from_cache(self) -> bool:
        """Load graph from cache file if it exists
        
        Returns:
            bool: True if graph was loaded successfully, False otherwise
        """
        if not self.graph_cache_file.exists():
            return False
            
        try:
            logger.info(f"Loading graph from cache: {self.graph_cache_file}")
            cached_data = joblib.load(self.graph_cache_file)
            
            if not isinstance(cached_data, dict) or 'graph' not in cached_data or 'nodes' not in cached_data:
                logger.warning("Invalid cache format")
                return False
                
            self.graph = cached_data['graph']
            self.nodes = cached_data['nodes']
            logger.info("Graph loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading graph from cache: {e}")
            return False
            
    def _save_to_cache(self) -> None:
        """Save current graph to cache file"""
        logger.info(f"Saving graph to cache: {self.graph_cache_file}")
        try:
            # Save with compression for better storage
            joblib.dump(
                {
                    'graph': self.graph,
                    'nodes': self.nodes
                },
                self.graph_cache_file,
                compress=3  # Use compression level 3 for better storage
            )
            logger.info("Graph saved successfully")
        except Exception as e:
            logger.error(f"Error saving graph to cache: {e}")
            raise
            
    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph"""
        return {
            'num_nodes': len(self.nodes),
            'num_edges': self.graph.number_of_edges(),
            'num_stops': len([n for n in self.nodes.values() if n.node_type == 'stop']),
            'num_pois': len([n for n in self.nodes.values() if n.node_type == 'poi'])
        }
        
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            self.graph.add_node(
                node.id,
                node_type=node.node_type,
                name=node.name,
                pos=(node.longitude, node.latitude),
                wheelchair_accessible=node.wheelchair_accessible
            )
            
    def add_edge(self, start_node: str, end_node: str, transport_mode: str,
                travel_time: float, cost: float, distance: float,
                wheelchair_accessible: bool = True, scenic_value: float = 0.0,
                bidirectional: bool = True) -> None:
        """Add an edge to the graph"""
        if start_node in self.nodes and end_node in self.nodes:
            edge = Edge(
                source=start_node,
                target=end_node,
                transport_mode=transport_mode,
                travel_time=travel_time,
                cost=cost,
                distance=distance,
                wheelchair_accessible=wheelchair_accessible,
                scenic_value=scenic_value
            )
            self._add_edge(edge)
            
            if bidirectional:
                reverse_edge = Edge(
                    source=end_node,
                    target=start_node,
                    transport_mode=transport_mode,
                    travel_time=travel_time,
                    cost=cost,
                    distance=distance,
                    wheelchair_accessible=wheelchair_accessible,
                    scenic_value=scenic_value
                )
                self._add_edge(reverse_edge)

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
        logger.info("Loading stops...")
        self._load_stops()
        
        logger.info("Loading POIs...")
        self._load_pois()
        
        logger.info("Loading road network...")
        self._load_road_network()
        
        logger.info("Loading transport routes...")
        self._load_transport_routes()
        
        logger.info("Loading transfers...")
        self._load_transfers()
        
        logger.info("Adding walking connections...")
        self._add_walking_connections()
        
        logger.info("Graph loading complete!")
        
    def _load_stops(self) -> None:
        """Load public transport stops from filtered dataset"""
        stops_file = self.data_dir / "filtered_data" / "major_stops.csv"
        if not stops_file.exists():
            raise RuntimeError(f"Major stops file not found: {stops_file}. Please run create_filtered_dataset.py first.")
            
        stops_df = pd.read_csv(stops_file)
        logger.info(f"Loading {len(stops_df)} stops from major cities")
        
        for _, row in stops_df.iterrows():
            node = Node(
                id=f"stop_{row['stop_id']}",
                name=row['stop_name'],
                latitude=row['stop_lat'],
                longitude=row['stop_lon'],
                node_type='stop',
                wheelchair_accessible=bool(row.get('wheelchair_boarding', 0))
            )
            self.add_node(node)
            
        logger.info("Finished loading stops")
            
    def _load_pois(self) -> None:
        """Load points of interest"""
        pois_file = self.data_dir / "csv_output" / "pois_with_scenic_scores.csv"
        if not pois_file.exists():
            raise RuntimeError(f"POIs file not found: {pois_file}")
            
        pois_df = pd.read_csv(pois_file)
        logger.info(f"Loaded {len(pois_df)} POIs")
        
        for idx, row in pois_df.iterrows():
            node = Node(
                id=f"poi_{idx}",  # Use index as POI ID
                name=row['name'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                node_type='poi',
                wheelchair_accessible=row.get('wheelchair', False)
            )
            self.add_node(node)
            
    def _load_road_network(self) -> None:
        """Load road network and add edges between nodes"""
        roads_file = self.data_dir / "csv_output" / "filtered_roads_cleaned.csv"
        if not roads_file.exists():
            raise RuntimeError(f"Roads file not found: {roads_file}")
            
        roads_df = pd.read_csv(roads_file)
        logger.info(f"Loaded {len(roads_df)} road segments")
        
        for _, row in roads_df.iterrows():
            # Parse geometry string to get coordinates
            # Format: 'LINESTRING (lon1 lat1, lon2 lat2)'
            try:
                coords = row['geometry'].strip('LINESTRING ()').split(',')
                start_coords = coords[0].strip().split()
                end_coords = coords[-1].strip().split()
                
                # Calculate distance using start and end points
                start_lat, start_lon = float(start_coords[1]), float(start_coords[0])
                end_lat, end_lon = float(end_coords[1]), float(end_coords[0])
                distance = self.haversine_distance(start_lat, start_lon, end_lat, end_lon)
                
                # Calculate travel time based on maxspeed
                max_speed = float(row.get('maxspeed', 50))  # Default 50 km/h if not specified
                travel_time = (distance / max_speed) * 60  # Convert to minutes
                
                # Create road nodes
                start_node = Node(
                    id=f"road_{row['name']}_{start_lat}_{start_lon}",
                    name=f"{row['name']} Start",
                    latitude=start_lat,
                    longitude=start_lon,
                    node_type='road'
                )
                end_node = Node(
                    id=f"road_{row['name']}_{end_lat}_{end_lon}",
                    name=f"{row['name']} End",
                    latitude=end_lat,
                    longitude=end_lon,
                    node_type='road'
                )
                
                self.add_node(start_node)
                self.add_node(end_node)
                
                # Add road segment as edge
                self.add_edge(
                    start_node=start_node.id,
                    end_node=end_node.id,
                    transport_mode="road",
                    travel_time=travel_time,
                    cost=distance * 0.1,  # Approximate cost based on distance
                    distance=distance,
                    wheelchair_accessible=True
                )
            except Exception as e:
                logger.warning(f"Error processing road segment: {e}")
                
    def _load_transport_routes(self) -> None:
        """Load public transport routes and add edges"""
        routes_file = self.data_dir / "csv_output" / "netherlands-transport-cleaned.csv"
        if not routes_file.exists():
            raise RuntimeError(f"Transport routes file not found: {routes_file}")
            
        routes_df = pd.read_csv(routes_file)
        logger.info(f"Loaded {len(routes_df)} transport routes")
        
        for _, row in routes_df.iterrows():
            # Determine transport mode
            if row.get('bus'):
                mode = 'bus'
            elif row.get('public_transport') == 'station':
                mode = 'train'
            else:
                mode = row.get('public_transport', 'bus')  # Default to bus if unknown
                
            # Create stop node
            stop_node = Node(
                id=f"stop_{row['name']}_{row['latitude']}_{row['longitude']}",
                name=row['name'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                node_type='stop',
                wheelchair_accessible=row.get('wheelchair', False)
            )
            self.add_node(stop_node)
            
            # Add edges to nearby stops (will be connected via _add_walking_connections)
            
    def _load_transfers(self) -> None:
        """Create transfer connections between nearby stops using KD-tree"""
        from scipy.spatial import cKDTree
        import numpy as np
        
        logger.info("Creating transfer connections between stops...")
        
        # Get all stop nodes
        stop_nodes = [node for node in self.nodes.values() if node.node_type == 'stop']
        if not stop_nodes:
            logger.warning("No stops found to create transfers")
            return
            
        # Create points array for KD-tree
        points = np.array([[node.latitude, node.longitude] for node in stop_nodes])
        tree = cKDTree(points)
        
        # Find all pairs of points within 500m
        # Convert 500m to degrees (approximate)
        max_distance_deg = 0.5 / 111  # 111km per degree
        pairs = tree.query_pairs(max_distance_deg)
        
        transfer_count = 0
        for i, j in pairs:
            node1, node2 = stop_nodes[i], stop_nodes[j]
            
            # Calculate exact distance
            distance = self.haversine_distance(
                node1.latitude, node1.longitude,
                node2.latitude, node2.longitude
            )
            
            # Connect stops within 500m
            if distance <= 0.5:  # 500m in km
                # Walking speed ~5 km/h = 12 minutes per km
                travel_time = distance * 12
                
                # Add bidirectional walking connection
                self.add_edge(
                    start_node=node1.id,
                    end_node=node2.id,
                    transport_mode='walk',
                    travel_time=travel_time,
                    cost=0,  # Walking is free
                    distance=distance,
                    wheelchair_accessible=node1.wheelchair_accessible and node2.wheelchair_accessible
                )
                transfer_count += 1
                
        logger.info(f"Created {transfer_count} transfer connections")

    def _add_walking_connections(self, max_walking_distance: float = 1.0) -> None:
        """Add walking connections between POIs and nearby transport nodes"""
        logger.info("Adding walking connections to POIs...")
        
        # Get POIs and transport nodes
        pois = [node for node in self.nodes.values() if node.node_type == 'poi']
        transport_nodes = [node for node in self.nodes.values() 
                         if node.node_type in ('stop', 'road')]
        
        # Connect POIs to nearby transport nodes
        connection_count = 0
        for poi in pois:
            for node in transport_nodes:
                # Calculate distance
                distance = self.haversine_distance(
                    poi.latitude, poi.longitude,
                    node.latitude, node.longitude
                )
                
                # Connect if within walking distance
                if distance <= max_walking_distance:
                    # Walking speed ~5 km/h = 12 minutes per km
                    travel_time = distance * 12
                    
                    # Add bidirectional walking connection
                    self.add_edge(
                        start_node=poi.id,
                        end_node=node.id,
                        transport_mode='walk',
                        travel_time=travel_time,
                        cost=0,  # Walking is free
                        distance=distance,
                        wheelchair_accessible=poi.wheelchair_accessible and node.wheelchair_accessible
                    )
                    connection_count += 1
                    
        logger.info(f"Added {connection_count} walking connections to POIs")

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
        """Calculate the great circle distance between two points on Earth
        
        Args:
            lat1: Latitude of first point in degrees
            lon1: Longitude of first point in degrees
            lat2: Latitude of second point in degrees
            lon2: Longitude of second point in degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r