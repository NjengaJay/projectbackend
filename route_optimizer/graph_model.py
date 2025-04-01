"""Graph model for transport network."""
import networkx as nx
import pandas as pd
import joblib
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
from scipy.spatial import cKDTree
import numpy as np

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
        # Store node object in self.nodes
        self.nodes[node.id] = node
        
        # Add node to networkx graph with its attributes
        self.graph.add_node(
            node.id,
            name=node.name,
            latitude=node.latitude,
            longitude=node.longitude,
            node_type=node.node_type,
            wheelchair_accessible=node.wheelchair_accessible
        )
        
        if len(self.nodes) % 1000 == 0:
            logger.debug(f"Added {len(self.nodes)} nodes to graph")
            
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
        progress_bar = tqdm(total=100, desc="Building Graph")
        
        try:
            logger.info("Loading stops...")
            self._load_stops()
            progress_bar.update(15)  # 15% complete
            
            logger.info("Loading POIs...")
            self._load_pois()
            progress_bar.update(15)  # 30% complete
            
            logger.info("Loading road network...")
            self._load_road_network()
            progress_bar.update(15)  # 45% complete
            
            logger.info("Loading transport routes...")
            self._load_transport_routes()
            progress_bar.update(15)  # 60% complete
            
            logger.info("Creating connections...")
            self._load_transfers()
            self._add_walking_connections()
            progress_bar.update(20)  # 80% complete
            
            logger.info("Propagating scenic scores...")
            self._propagate_scenic_scores()
            progress_bar.update(10)  # 90% complete
            
            # Validate the graph before marking as complete
            self._validate_graph()
            progress_bar.update(10)  # 100% complete
            
            progress_bar.close()
            logger.info("Graph loading complete!")
            
        except Exception as e:
            progress_bar.close()
            logger.error(f"Error during graph building: {str(e)}")
            # Clear the graph to prevent using a partially built graph
            self.graph = nx.MultiDiGraph()
            self.nodes = {}
            raise RuntimeError(f"Graph building failed: {str(e)}")
            
    def _validate_graph(self) -> None:
        """Validate the built graph"""
        logger.info("Starting graph validation...")
        logger.info(f"Number of nodes in self.nodes: {len(self.nodes)}")
        logger.info(f"Number of nodes in graph: {self.graph.number_of_nodes()}")
        logger.info(f"Number of edges in graph: {self.graph.number_of_edges()}")
        
        # Check that we have nodes
        if len(self.graph.nodes) == 0:
            raise ValueError("Graph has no nodes")
            
        # Check that we have edges
        if len(self.graph.edges) == 0:
            raise ValueError("Graph has no edges")
            
        # Check graph connectivity
        components = list(nx.weakly_connected_components(self.graph))
        if not components:
            raise ValueError("Graph has no connected components")
            
        largest_component = max(components, key=len)
        component_coverage = len(largest_component) / len(self.graph.nodes)
        
        # Since we're now only using active stops, we expect better connectivity
        if component_coverage < 0.9:  # Increased from 0.8 since we're using filtered stops
            raise ValueError(
                f"Graph validation failed: Largest component only covers {component_coverage:.1%} of nodes. "
                "This suggests a problem with the graph building process."
            )
            
        logger.info(f"Graph validation passed. Largest component covers {component_coverage:.1%} of nodes")
        
    def _load_stops(self) -> None:
        """Load public transport stops from filtered dataset"""
        logger.info("Loading stops...")
        
        # Load stop times first to get active stop IDs
        stop_times_file = self.data_dir / "csv_output" / "stop_times_cleaned_final.csv"
        if not stop_times_file.exists():
            logger.warning("Stop times file not found, using all stops")
            active_stop_ids = None
        else:
            stop_times_df = pd.read_csv(stop_times_file)
            active_stop_ids = set(stop_times_df['stop_id'].unique())
            logger.info(f"Found {len(active_stop_ids)} active stops in stop_times")
        
        # Load stops data
        stops_file = self.data_dir / "csv_output" / "stops_cleaned_filled_final.csv"
        if not stops_file.exists():
            logger.warning("Stops file not found")
            return
            
        stops_df = pd.read_csv(stops_file)
        
        # Filter to only active stops if we have that info
        if active_stop_ids is not None:
            original_len = len(stops_df)
            stops_df = stops_df[stops_df['stop_id'].isin(active_stop_ids)]
            logger.info(f"Filtered stops from {original_len} to {len(stops_df)} active stops")
        
        # Add stops to graph
        for _, row in stops_df.iterrows():
            stop_node = Node(
                id=f"stop_{row['stop_id']}",
                name=row['stop_name'],
                latitude=row['stop_lat'],
                longitude=row['stop_lon'],
                node_type='stop',
                wheelchair_accessible=row.get('wheelchair_boarding', False)
            )
            self.add_node(stop_node)
            
        logger.info(f"Added {len(stops_df)} stops to graph")

    def _load_pois(self) -> None:
        """Load points of interest"""
        pois_file = self.data_dir / "csv_output" / "pois_with_scenic_scores_cleaned.csv"
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
            # Store scenic score in node attributes
            self.graph.nodes[node.id]['scenic_score'] = float(row['scenic_score'])
            
    def _load_road_network(self) -> None:
        """Load road network and add edges between nodes"""
        roads_file = self.data_dir / "csv_output" / "filtered_roads_cleaned_final.csv"
        if not roads_file.exists():
            raise RuntimeError(f"Roads file not found: {roads_file}")
            
        roads_df = pd.read_csv(roads_file)
        logger.info(f"Loaded {len(roads_df)} road segments")
        
        for _, row in roads_df.iterrows():
            try:
                coords = row['geometry'].strip('LINESTRING ()').split(',')
                start_coords = coords[0].strip().split()
                end_coords = coords[-1].strip().split()
                
                start_lat, start_lon = float(start_coords[1]), float(start_coords[0])
                end_lat, end_lon = float(end_coords[1]), float(end_coords[0])
                distance = self.haversine_distance(start_lat, start_lon, end_lat, end_lon)
                
                # Calculate travel time based on maxspeed and road type
                max_speed = float(row['maxspeed']) if pd.notnull(row['maxspeed']) else 50
                travel_time = (distance / max_speed) * 60  # Convert to minutes
                
                # Calculate scenic value based on road attributes
                scenic_value = 0.0
                
                # Roads near water or in parks are more scenic
                if row.get('bridge') == 'yes':
                    scenic_value += 0.2  # Bridge views are often scenic
                    
                # Surface type affects scenic value
                surface = row.get('surface', '').lower()
                if surface in ['asphalt', 'paved']:
                    scenic_value += 0.1  # Well-maintained roads
                elif surface in ['unpaved', 'gravel']:
                    scenic_value += 0.15  # More natural/rural feel
                    
                # Road type affects scenic value
                highway = row.get('highway', '').lower()
                if highway in ['pedestrian', 'cycleway', 'footway']:
                    scenic_value += 0.2  # Dedicated paths are often in nice areas
                elif highway in ['residential', 'living_street']:
                    scenic_value += 0.15  # Local streets can be charming
                elif highway in ['primary', 'secondary']:
                    scenic_value += 0.05  # Main roads less scenic
                
                # Normalize scenic value to 0-1 range
                scenic_value = min(1.0, scenic_value)
                
                # Create unique node IDs for road endpoints
                start_node = f"road_{start_lat:.6f}_{start_lon:.6f}"
                end_node = f"road_{end_lat:.6f}_{end_lon:.6f}"
                
                # Add nodes if they don't exist
                for node_id, lat, lon in [(start_node, start_lat, start_lon), 
                                        (end_node, end_lat, end_lon)]:
                    if node_id not in self.nodes:
                        node = Node(
                            id=node_id,
                            name=f"Road Junction at {lat:.6f}, {lon:.6f}",
                            latitude=lat,
                            longitude=lon,
                            node_type='junction'
                        )
                        self.add_node(node)
                
                # Add road segment as edge
                self.add_edge(
                    start_node, end_node,
                    transport_mode='road',
                    travel_time=travel_time,
                    cost=0,  # Assume road travel is free
                    distance=distance,
                    wheelchair_accessible=True,  # Assume roads are accessible by default
                    scenic_value=scenic_value,
                    bidirectional=row.get('oneway', 'no').lower() != 'yes'
                )
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing road segment: {str(e)}")
                continue

    def _load_transport_routes(self) -> None:
        """Load public transport routes and add edges between connected stops"""
        logger.info("Loading transport routes from stop times...")
        
        # Load stop times data
        stop_times_file = self.data_dir / "csv_output" / "stop_times_cleaned_final.csv"
        if not stop_times_file.exists():
            logger.warning("Stop times file not found, skipping route connections")
            return
            
        stop_times_df = pd.read_csv(stop_times_file)
        logger.info(f"Loaded {len(stop_times_df)} stop times")
        
        # Sort by trip_id and stop_sequence
        stop_times_df = stop_times_df.sort_values(['trip_id', 'stop_sequence'])
        
        # Create a dictionary to store the best connections between stops
        connections = {}  # (from_stop, to_stop) -> {'distance': X, 'time': Y, 'cost': Z, 'count': N}
        
        # Process each trip to find connections
        for trip_id, trip_stops in stop_times_df.groupby('trip_id'):
            # Get consecutive pairs of stops in this trip
            for i in range(len(trip_stops) - 1):
                current = trip_stops.iloc[i]
                next_stop = trip_stops.iloc[i + 1]
                
                # Create key for this stop pair
                pair_key = (current['stop_id'], next_stop['stop_id'])
                
                # Calculate metrics
                distance = float(next_stop['shape_dist_traveled'] - current['shape_dist_traveled'])
                fare = float(next_stop['fare_units_traveled'] - current['fare_units_traveled'])
                
                try:
                    current_time = pd.to_datetime(current['departure_time'])
                    next_time = pd.to_datetime(next_stop['arrival_time'])
                    travel_time = (next_time - current_time).total_seconds() / 60  # Convert to minutes
                except (ValueError, pd.errors.OutOfBoundsDatetime):
                    travel_time = distance / 500  # Estimate: 30 km/h = 0.5 km/min
                
                # Ensure positive values
                distance = max(0.001, distance)  # At least 1m
                travel_time = max(0.1, travel_time)  # At least 6 seconds
                fare = max(0, fare)
                
                # Update connection info
                if pair_key not in connections:
                    connections[pair_key] = {
                        'distance': distance,
                        'time': travel_time,
                        'cost': fare,
                        'count': 1
                    }
                else:
                    # Update metrics:
                    # - Keep the shortest distance (it's fixed)
                    # - Keep the minimum travel time (fastest connection)
                    # - Keep the minimum fare (cheapest connection)
                    # - Count how many trips use this connection
                    conn = connections[pair_key]
                    conn['distance'] = min(conn['distance'], distance)
                    conn['time'] = min(conn['time'], travel_time)
                    conn['cost'] = min(conn['cost'], fare)
                    conn['count'] += 1
        
        # Add edges for each unique connection
        logger.info(f"Found {len(connections)} unique stop connections")
        edges_added = 0
        
        for (from_stop, to_stop), metrics in connections.items():
            try:
                self.add_edge(
                    start_node=f"stop_{from_stop}",
                    end_node=f"stop_{to_stop}",
                    transport_mode='bus',  # Using bus as default mode
                    travel_time=metrics['time'],
                    cost=metrics['cost'],
                    distance=metrics['distance'],
                    wheelchair_accessible=True,  # We can update this if we have accessibility data
                    bidirectional=False  # Transit routes are directional
                )
                edges_added += 1
            except KeyError as e:
                logger.debug(f"Skipping edge {from_stop} -> {to_stop}: {str(e)}")
                continue
                
        logger.info(f"Added {edges_added} edges between stops")

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
        
        # Find nearest neighbors within 500m for each point
        # Convert 500m to degrees (approximate)
        max_distance_deg = 0.5 / 111  # 111km per degree
        
        # Query k=10 nearest neighbors for each point (adjust k based on density)
        distances, indices = tree.query(points, k=10, distance_upper_bound=max_distance_deg)
        
        transfer_count = 0
        with tqdm(total=len(stop_nodes), desc="Creating transfers") as pbar:
            for i, (dists, idxs) in enumerate(zip(distances, indices)):
                valid_idxs = idxs[dists < max_distance_deg]
                valid_idxs = valid_idxs[valid_idxs != i]  # Remove self
                
                for j in valid_idxs:
                    if j >= len(stop_nodes):  # Skip invalid indices from KDTree
                        continue
                        
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
                pbar.update(1)
                    
        logger.info(f"Created {transfer_count} transfer connections")

    def _add_walking_connections(self, max_walking_distance: float = 1.0) -> None:
        """Add walking connections between POIs and nearby transport nodes"""
        from scipy.spatial import cKDTree
        import numpy as np
        
        logger.info("Adding walking connections to POIs...")
        
        # Get POIs and transport nodes
        pois = [node for node in self.nodes.values() if node.node_type == 'poi']
        transport_nodes = [node for node in self.nodes.values() 
                         if node.node_type in ('stop', 'road')]
                         
        if not pois or not transport_nodes:
            logger.warning("No POIs or transport nodes found")
            return
            
        # Create KD-tree for transport nodes
        transport_points = np.array([[node.latitude, node.longitude] for node in transport_nodes])
        tree = cKDTree(transport_points)
        
        # Convert max walking distance to degrees
        max_distance_deg = max_walking_distance / 111
        
        # For each POI, find nearest transport nodes
        poi_points = np.array([[poi.latitude, poi.longitude] for poi in pois])
        
        # Query k=5 nearest transport nodes for each POI
        distances, indices = tree.query(poi_points, k=5, distance_upper_bound=max_distance_deg)
        
        connection_count = 0
        with tqdm(total=len(pois), desc="Adding POI connections") as pbar:
            for poi_idx, (dists, idxs) in enumerate(zip(distances, indices)):
                valid_idxs = idxs[dists < max_distance_deg]
                
                for transport_idx in valid_idxs:
                    if transport_idx >= len(transport_nodes):  # Skip invalid indices
                        continue
                        
                    poi = pois[poi_idx]
                    node = transport_nodes[transport_idx]
                    
                    # Calculate exact distance
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
                pbar.update(1)
                    
        logger.info(f"Created {connection_count} POI walking connections")
        
    def _add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph with all its attributes"""
        # Check if edge with same transport mode already exists
        existing_edges = self.graph.get_edge_data(edge.source, edge.target)
        if existing_edges:
            for key, data in existing_edges.items():
                if data['transport_mode'] == edge.transport_mode:
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

    def find_shortest_path(self, source: str, target: str, weight: str = 'distance', scenic_preference: float = 0.0) -> tuple:
        """Find shortest path between two nodes using specified weight
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Weight to use for path finding ('distance', 'time', 'cost')
            scenic_preference: How much to prioritize scenic routes (0.0 to 1.0)
            
        Returns:
            tuple: (path, total_distance, total_time, total_cost, avg_scenic_value)
        """
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Source or target node not found in graph")
            
        # Create weight function that considers scenic value
        def edge_weight(u, v, edge_data):
            base_weight = edge_data.get(weight, float('inf'))
            if scenic_preference > 0:
                # Reduce weight for scenic edges (making them more likely to be chosen)
                scenic_value = edge_data.get('scenic_value', 0.0)
                return base_weight * (1 - scenic_preference * scenic_value)
            return base_weight
            
        try:
            # Find shortest path using custom weight function
            path = nx.shortest_path(self.graph, source, target, weight=edge_weight)
            
            # Calculate path metrics
            total_distance = 0
            total_time = 0
            total_cost = 0
            total_scenic = 0
            
            for i in range(len(path)-1):
                edges = self.get_edge_info(path[i], path[i+1])
                if edges:
                    # Get the first (best) edge between these nodes
                    edge = edges[0]
                    total_distance += edge.get('distance', 0)
                    total_time += edge.get('travel_time', 0)
                    total_cost += edge.get('cost', 0)
                    total_scenic += edge.get('scenic_value', 0)
            
            avg_scenic_value = total_scenic / (len(path) - 1) if len(path) > 1 else 0
            
            return path, total_distance, total_time, total_cost, avg_scenic_value
            
        except nx.NetworkXNoPath:
            raise ValueError(f"No path found between {source} and {target}")
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

    def _calculate_combined_scenic_value(self, base_score: float, nearby_scores: list) -> float:
        """Calculate combined scenic value using diminishing returns formula
        
        Args:
            base_score: Base scenic score of the road/POI
            nearby_scores: List of scenic scores from nearby features
            
        Returns:
            Combined scenic score between 0 and 1
        """
        if not nearby_scores:
            return base_score
            
        # Sort scores in descending order
        sorted_scores = sorted(nearby_scores, reverse=True)
        
        # Each additional score contributes less
        combined = base_score
        for i, score in enumerate(sorted_scores):
            # Diminishing factor based on position
            factor = 1.0 / (2 ** i)  # 1, 1/2, 1/4, 1/8, etc.
            combined += score * factor
            
        # Normalize to 0-1 range
        return min(1.0, combined)

    def _build_spatial_index(self) -> tuple:
        """Build spatial index for efficient proximity queries
        
        Returns:
            tuple: (KDTree, list of node IDs, list of coordinates)
        """
        # Collect coordinates and IDs
        coords = []
        node_ids = []
        
        for node_id, node in self.nodes.items():
            # Access node attributes directly
            coords.append([node.longitude, node.latitude])
            node_ids.append(node_id)
            
        # Build KDTree for efficient spatial queries
        tree = cKDTree(coords)
        return tree, node_ids, coords

    def _propagate_scenic_scores(self) -> None:
        """Propagate scenic scores from POIs to nearby roads"""
        logger.info("Propagating scenic scores...")
        
        # Build spatial index
        tree, node_ids, coords = self._build_spatial_index()
        
        # Get POI nodes and their scores
        poi_nodes = {
            node_id: self.graph.nodes[node_id].get('scenic_score', 0.0)
            for node_id in self.nodes
            if self.nodes[node_id].node_type == 'poi'
        }
        
        # For each road segment
        road_updates = {}
        for node_id, node in self.nodes.items():
            if node.node_type == 'junction':
                # Find POIs within 250m
                nearby_indices = tree.query_ball_point(
                    [node.longitude, node.latitude], 
                    r=0.00225  # Approximately 250m in degrees
                )
                
                nearby_scores = []
                for idx in nearby_indices:
                    poi_id = node_ids[idx]
                    if poi_id in poi_nodes:
                        distance = self.haversine_distance(
                            node.latitude, node.longitude,
                            self.nodes[poi_id].latitude, 
                            self.nodes[poi_id].longitude
                        )
                        
                        # Calculate score contribution based on distance
                        if distance <= 0.1:  # 100m
                            score_contribution = poi_nodes[poi_id] * 0.5
                        else:  # 100m-250m
                            score_contribution = poi_nodes[poi_id] * 0.25
                            
                        nearby_scores.append(score_contribution)
                
                if nearby_scores:
                    # Get base scenic value
                    base_score = self.graph.nodes[node_id].get('scenic_value', 0.0)
                    
                    # Calculate combined score
                    combined_score = self._calculate_combined_scenic_value(base_score, nearby_scores)
                    
                    # Store update
                    road_updates[node_id] = combined_score
        
        # Apply updates
        nx.set_node_attributes(self.graph, road_updates, 'scenic_value')