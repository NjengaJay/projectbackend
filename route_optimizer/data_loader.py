import os
import pandas as pd
from typing import Dict, List, Optional
import networkx as nx
from .graph_model import TransportGraph, Node, Edge

class GTFSDataLoader:
    """Load and process GTFS data files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def load_stops(self) -> pd.DataFrame:
        """Load stops data"""
        stops_file = os.path.join(self.data_dir, 'stops_cleaned_filled.csv')
        if not os.path.exists(stops_file):
            raise FileNotFoundError(f"Stops file not found: {stops_file}")
        
        stops = pd.read_csv(stops_file)
        # Convert wheelchair_boarding to boolean
        stops['wheelchair_boarding'] = stops['wheelchair_boarding'].astype(bool)
        required_columns = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
        
        if not all(col in stops.columns for col in required_columns):
            raise ValueError("Stops file missing required columns")
            
        return stops
    
    def load_routes(self) -> pd.DataFrame:
        """Load routes data"""
        routes_file = os.path.join(self.data_dir, 'routes.csv')
        if not os.path.exists(routes_file):
            raise FileNotFoundError(f"Routes file not found: {routes_file}")
        
        routes = pd.read_csv(routes_file)
        # Convert wheelchair_accessible to boolean
        routes['wheelchair_accessible'] = routes['wheelchair_accessible'].astype(bool)
        required_columns = ['route_id', 'route_name', 'route_type']
        
        if not all(col in routes.columns for col in required_columns):
            raise ValueError("Routes file missing required columns")
            
        return routes
    
    def load_trips(self) -> pd.DataFrame:
        """Load trips data"""
        trips_file = os.path.join(self.data_dir, 'trips.csv')
        if not os.path.exists(trips_file):
            raise FileNotFoundError(f"Trips file not found: {trips_file}")
        
        trips = pd.read_csv(trips_file)
        # Convert wheelchair_accessible to boolean
        trips['wheelchair_accessible'] = trips['wheelchair_accessible'].astype(bool)
        required_columns = ['trip_id', 'route_id', 'service_id']
        
        if not all(col in trips.columns for col in required_columns):
            raise ValueError("Trips file missing required columns")
            
        return trips
    
    def load_stop_times(self) -> pd.DataFrame:
        """Load stop times data"""
        stop_times_file = os.path.join(self.data_dir, 'stop_times.csv')
        if not os.path.exists(stop_times_file):
            raise FileNotFoundError(f"Stop times file not found: {stop_times_file}")
        
        stop_times = pd.read_csv(stop_times_file)
        required_columns = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
        
        if not all(col in stop_times.columns for col in required_columns):
            raise ValueError("Stop times file missing required columns")
            
        return stop_times
    
    def create_transport_graph(self) -> TransportGraph:
        """Create a TransportGraph from GTFS data"""
        try:
            print("Loading GTFS data...")
            
            # Load all required data
            stops = self.load_stops()
            routes = self.load_routes()
            trips = self.load_trips()
            stop_times = self.load_stop_times()
            
            print(f"Loaded data: {len(stops)} stops, {len(routes)} routes, {len(trips)} trips, {len(stop_times)} stop times")
            
            # Create graph
            graph = TransportGraph()
            
            # Add stops as nodes
            for _, stop in stops.iterrows():
                # Convert numeric stop_id to string
                stop_id = str(stop['stop_id'])
                # Use wheelchair_boarding column for wheelchair_accessible
                wheelchair_accessible = stop.get('wheelchair_boarding', False)
                graph.add_node(
                    id=f"stop_{stop_id}",
                    name=stop['stop_name'],
                    latitude=stop['stop_lat'],
                    longitude=stop['stop_lon'],
                    node_type='stop',
                    wheelchair_accessible=wheelchair_accessible
                )
            
            print(f"Added {len(graph.nodes)} nodes to the graph")
            
            # Process stop times to create edges
            trip_routes = trips.merge(routes, on='route_id')
            stop_times_with_routes = stop_times.merge(trip_routes, on='trip_id')
            
            print(f"Processing {len(stop_times_with_routes)} stop times to create edges")
            
            # Group by trip to create edges between consecutive stops
            edge_count = 0
            for trip_id, trip_stops in stop_times_with_routes.groupby('trip_id'):
                trip_stops = trip_stops.sort_values('stop_sequence')
                
                for i in range(len(trip_stops) - 1):
                    from_stop = trip_stops.iloc[i]
                    to_stop = trip_stops.iloc[i + 1]
                    
                    # Calculate travel time (in minutes)
                    try:
                        from_time = pd.to_datetime(from_stop['departure_time'])
                        to_time = pd.to_datetime(to_stop['arrival_time'])
                        travel_time = (to_time - from_time).total_seconds() / 60
                    except (ValueError, KeyError):
                        # If time data is missing or invalid, use a default value
                        travel_time = 30.0  # Default 30 minutes
                    
                    # Find stop coordinates safely
                    try:
                        from_stop_data = stops[stops['stop_id'] == from_stop['stop_id']]
                        to_stop_data = stops[stops['stop_id'] == to_stop['stop_id']]
                        
                        if from_stop_data.empty or to_stop_data.empty:
                            print(f"Warning: Stop not found - from_id: {from_stop['stop_id']}, to_id: {to_stop['stop_id']}")
                            continue
                        
                        from_lat = from_stop_data['stop_lat'].iloc[0]
                        from_lon = from_stop_data['stop_lon'].iloc[0]
                        to_lat = to_stop_data['stop_lat'].iloc[0]
                        to_lon = to_stop_data['stop_lon'].iloc[0]
                        
                        # Calculate distance (rough approximation)
                        distance = ((to_lat - from_lat) ** 2 + (to_lon - from_lon) ** 2) ** 0.5 * 111  # km
                        
                        # Add edge if it doesn't exist
                        from_id = f"stop_{str(from_stop['stop_id'])}"
                        to_id = f"stop_{str(to_stop['stop_id'])}"
                        
                        # Check wheelchair accessibility
                        from_accessible = from_stop_data['wheelchair_boarding'].iloc[0]
                        to_accessible = to_stop_data['wheelchair_boarding'].iloc[0]
                        edge_accessible = from_accessible and to_accessible
                        
                        if not graph.graph.has_edge(from_id, to_id):
                            transport_mode = 'train' if from_stop.get('route_type', 0) == 2 else 'bus'
                            graph.add_edge(
                                from_id,
                                to_id,
                                transport_mode=transport_mode,
                                travel_time=travel_time,
                                cost=distance * 0.5,  # €0.5 per km
                                distance=distance,
                                wheelchair_accessible=edge_accessible
                            )
                            edge_count += 1
                    except (IndexError, KeyError) as e:
                        print(f"Warning: Error processing stops - {str(e)}")
                        continue
            
            print(f"Added {edge_count} edges to the graph")
            print(f"Final graph: {len(graph.nodes)} nodes, {len(graph.graph.edges())} edges")
            
            return graph
            
        except FileNotFoundError as e:
            print(f"Error loading test data: {str(e)}")
            raise
