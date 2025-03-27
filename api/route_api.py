from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Optional
from datetime import datetime
import folium
from ..route_optimizer.mcsp import MCSpRouter
from ..route_optimizer.graph_model import TransportGraph
from pydantic import BaseModel
from ..route_optimizer.routing import Route
import os

class RouteRequest(BaseModel):
    source: str
    destination: str
    preferences: Dict[str, float]
    accessibility_required: bool = False
    max_routes: int = 3
    departure_time: Optional[datetime] = None

class RouteResponse(BaseModel):
    routes: List[Dict]
    map_url: str
    total_options: int
    computation_time: float

app = FastAPI(title="Route Optimization API")
graph = TransportGraph()  # Initialize with data
router = MCSpRouter(graph)

@app.post("/api/route/optimize", response_model=RouteResponse)
async def optimize_route(request: RouteRequest):
    """Find optimal routes based on user preferences"""
    try:
        # Update router weights based on user preferences
        if request.preferences:
            router.update_weights(request.preferences)
        
        # Get weather and traffic data (mock for now)
        weather_conditions = {
            'rain': 0.0,
            'snow': 0.0,
            'wind': 0.0
        }
        
        traffic_conditions = {
            'congestion_level': 0.0
        }
        
        # Find Pareto-optimal routes
        routes = router.find_pareto_optimal_routes(
            request.source,
            request.destination,
            max_routes=request.max_routes
        )
        
        if not routes:
            raise HTTPException(
                status_code=404,
                detail="No routes found"
            )
        
        # Create map visualization
        map_url = create_route_map(routes)
        
        # Convert routes to API response format
        route_responses = []
        for route in routes:
            route_responses.append({
                'steps': [
                    {
                        'from_location': step.from_node.name,
                        'to_location': step.to_node.name,
                        'transport_mode': step.transport_mode,
                        'travel_time': step.travel_time,
                        'cost': step.cost,
                        'distance': step.distance
                    }
                    for step in route.steps
                ],
                'total_time': route.total_time,
                'total_cost': route.total_cost,
                'total_distance': route.total_distance,
                'scenic_score': sum(
                    getattr(step, 'scenic_value', 0)
                    for step in route.steps
                )
            })
        
        return RouteResponse(
            routes=route_responses,
            map_url=map_url,
            total_options=len(routes),
            computation_time=0.0  # TODO: Add actual computation time
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def get_transport_mode_style(mode: str, color: str) -> Dict:
    """Get line style based on transport mode"""
    style = {
        'stroke': True,
        'color': color,
        'opacity': 0.8,
        'smoothFactor': 1.0,
        'noClip': False,
        'weight': 2
    }
    
    if mode == 'train':
        style['weight'] = 4
        style['dashArray'] = '10,10'
    elif mode == 'bus':
        style['weight'] = 3
        style['dashArray'] = '5,5'
    elif mode == 'walk':
        style['weight'] = 2
        style['dashArray'] = '2,4'
    elif mode == 'tram':
        style['weight'] = 3
        style['dashArray'] = '8,4'
    elif mode == 'metro':
        style['weight'] = 4
        style['dashArray'] = '15,5'
    
    return style

def get_stop_icon(node_type: str) -> folium.Icon:
    """Get appropriate icon for stop type"""
    if node_type == 'stop':
        return folium.Icon(color='red', icon='fa-flag', prefix='fa')
    elif node_type == 'poi':
        return folium.Icon(color='green', icon='fa-info-circle', prefix='fa')
    else:
        return folium.Icon(color='blue', icon='fa-info-circle', prefix='fa')

def create_stop_popup(node) -> str:
    """Create popup content for a stop"""
    content = f"""
    <div style="min-width: 200px;">
        <h4>{node.name}</h4>
        <p>Type: {node.node_type}</p>
        <p>Location: {node.latitude:.4f}, {node.longitude:.4f}</p>
    """
    
    if hasattr(node, 'description'):
        content += f"<p>{node.description}</p>"
    if hasattr(node, 'scenic_score') and node.scenic_score > 0:
        content += f"<p>Scenic Score: {node.scenic_score:.2f}</p>"
    if hasattr(node, 'wheelchair_accessible'):
        content += f"""<p>
            {'<span style="color: green;">♿ Wheelchair Accessible</span>' if node.wheelchair_accessible else '<span style="color: red;">❌ Not Wheelchair Accessible</span>'}
        </p>"""
    
    content += "</div>"
    return content

def create_route_map(routes: List[Route], center_lat: float = 52.0, center_lon: float = 5.0) -> str:
    """Create an interactive map visualization of routes"""
    # Get all coordinates to calculate bounds
    all_coords = []
    for route in routes:
        for step in route.steps:
            all_coords.append((step.from_node.latitude, step.from_node.longitude))
            all_coords.append((step.to_node.latitude, step.to_node.longitude))
    
    # Calculate bounds
    if all_coords:
        min_lat = min(lat for lat, _ in all_coords)
        max_lat = max(lat for lat, _ in all_coords)
        min_lon = min(lon for _, lon in all_coords)
        max_lon = max(lon for _, lon in all_coords)
        
        # Center map on route bounds
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    
    # Create base map centered on calculated center
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='cartodbpositron'
    )
    
    # Colors for different routes
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # Add each route to the map
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        
        # Create route line with different styles per transport mode
        for step in route.steps:
            # Get line style based on transport mode
            style = get_transport_mode_style(step.transport_mode, color)
            
            # Create line coordinates
            coords = [
                [step.from_node.latitude, step.from_node.longitude],
                [step.to_node.latitude, step.to_node.longitude]
            ]
            
            # Add line to map with style
            line = folium.PolyLine(
                coords,
                tooltip=f"{step.transport_mode}: {int(step.travel_time)}min, €{step.cost:.2f}",
                **style
            )
            line.add_to(m)
            
            # Add markers for stops
            folium.Marker(
                [step.from_node.latitude, step.from_node.longitude],
                popup=create_stop_popup(step.from_node),
                icon=get_stop_icon(step.from_node.node_type)
            ).add_to(m)
        
        # Add final destination marker
        last_step = route.steps[-1]
        folium.Marker(
            [last_step.to_node.latitude, last_step.to_node.longitude],
            popup=create_stop_popup(last_step.to_node),
            icon=get_stop_icon(last_step.to_node.node_type)
        ).add_to(m)
    
    # Fit map bounds to include all routes
    if all_coords:
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    
    # Save map to file
    os.makedirs("static", exist_ok=True)
    output_path = "static/route_map.html"
    m.save(output_path)
    return output_path

@app.get("/api/route/stats")
async def get_route_stats():
    """Get statistics about route optimization performance"""
    return {
        "total_routes_computed": 0,  # TODO: Add tracking
        "average_computation_time": 0.0,
        "most_popular_routes": [],
        "success_rate": 0.0
    }

@app.post("/api/route/feedback")
async def submit_route_feedback(
    route_id: str,
    satisfaction_score: float,
    comments: Optional[str] = None
):
    """Submit user feedback for route optimization training"""
    # TODO: Store feedback and use it for training
    return {"status": "success"}
