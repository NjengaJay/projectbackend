from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import random

trips_bp = Blueprint('trips', __name__)

# Store trips in memory for MVP
SAVED_TRIPS = []

# Sample activities for different trip types
ACTIVITIES = {
    'beach': [
        'Sunrise beach yoga', 'Snorkeling tour', 'Sunset sailing',
        'Beach volleyball', 'Surfing lesson', 'Beachfront dinner'
    ],
    'city': [
        'Museum visit', 'Walking food tour', 'Historic site tour',
        'Shopping at local markets', 'Evening entertainment', 'Local cooking class'
    ],
    'adventure': [
        'Hiking expedition', 'Rock climbing', 'White water rafting',
        'Mountain biking', 'Zip lining', 'Camping under stars'
    ],
    'cultural': [
        'Temple/shrine visit', 'Traditional craft workshop', 'Local festival attendance',
        'Historical walking tour', 'Traditional music show', 'Tea ceremony'
    ]
}

def generate_itinerary(trip_data):
    """Generate a fake itinerary based on user preferences"""
    start_date = datetime.strptime(trip_data['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(trip_data['end_date'], '%Y-%m-%d')
    num_days = (end_date - start_date).days + 1
    budget = float(trip_data['budget'])
    daily_budget = budget / num_days

    # Select activities based on preferences
    available_activities = []
    for trip_type in trip_data['preferences']['type']:
        if trip_type in ACTIVITIES:
            available_activities.extend(ACTIVITIES[trip_type])

    if not available_activities:
        available_activities = sum(ACTIVITIES.values(), [])

    daily_plans = []
    total_cost = 0

    for day in range(num_days):
        # Select 3 random activities for each day
        day_activities = random.sample(available_activities, min(3, len(available_activities)))
        day_cost = random.uniform(daily_budget * 0.7, daily_budget * 1.1)
        total_cost += day_cost

        daily_plans.append({
            "day": day + 1,
            "activities": day_activities,
            "estimated_cost": round(day_cost, 2)
        })

    return {
        "daily_plans": daily_plans,
        "total_cost": round(total_cost, 2)
    }

@trips_bp.route('/api/trips/plan', methods=['POST'])
def plan_trip():
    trip_data = request.json
    
    # Generate itinerary
    itinerary = generate_itinerary(trip_data)
    
    # Create trip plan
    trip = {
        "id": len(SAVED_TRIPS) + 1,
        "destination": trip_data['destination'],
        "start_date": trip_data['start_date'],
        "end_date": trip_data['end_date'],
        "budget": float(trip_data['budget']),
        "preferences": trip_data['preferences'],
        "itinerary": itinerary,
        "created_at": datetime.now().isoformat()
    }
    
    # Save trip
    SAVED_TRIPS.append(trip)
    
    return jsonify(trip)

@trips_bp.route('/api/trips/saved', methods=['GET'])
def get_saved_trips():
    return jsonify(SAVED_TRIPS)
