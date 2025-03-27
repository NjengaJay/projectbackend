from flask import Blueprint, jsonify

destinations_bp = Blueprint('destinations', __name__)

# Hardcoded destinations data for MVP
DESTINATIONS = [
    {
        "id": 1,
        "name": "Paris, France",
        "description": "The City of Light, known for its iconic Eiffel Tower, world-class museums, and exquisite cuisine.",
        "image_url": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34",
        "best_time_to_visit": "April to October",
        "estimated_budget": 2500,
        "accessibility_features": ["wheelchair accessible", "audio guides", "elevators"],
        "activities": ["Visit the Louvre", "Climb the Eiffel Tower", "Seine River Cruise"],
        "type": "city"
    },
    {
        "id": 2,
        "name": "Bali, Indonesia",
        "description": "A tropical paradise with beautiful beaches, lush rice terraces, and rich cultural heritage.",
        "image_url": "https://images.unsplash.com/photo-1537996194471-e657df975ab4",
        "best_time_to_visit": "April to October",
        "estimated_budget": 1500,
        "accessibility_features": ["beach wheelchairs", "accessible villas"],
        "activities": ["Visit temples", "Surf lessons", "Rice terrace tours"],
        "type": "beach"
    },
    {
        "id": 3,
        "name": "Queenstown, New Zealand",
        "description": "Adventure capital of the world, offering stunning landscapes and adrenaline-pumping activities.",
        "image_url": "https://images.unsplash.com/photo-1589308078059-be1415eab4c3",
        "best_time_to_visit": "December to February",
        "estimated_budget": 3000,
        "accessibility_features": ["adaptive equipment", "guided tours"],
        "activities": ["Bungee jumping", "Skiing", "Lake cruises"],
        "type": "adventure"
    },
    {
        "id": 4,
        "name": "Kyoto, Japan",
        "description": "Ancient capital of Japan, famous for its temples, traditional gardens, and tea ceremonies.",
        "image_url": "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e",
        "best_time_to_visit": "March to May, October to November",
        "estimated_budget": 2000,
        "accessibility_features": ["ramps", "accessible transportation"],
        "activities": ["Temple visits", "Tea ceremony", "Garden tours"],
        "type": "cultural"
    }
]

@destinations_bp.route('/api/destinations', methods=['GET'])
def get_destinations():
    return jsonify(DESTINATIONS)
