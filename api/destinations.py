from flask import Blueprint, jsonify
from models.destination import Destination

destinations_bp = Blueprint('destinations', __name__)

# Sample destinations data
destinations = [
    {
        'id': 1,
        'name': 'Paris',
        'country': 'France',
        'description': 'The City of Light, known for the Eiffel Tower, Louvre Museum, and exquisite cuisine.',
        'image_url': 'https://example.com/paris.jpg',
        'budget_range': '$200-500 per day',
        'best_time_to_visit': 'April to October'
    },
    {
        'id': 2,
        'name': 'Tokyo',
        'country': 'Japan',
        'description': 'A bustling metropolis blending ultra-modern technology with traditional culture.',
        'image_url': 'https://example.com/tokyo.jpg',
        'budget_range': '$150-400 per day',
        'best_time_to_visit': 'March to May or September to November'
    },
    {
        'id': 3,
        'name': 'Bali',
        'country': 'Indonesia',
        'description': 'A tropical paradise known for beaches, temples, and vibrant culture.',
        'image_url': 'https://example.com/bali.jpg',
        'budget_range': '$50-200 per day',
        'best_time_to_visit': 'April to October'
    }
]

@destinations_bp.route('/', methods=['GET'])
def get_destinations():
    """Get all destinations"""
    try:
        # Get destinations from database
        db_destinations = Destination.query.all()
        return jsonify([dest.to_dict() for dest in db_destinations])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@destinations_bp.route('/<int:destination_id>', methods=['GET'])
def get_destination(destination_id):
    """Get a specific destination by ID"""
    try:
        destination = Destination.query.get(destination_id)
        if not destination:
            return jsonify({'error': 'Destination not found'}), 404
        return jsonify(destination.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
