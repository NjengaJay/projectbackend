from flask import Blueprint, jsonify, request
from ..models import db, Destination, Accommodation

destinations = Blueprint('destinations', __name__)

@destinations.route('/destinations', methods=['GET'])
def get_destinations():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    destinations = Destination.query.paginate(
        page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'destinations': [dest.to_dict() for dest in destinations.items],
        'total': destinations.total,
        'pages': destinations.pages,
        'current_page': destinations.page
    }), 200

@destinations.route('/destinations/<int:destination_id>', methods=['GET'])
def get_destination(destination_id):
    destination = Destination.query.get_or_404(destination_id)
    return jsonify(destination.to_dict()), 200

@destinations.route('/destinations/<int:destination_id>/accommodations', methods=['GET'])
def get_destination_accommodations(destination_id):
    destination = Destination.query.get_or_404(destination_id)
    accommodations = Accommodation.query.filter_by(destination_id=destination_id).all()
    return jsonify([acc.to_dict() for acc in accommodations]), 200

@destinations.route('/destinations/search', methods=['GET'])
def search_destinations():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'message': 'No search query provided'}), 400
    
    destinations = Destination.query.filter(
        (Destination.name.ilike(f'%{query}%')) |
        (Destination.country.ilike(f'%{query}%')) |
        (Destination.description.ilike(f'%{query}%'))
    ).all()
    
    return jsonify([dest.to_dict() for dest in destinations]), 200
