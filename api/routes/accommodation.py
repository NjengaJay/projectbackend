from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from api.models import db, Accommodation
import json
from flask_cors import cross_origin
import urllib.parse
import traceback

accommodation_bp = Blueprint('accommodation', __name__, url_prefix='/api/accommodations')

@accommodation_bp.route('/', methods=['GET', 'OPTIONS'])
def get_accommodations():
    try:
        if request.method == 'OPTIONS':
            response = jsonify({'message': 'OK'})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            current_app.logger.info('Returning OPTIONS response: %s', dict(response.headers))
            return response, 200
            
        # Get query parameters
        search = request.args.get('search', '')
        min_price = request.args.get('min_price', type=float, default=0)
        max_price = request.args.get('max_price', type=float, default=float('inf'))
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Get filter parameters
        accessibility = request.args.get('accessibility', '{}')
        accommodation_type = request.args.get('type', '{}')
        
        try:
            accessibility = json.loads(accessibility)
            accommodation_type = json.loads(accommodation_type)
        except json.JSONDecodeError:
            accessibility = {}
            accommodation_type = {}
        
        # Build query
        query = Accommodation.query
        
        if search:
            search = urllib.parse.unquote(search)
            search = f"%{search}%"
            query = query.filter(
                (Accommodation.name.ilike(search)) |
                (Accommodation.city.ilike(search)) |
                (Accommodation.description.ilike(search))
            )
        
        # Get all accommodations that match the search
        accommodations = query.all()
        current_app.logger.info(f'Found {len(accommodations)} accommodations before filtering')
        
        # Filter by price range
        filtered_accommodations = []
        for acc in accommodations:
            try:
                if acc.price_range:
                    price_range = json.loads(acc.price_range)
                    acc_min = float(price_range.get("min", 0))
                    acc_max = float(price_range.get("max", float('inf')))
                    if acc_min <= max_price and acc_max >= min_price:
                        filtered_accommodations.append(acc)
                else:
                    filtered_accommodations.append(acc)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                filtered_accommodations.append(acc)
        
        current_app.logger.info(f'Found {len(filtered_accommodations)} accommodations after price filtering')
        
        # Apply type filters
        if accommodation_type and any(accommodation_type.values()):
            filtered_accommodations = [
                acc for acc in filtered_accommodations
                if acc.type and any(acc.type == key for key, value in accommodation_type.items() if value)
            ]
            current_app.logger.info(f'Found {len(filtered_accommodations)} accommodations after type filtering')
        
        # Apply accessibility filters
        if accessibility and any(accessibility.values()):
            filtered_results = []
            current_app.logger.info(f'Applying accessibility filters: {accessibility}')
            
            def has_accessibility_feature(features, feature_name):
                try:
                    if not features:
                        return False
                    features_dict = json.loads(features) if isinstance(features, str) else features
                    
                    # Map frontend feature names to database structure
                    feature_mapping = {
                        'wheelchair_accessible': ('mobility', ['Wheelchair accessible entrance', 'Accessible parking']),
                        'elevator_access': ('mobility', ['Elevator']),
                        'ground_floor': ('mobility', ['Ground floor access']),
                        'step_free_access': ('mobility', ['Step-free access'])
                    }
                    
                    if feature_name not in feature_mapping:
                        return False
                        
                    category, db_features = feature_mapping[feature_name]
                    if category not in features_dict:
                        return False
                        
                    return any(feature['name'] in db_features for feature in features_dict[category])
                except (json.JSONDecodeError, KeyError, TypeError):
                    return False
            
            for acc in filtered_accommodations:
                # Check if accommodation has any of the selected accessibility features
                selected_features = [key for key, value in accessibility.items() if value]
                if any(has_accessibility_feature(acc.accessibility_features, feature) for feature in selected_features):
                    filtered_results.append(acc)
                    
            filtered_accommodations = filtered_results
            current_app.logger.info(f'Found {len(filtered_accommodations)} accommodations after accessibility filtering')
        
        # Paginate results
        total = len(filtered_accommodations)
        pages = (total + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        
        # Prepare response
        accommodations = [acc.to_dict() for acc in filtered_accommodations[start:end]]
        
        response_data = {
            'items': accommodations,
            'total': total,
            'pages': pages,
            'current_page': page
        }
        
        current_app.logger.info(f'Returning {len(accommodations)} accommodations')
        return jsonify(response_data), 200
        
    except Exception as e:
        current_app.logger.error('Error fetching accommodations: %s\n%s', str(e), traceback.format_exc())
        return jsonify({'error': 'Failed to fetch accommodations'}), 500

@accommodation_bp.route('/<int:id>', methods=['GET'])
def get_accommodation(id):
    try:
        accommodation = Accommodation.query.get_or_404(id)
        return jsonify(accommodation.to_dict()), 200
    except Exception as e:
        current_app.logger.error('Error fetching accommodation %s: %s', id, str(e))
        return jsonify({'error': 'Failed to fetch accommodation'}), 500
