from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from api.models import db, Accommodation
import json
from flask_cors import cross_origin
import urllib.parse
import traceback

accommodation_bp = Blueprint('accommodation', __name__)

@accommodation_bp.route('/accommodations', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_accommodations():
    try:
        if request.method == 'OPTIONS':
            return jsonify({}), 200
            
        # Get query parameters
        search = request.args.get('search', '')
        min_price = request.args.get('min_price', type=float, default=0)
        max_price = request.args.get('max_price', type=float, default=float('inf'))
        city = request.args.get('city', '')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Parse JSON strings from URL parameters
        try:
            accessibility = json.loads(urllib.parse.unquote(request.args.get('accessibility', '{}')))
        except:
            accessibility = {}
            
        try:
            acc_type = json.loads(urllib.parse.unquote(request.args.get('type', '{}')))
        except:
            acc_type = {}
        
        # Start with base query
        query = Accommodation.query
        
        # Apply filters
        if search:
            query = query.filter(Accommodation.name.ilike(f'%{search}%'))
            
        if city:
            query = query.filter(Accommodation.city.ilike(f'%{city}%'))
            
        if acc_type and any(acc_type.values()):
            selected_types = [t for t, selected in acc_type.items() if selected]
            if selected_types:
                query = query.filter(Accommodation.type.in_(selected_types))
        
        # Get results
        accommodations = query.all()
        
        # Post-query filtering and conversion to dict
        filtered_accommodations = []
        for acc in accommodations:
            acc_dict = acc.to_dict()
            
            # Price range filter
            price_range = acc_dict['price_range']
            if price_range:
                try:
                    min_acc_price = float(price_range.get('min', 0))
                    max_acc_price = float(price_range.get('max', float('inf')))
                    if min_acc_price > max_price or max_acc_price < min_price:
                        continue
                except (ValueError, TypeError):
                    pass
                    
            # Accessibility filter
            if accessibility and any(accessibility.values()):
                acc_features = acc_dict['accessibility_features']
                has_required_features = True
                
                for feature, required in accessibility.items():
                    if required:
                        # Map frontend feature names to backend feature categories
                        feature_map = {
                            'wheelchairAccessible': ('mobility', ['Wheelchair accessible entrance']),
                            'audioGuides': ('auditory', ['Audio guides']),
                            'visualAids': ('visual', ['Visual aids', 'High contrast signage']),
                            'mobilitySupport': ('mobility', ['Transport assistance'])
                        }
                        
                        if feature in feature_map:
                            category, required_features = feature_map[feature]
                            category_features = acc_features.get(category, [])
                            if not any(req_feature in str(category_features) for req_feature in required_features):
                                has_required_features = False
                                break
                
                if not has_required_features:
                    continue
                    
            # Add to filtered results
            filtered_accommodations.append(acc_dict)
        
        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = filtered_accommodations[start_idx:end_idx]
        
        return jsonify({
            'accommodations': paginated_results,
            'total': len(filtered_accommodations),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(filtered_accommodations) + per_page - 1) // per_page
        })
            
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@accommodation_bp.route('/accommodations/<int:id>', methods=['GET'])
@cross_origin()
def get_accommodation(id):
    try:
        accommodation = Accommodation.query.get_or_404(id)
        return jsonify(accommodation.to_dict())
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
