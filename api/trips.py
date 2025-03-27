from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import jwt
from models import db
from models.trip import Trip
from models.user import User
from models.destination import Destination
from api.destinations import destinations  # Import the sample destinations

trips_bp = Blueprint('trips', __name__)

def get_user_from_token():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return None, ('Missing authorization token', 401)
        
    try:
        payload = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        user = User.query.get(payload['user_id'])
        if not user:
            return None, ('User not found', 404)
        return user, None
    except jwt.ExpiredSignatureError:
        return None, ('Token has expired', 401)
    except jwt.InvalidTokenError:
        return None, ('Invalid token', 401)

@trips_bp.route('/saved', methods=['GET'])
def get_saved_trips():
    try:
        user, error = get_user_from_token()
        if error:
            return jsonify({'message': error[0]}), error[1]
            
        trips = Trip.query.filter_by(user_id=user.id).all()
        return jsonify([trip.to_dict() for trip in trips])
    except Exception as e:
        current_app.logger.error(f"Error in get_saved_trips: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@trips_bp.route('/plan', methods=['POST'])
def plan_trip():
    user, error = get_user_from_token()
    if error:
        return jsonify({'message': error[0]}), error[1]
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['destination_id', 'start_date', 'end_date', 'budget']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Validate destination exists in our sample data
        destination = next((d for d in destinations if d['id'] == data['destination_id']), None)
        if not destination:
            return jsonify({'error': 'Invalid destination ID'}), 400

        # Create new trip
        new_trip = Trip(
            user_id=user.id,
            destination_id=data['destination_id'],
            start_date=datetime.strptime(data['start_date'], '%Y-%m-%d').date(),
            end_date=datetime.strptime(data['end_date'], '%Y-%m-%d').date(),
            budget=float(data['budget']),
            preferences=data.get('preferences', {}),
            itinerary={
                'daily_plans': [
                    {
                        'day': 1,
                        'activities': [
                            f'Visit popular attractions in {destination["name"]}',
                            f'Try local cuisine in {destination["name"]}',
                            'Evening leisure activities'
                        ],
                        'estimated_cost': float(data['budget']) * 0.3
                    },
                    {
                        'day': 2,
                        'activities': [
                            'Morning sightseeing',
                            'Afternoon cultural activities',
                            'Evening entertainment'
                        ],
                        'estimated_cost': float(data['budget']) * 0.4
                    }
                ],
                'total_cost': float(data['budget']) * 0.7
            }
        )

        db.session.add(new_trip)
        db.session.commit()

        return jsonify({
            'message': 'Trip planned successfully',
            'trip': new_trip.to_dict()
        }), 201

    except ValueError as e:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error in plan_trip: {str(e)}")
        return jsonify({'error': str(e)}), 500

@trips_bp.route('/<int:trip_id>', methods=['GET'])
def get_trip(trip_id):
    user, error = get_user_from_token()
    if error:
        return jsonify({'message': error[0]}), error[1]
        
    trip = Trip.query.filter_by(id=trip_id, user_id=user.id).first()
    if not trip:
        return jsonify({'message': 'Trip not found'}), 404
        
    return jsonify(trip.to_dict())
