from flask import Blueprint, request, jsonify
from datetime import datetime
import uuid
from models import db, User, Trip, Destination, TripPlan
from .auth import token_required

travel_bp = Blueprint('travel', __name__)

@travel_bp.route('/destinations', methods=['GET'])
def get_destinations():
    destinations = Destination.query.all()
    return jsonify([{
        'id': d.id,
        'name': d.name,
        'country': d.country,
        'description': d.description,
        'image_url': d.image_url,
        'best_time_to_visit': d.best_time_to_visit,
        'estimated_cost': d.estimated_cost,
        'activities': d.activities
    } for d in destinations])

@travel_bp.route('/trip-plans', methods=['GET', 'POST'])
@token_required
def handle_trip_plans(current_user):
    if request.method == 'GET':
        trip_plans = TripPlan.query.filter_by(user_id=current_user.id).all()
        return jsonify([plan.to_dict() for plan in trip_plans])
    
    data = request.get_json()
    required_fields = ['destination', 'start_date', 'end_date', 'budget']
    
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing required fields'}), 400
    
    try:
        # Create a new trip plan
        trip_plan = TripPlan(
            user_id=current_user.id,
            destination=data['destination'],
            start_date=datetime.fromisoformat(data['start_date']),
            end_date=datetime.fromisoformat(data['end_date']),
            budget=float(data['budget']),
            preferences=data.get('preferences', {}),
            accessibility_requirements=data.get('accessibility_requirements', []),
            share_token=str(uuid.uuid4())
        )
        
        # Generate a simple itinerary based on destination and budget
        destination = Destination.query.filter_by(name=data['destination']).first()
        if destination:
            trip_plan.itinerary = {
                'daily_plans': [
                    {
                        'day': 1,
                        'activities': destination.activities[:2],
                        'estimated_cost': destination.estimated_cost / 3
                    },
                    {
                        'day': 2,
                        'activities': destination.activities[2:] if len(destination.activities) > 2 else destination.activities,
                        'estimated_cost': destination.estimated_cost / 3
                    }
                ],
                'total_cost': destination.estimated_cost
            }
        
        db.session.add(trip_plan)
        db.session.commit()
        
        return jsonify(trip_plan.to_dict()), 201
        
    except ValueError:
        return jsonify({'message': 'Invalid date format. Use ISO format (YYYY-MM-DD)'}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error creating trip plan', 'error': str(e)}), 500

@travel_bp.route('/trip-plans/<int:plan_id>', methods=['GET', 'PUT', 'DELETE'])
@token_required
def handle_trip_plan(current_user, plan_id):
    trip_plan = TripPlan.query.get_or_404(plan_id)
    
    if trip_plan.user_id != current_user.id:
        return jsonify({'message': 'Unauthorized access'}), 403
    
    if request.method == 'GET':
        return jsonify(trip_plan.to_dict())
    
    elif request.method == 'PUT':
        data = request.get_json()
        
        try:
            if 'start_date' in data:
                trip_plan.start_date = datetime.fromisoformat(data['start_date'])
            if 'end_date' in data:
                trip_plan.end_date = datetime.fromisoformat(data['end_date'])
            if 'budget' in data:
                trip_plan.budget = float(data['budget'])
            if 'preferences' in data:
                trip_plan.preferences.update(data['preferences'])
            if 'accessibility_requirements' in data:
                trip_plan.accessibility_requirements = data['accessibility_requirements']
            
            db.session.commit()
            return jsonify(trip_plan.to_dict())
            
        except ValueError:
            return jsonify({'message': 'Invalid date format. Use ISO format (YYYY-MM-DD)'}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({'message': 'Error updating trip plan', 'error': str(e)}), 500
    
    elif request.method == 'DELETE':
        try:
            db.session.delete(trip_plan)
            db.session.commit()
            return jsonify({'message': 'Trip plan deleted successfully'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'message': 'Error deleting trip plan', 'error': str(e)}), 500

@travel_bp.route('/shared-trip/<share_token>', methods=['GET'])
def view_shared_trip(share_token):
    trip_plan = TripPlan.query.filter_by(share_token=share_token).first_or_404()
    return jsonify(trip_plan.to_dict())
