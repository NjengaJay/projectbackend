from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

from ..models import Trip, Destination
from .. import db

trips = Blueprint('trips', __name__)

@trips.route('/api/trips', methods=['GET'])
@login_required
def get_trips():
    """Get all trips for the current user"""
    try:
        # Get query parameters for filtering
        status = request.args.get('status')
        
        # Base query
        query = Trip.query.filter_by(user_id=current_user.id)
        
        # Apply filters if provided
        if status:
            query = query.filter_by(status=status)
            
        # Execute query and get trips
        trips = query.order_by(Trip.start_date.desc()).all()
        
        return jsonify({
            'status': 'success',
            'trips': [trip.to_dict() for trip in trips]
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@trips.route('/api/trips/<int:trip_id>', methods=['GET'])
@login_required
def get_trip(trip_id):
    """Get a specific trip by ID"""
    try:
        trip = Trip.query.filter_by(id=trip_id, user_id=current_user.id).first()
        
        if not trip:
            return jsonify({
                'status': 'error',
                'message': 'Trip not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'trip': trip.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@trips.route('/api/trips', methods=['POST'])
@login_required
def create_trip():
    """Create a new trip"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'start_date', 'end_date']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Convert date strings to datetime objects
        try:
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid date format. Use YYYY-MM-DD'
            }), 400
            
        # Create new trip
        new_trip = Trip(
            user_id=current_user.id,
            name=data['name'],
            description=data.get('description'),
            start_date=start_date,
            end_date=end_date,
            status=data.get('status', 'planned'),
            itinerary=data.get('itinerary', []),
            route_details=data.get('route_details', {}),
            accommodation_details=data.get('accommodation_details', {}),
            total_cost=data.get('total_cost', 0.0),
            notes=data.get('notes', {})
        )
        
        # Add destinations if provided
        if 'destination_ids' in data:
            for dest_id in data['destination_ids']:
                destination = Destination.query.get(dest_id)
                if destination:
                    new_trip.destinations.append(destination)
        
        db.session.add(new_trip)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Trip created successfully',
            'trip': new_trip.to_dict()
        }), 201
        
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': 'Database error occurred'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@trips.route('/api/trips/<int:trip_id>', methods=['PUT'])
@login_required
def update_trip(trip_id):
    """Update an existing trip"""
    try:
        trip = Trip.query.filter_by(id=trip_id, user_id=current_user.id).first()
        
        if not trip:
            return jsonify({
                'status': 'error',
                'message': 'Trip not found'
            }), 404
            
        data = request.get_json()
        
        # Update basic fields
        if 'name' in data:
            trip.name = data['name']
        if 'description' in data:
            trip.description = data['description']
        if 'status' in data:
            trip.status = data['status']
        if 'start_date' in data:
            try:
                trip.start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid start_date format. Use YYYY-MM-DD'
                }), 400
        if 'end_date' in data:
            try:
                trip.end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid end_date format. Use YYYY-MM-DD'
                }), 400
                
        # Update JSON fields
        if 'itinerary' in data:
            trip.itinerary = data['itinerary']
        if 'route_details' in data:
            trip.route_details = data['route_details']
        if 'accommodation_details' in data:
            trip.accommodation_details = data['accommodation_details']
        if 'total_cost' in data:
            trip.total_cost = data['total_cost']
        if 'notes' in data:
            trip.notes = data['notes']
            
        # Update destinations if provided
        if 'destination_ids' in data:
            # Clear existing destinations
            trip.destinations = []
            # Add new destinations
            for dest_id in data['destination_ids']:
                destination = Destination.query.get(dest_id)
                if destination:
                    trip.destinations.append(destination)
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Trip updated successfully',
            'trip': trip.to_dict()
        }), 200
        
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': 'Database error occurred'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@trips.route('/api/trips/<int:trip_id>', methods=['DELETE'])
@login_required
def delete_trip(trip_id):
    """Delete a trip"""
    try:
        trip = Trip.query.filter_by(id=trip_id, user_id=current_user.id).first()
        
        if not trip:
            return jsonify({
                'status': 'error',
                'message': 'Trip not found'
            }), 404
            
        db.session.delete(trip)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Trip deleted successfully'
        }), 200
        
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': 'Database error occurred'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@trips.route('/api/trips/search', methods=['GET'])
@login_required
def search_trips():
    """Search trips with various filters"""
    try:
        # Get query parameters
        name = request.args.get('name')
        status = request.args.get('status')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        destination = request.args.get('destination')
        
        # Base query
        query = Trip.query.filter_by(user_id=current_user.id)
        
        # Apply filters
        if name:
            query = query.filter(Trip.name.ilike(f'%{name}%'))
        if status:
            query = query.filter_by(status=status)
        if start_date:
            try:
                start = datetime.strptime(start_date, '%Y-%m-%d')
                query = query.filter(Trip.start_date >= start)
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid start_date format. Use YYYY-MM-DD'
                }), 400
        if end_date:
            try:
                end = datetime.strptime(end_date, '%Y-%m-%d')
                query = query.filter(Trip.end_date <= end)
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid end_date format. Use YYYY-MM-DD'
                }), 400
        if destination:
            query = query.join(Trip.destinations).filter(
                Destination.name.ilike(f'%{destination}%')
            )
        
        # Execute query
        trips = query.order_by(Trip.start_date.desc()).all()
        
        return jsonify({
            'status': 'success',
            'trips': [trip.to_dict() for trip in trips]
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
