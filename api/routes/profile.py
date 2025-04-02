from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..models import db, UserProfile, AccessibilityPreferences, Favorite, Reservation, Accommodation
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc

profile_bp = Blueprint('profile', __name__)

@profile_bp.before_request
def log_request_info():
    current_app.logger.info('Request Method: %s', request.method)
    current_app.logger.info('Request Path: %s', request.path)
    current_app.logger.info('Request Headers: %s', dict(request.headers))

@profile_bp.route('/', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    
    if not profile:
        current_app.logger.error('Profile not found for user %s', user_id)
        return jsonify({"error": "Profile not found"}), 404
    
    current_app.logger.info('Successfully fetched profile for user %s', user_id)
    return jsonify(profile.to_dict()), 200

@profile_bp.route('/accessibility', methods=['GET', 'PUT', 'OPTIONS'])
@jwt_required()
def manage_accessibility():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        current_app.logger.info('Returning OPTIONS response: %s', dict(response.headers))
        return response, 200

    user_id = get_jwt_identity()
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    
    if not profile:
        current_app.logger.error('Profile not found for user %s', user_id)
        return jsonify({"error": "Profile not found"}), 404

    if request.method == 'GET':
        if not profile.accessibility_preferences:
            current_app.logger.info('No accessibility preferences found for user %s', user_id)
            return jsonify({}), 200
        current_app.logger.info('Successfully fetched accessibility preferences for user %s', user_id)
        return jsonify(profile.accessibility_preferences.to_dict()), 200

    # PUT request
    data = request.get_json()
    try:
        if not profile.accessibility_preferences:
            profile.accessibility_preferences = AccessibilityPreferences()
        
        prefs = profile.accessibility_preferences
        prefs.wheelchair_accessible = data.get('wheelchair_accessible', prefs.wheelchair_accessible)
        prefs.step_free_access = data.get('step_free_access', prefs.step_free_access)
        prefs.accessible_parking = data.get('accessible_parking', prefs.accessible_parking)
        prefs.accessible_bathroom = data.get('accessible_bathroom', prefs.accessible_bathroom)
        
        db.session.commit()
        current_app.logger.info('Successfully updated accessibility preferences for user %s', user_id)
        return jsonify(prefs.to_dict()), 200
    except Exception as e:
        current_app.logger.error('Error updating accessibility preferences for user %s: %s', user_id, str(e))
        return jsonify({"error": "Database error"}), 500

@profile_bp.route('/favorites', methods=['GET', 'OPTIONS'])
@jwt_required()
def favorites():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        current_app.logger.info('Returning OPTIONS response: %s', dict(response.headers))
        return response, 200

    current_user_id = get_jwt_identity()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    try:
        query = Favorite.query.filter_by(user_id=current_user_id)
        paginated_favorites = query.paginate(page=page, per_page=per_page, error_out=False)
        
        favorites = []
        for favorite in paginated_favorites.items:
            accommodation = Accommodation.query.get(favorite.accommodation_id)
            if accommodation:
                favorites.append(accommodation.to_dict())
        
        response_data = {
            'items': favorites,
            'total': paginated_favorites.total,
            'pages': paginated_favorites.pages,
            'current_page': paginated_favorites.page
        }
        current_app.logger.info('Successfully fetched favorites: %s', response_data)
        return jsonify(response_data), 200
    except Exception as e:
        current_app.logger.error('Error fetching favorites: %s', str(e))
        return jsonify({'error': 'Failed to fetch favorites'}), 500

@profile_bp.route('/favorites/<int:accommodation_id>', methods=['DELETE', 'OPTIONS'])
@jwt_required()
def favorite_detail(accommodation_id):
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        current_app.logger.info('Returning OPTIONS response: %s', dict(response.headers))
        return response, 200

    current_user_id = get_jwt_identity()
    try:
        favorite = Favorite.query.filter_by(
            user_id=current_user_id,
            accommodation_id=accommodation_id
        ).first()
        
        if not favorite:
            return jsonify({'error': 'Favorite not found'}), 404
        
        db.session.delete(favorite)
        db.session.commit()
        current_app.logger.info('Successfully deleted favorite %s for user %s', accommodation_id, current_user_id)
        return jsonify({'message': 'Favorite removed successfully'}), 200
    except Exception as e:
        current_app.logger.error('Error deleting favorite %s: %s', accommodation_id, str(e))
        return jsonify({'error': 'Failed to remove favorite'}), 500

@profile_bp.route('/bookings', methods=['GET', 'OPTIONS'])
@jwt_required()
def bookings():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        current_app.logger.info('Returning OPTIONS response: %s', dict(response.headers))
        return response, 200

    user_id = get_jwt_identity()
    page = request.args.get('page', 1, type=int)
    per_page = 10

    try:
        bookings = Reservation.query.filter_by(user_id=user_id)\
            .order_by(desc(Reservation.created_at))\
            .paginate(page=page, per_page=per_page)

        response_data = {
            'items': [booking.to_dict() for booking in bookings.items],
            'total': bookings.total,
            'pages': bookings.pages,
            'current_page': bookings.page
        }
        current_app.logger.info('Successfully fetched bookings: %s', response_data)
        return jsonify(response_data), 200
    except SQLAlchemyError as e:
        current_app.logger.error('Error fetching bookings for user %s: %s', user_id, str(e))
        return jsonify({"error": "Failed to fetch bookings"}), 500

@profile_bp.route('/bookings/<int:booking_id>/cancel', methods=['POST', 'OPTIONS'])
@jwt_required()
def cancel_booking(booking_id):
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        current_app.logger.info('Returning OPTIONS response: %s', dict(response.headers))
        return response, 200

    user_id = get_jwt_identity()
    try:
        booking = Reservation.query.filter_by(
            id=booking_id,
            user_id=user_id
        ).first()
        
        if not booking:
            current_app.logger.error('Booking not found for user %s and booking %s', user_id, booking_id)
            return jsonify({"error": "Booking not found"}), 404
            
        if booking.status == 'cancelled':
            current_app.logger.error('Booking is already cancelled for user %s and booking %s', user_id, booking_id)
            return jsonify({"error": "Booking is already cancelled"}), 400
            
        booking.status = 'cancelled'
        db.session.commit()
        current_app.logger.info('Successfully cancelled booking for user %s and booking %s', user_id, booking_id)
        return jsonify({"message": "Booking cancelled successfully"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error('Error cancelling booking for user %s and booking %s: %s', user_id, booking_id, str(e))
        return jsonify({"error": "Failed to cancel booking"}), 500
