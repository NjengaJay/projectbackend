from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from api.models import db, Reservation, User, Accommodation
from datetime import datetime

reservations_bp = Blueprint('reservations', __name__)

@reservations_bp.before_request
def log_request_info():
    current_app.logger.info('Request Method: %s', request.method)
    current_app.logger.info('Request Path: %s', request.path)
    current_app.logger.info('Request Headers: %s', dict(request.headers))

@reservations_bp.route('/', methods=['GET', 'POST', 'OPTIONS'])
@jwt_required()
def reservations():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        current_app.logger.info('Returning OPTIONS response: %s', dict(response.headers))
        return response, 200

    current_user_id = get_jwt_identity()
    
    if request.method == 'GET':
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 10, type=int)
            
            # Only show active and confirmed reservations
            query = Reservation.query.filter_by(user_id=current_user_id).filter(
                Reservation.status.in_(['active', 'confirmed'])
            )
            paginated_reservations = query.paginate(page=page, per_page=per_page, error_out=False)
            
            response_data = {
                'items': [reservation.to_dict() for reservation in paginated_reservations.items],
                'total': paginated_reservations.total,
                'pages': paginated_reservations.pages,
                'current_page': paginated_reservations.page
            }
            current_app.logger.info('Successfully fetched reservations: %s', response_data)
            return jsonify(response_data), 200
        except Exception as e:
            current_app.logger.error('Error fetching reservations: %s', str(e))
            return jsonify({'error': 'Failed to fetch reservations'}), 500
    
    # POST request
    try:
        data = request.get_json()
        accommodation_id = data.get('accommodation_id')
        check_in = datetime.strptime(data.get('check_in'), '%Y-%m-%d').date()
        check_out = datetime.strptime(data.get('check_out'), '%Y-%m-%d').date()
        guests = data.get('guests')
        total_price = data.get('total_price')
        
        if not all([accommodation_id, check_in, check_out, guests, total_price]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Validate accommodation exists
        accommodation = Accommodation.query.get(accommodation_id)
        if not accommodation:
            return jsonify({'error': 'Accommodation not found'}), 404
        
        # Create reservation
        reservation = Reservation(
            user_id=current_user_id,
            accommodation_id=accommodation_id,
            check_in=check_in,
            check_out=check_out,
            guests=guests,
            total_price=total_price,
            status='confirmed'
        )
        
        db.session.add(reservation)
        db.session.commit()
        
        current_app.logger.info('Successfully created reservation: %s', reservation.to_dict())
        return jsonify(reservation.to_dict()), 201
    except ValueError as e:
        return jsonify({'error': 'Invalid date format'}), 400
    except Exception as e:
        current_app.logger.error('Error creating reservation: %s', str(e))
        return jsonify({'error': 'Failed to create reservation'}), 500

@reservations_bp.route('/<int:id>/cancel', methods=['POST', 'OPTIONS'])
@jwt_required()
def cancel_reservation(id):
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        current_app.logger.info('Returning OPTIONS response: %s', dict(response.headers))
        return response, 200

    current_user_id = get_jwt_identity()
    try:
        reservation = Reservation.query.filter_by(id=id, user_id=current_user_id).first()
        
        if not reservation:
            return jsonify({'error': 'Reservation not found'}), 404
        
        if reservation.status == 'cancelled':
            return jsonify({'error': 'Reservation already cancelled'}), 400
        
        reservation.status = 'cancelled'
        db.session.commit()
        
        current_app.logger.info('Successfully cancelled reservation: %s', reservation.to_dict())
        return jsonify(reservation.to_dict()), 200
    except Exception as e:
        current_app.logger.error('Error cancelling reservation: %s', str(e))
        return jsonify({'error': 'Failed to cancel reservation'}), 500
