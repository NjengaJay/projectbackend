from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.models import User, UserPreference, Review, db

user_bp = Blueprint('user', __name__)

@user_bp.route('/preferences', methods=['GET'])
@jwt_required()
def get_preferences():
    """Get user preferences"""
    current_user_id = get_jwt_identity()
    
    try:
        preferences = UserPreference.query.filter_by(user_id=current_user_id).first()
        if not preferences:
            return jsonify({'message': 'No preferences set'}), 404
            
        return jsonify({
            'preferences': {
                'budget_range': preferences.budget_range,
                'preferred_activities': preferences.preferred_activities,
                'accessibility_needs': preferences.accessibility_needs,
                'preferred_climate': preferences.preferred_climate,
                'travel_style': preferences.travel_style
            }
        }), 200
    except Exception as e:
        return jsonify({'error': 'Failed to fetch preferences'}), 500

@user_bp.route('/preferences', methods=['POST', 'PUT'])
@jwt_required()
def update_preferences():
    """Create or update user preferences"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    try:
        preferences = UserPreference.query.filter_by(user_id=current_user_id).first()
        
        if not preferences:
            # Create new preferences
            preferences = UserPreference(
                user_id=current_user_id,
                budget_range=data.get('budget_range'),
                preferred_activities=data.get('preferred_activities', []),
                accessibility_needs=data.get('accessibility_needs', []),
                preferred_climate=data.get('preferred_climate'),
                travel_style=data.get('travel_style')
            )
            db.session.add(preferences)
        else:
            # Update existing preferences
            if 'budget_range' in data:
                preferences.budget_range = data['budget_range']
            if 'preferred_activities' in data:
                preferences.preferred_activities = data['preferred_activities']
            if 'accessibility_needs' in data:
                preferences.accessibility_needs = data['accessibility_needs']
            if 'preferred_climate' in data:
                preferences.preferred_climate = data['preferred_climate']
            if 'travel_style' in data:
                preferences.travel_style = data['travel_style']
        
        db.session.commit()
        
        return jsonify({
            'message': 'Preferences updated successfully',
            'preferences': {
                'budget_range': preferences.budget_range,
                'preferred_activities': preferences.preferred_activities,
                'accessibility_needs': preferences.accessibility_needs,
                'preferred_climate': preferences.preferred_climate,
                'travel_style': preferences.travel_style
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to update preferences'}), 500

@user_bp.route('/reviews', methods=['GET'])
@jwt_required()
def get_user_reviews():
    """Get all reviews by the current user"""
    current_user_id = get_jwt_identity()
    
    try:
        reviews = Review.query.filter_by(user_id=current_user_id).all()
        return jsonify({
            'reviews': [{
                'id': review.id,
                'destination_id': review.destination_id,
                'rating': review.rating,
                'content': review.content,
                'sentiment_score': review.sentiment_score,
                'created_at': review.created_at.isoformat()
            } for review in reviews]
        }), 200
    except Exception as e:
        return jsonify({'error': 'Failed to fetch reviews'}), 500

@user_bp.route('/reviews/<int:destination_id>', methods=['POST'])
@jwt_required()
def create_review(destination_id):
    """Create a review for a destination"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or 'rating' not in data or 'content' not in data:
        return jsonify({'error': 'Rating and content are required'}), 400
        
    try:
        # Check if user already reviewed this destination
        existing_review = Review.query.filter_by(
            user_id=current_user_id,
            destination_id=destination_id
        ).first()
        
        if existing_review:
            return jsonify({'error': 'You have already reviewed this destination'}), 409
            
        # Create new review
        review = Review(
            user_id=current_user_id,
            destination_id=destination_id,
            rating=data['rating'],
            content=data['content']
        )
        
        # TODO: Add sentiment analysis score
        
        db.session.add(review)
        db.session.commit()
        
        return jsonify({
            'message': 'Review created successfully',
            'review': {
                'id': review.id,
                'destination_id': review.destination_id,
                'rating': review.rating,
                'content': review.content,
                'sentiment_score': review.sentiment_score,
                'created_at': review.created_at.isoformat()
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to create review'}), 500

@user_bp.route('/reviews/<int:review_id>', methods=['PUT'])
@jwt_required()
def update_review(review_id):
    """Update a review"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    try:
        review = Review.query.filter_by(id=review_id, user_id=current_user_id).first()
        if not review:
            return jsonify({'error': 'Review not found'}), 404
            
        if 'rating' in data:
            review.rating = data['rating']
        if 'content' in data:
            review.content = data['content']
            # TODO: Update sentiment analysis score
            
        db.session.commit()
        
        return jsonify({
            'message': 'Review updated successfully',
            'review': {
                'id': review.id,
                'destination_id': review.destination_id,
                'rating': review.rating,
                'content': review.content,
                'sentiment_score': review.sentiment_score,
                'created_at': review.created_at.isoformat()
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to update review'}), 500

@user_bp.route('/reviews/<int:review_id>', methods=['DELETE'])
@jwt_required()
def delete_review(review_id):
    """Delete a review"""
    current_user_id = get_jwt_identity()
    
    try:
        review = Review.query.filter_by(id=review_id, user_id=current_user_id).first()
        if not review:
            return jsonify({'error': 'Review not found'}), 404
            
        db.session.delete(review)
        db.session.commit()
        
        return jsonify({'message': 'Review deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to delete review'}), 500
