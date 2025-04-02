from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os
from ..models.profile import UserProfile, AccessibilityPreferences, Favorite
from ..models.user import User
from .. import db

profile_bp = Blueprint('profile', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@profile_bp.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    current_user_id = get_jwt_identity()
    profile = UserProfile.query.filter_by(user_id=current_user_id).first()
    
    if not profile:
        return jsonify({'message': 'Profile not found'}), 404
    
    user = User.query.get(current_user_id)
    response_data = {
        **profile.to_dict(),
        'username': user.username,
        'email': user.email
    }
    
    if profile.accessibility_preferences:
        response_data['accessibility'] = profile.accessibility_preferences.to_dict()
    
    return jsonify(response_data)

@profile_bp.route('/api/profile/picture', methods=['POST'])
@jwt_required()
def upload_profile_picture():
    if 'file' not in request.files:
        return jsonify({'message': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        user_id = get_jwt_identity()
        
        # Create user-specific directory if it doesn't exist
        user_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        file_path = os.path.join(user_dir, filename)
        file.save(file_path)
        
        # Update profile picture path in database
        profile = UserProfile.query.filter_by(user_id=user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id)
            db.session.add(profile)
        
        profile.profile_picture = os.path.join(str(user_id), filename)
        db.session.commit()
        
        return jsonify({
            'message': 'Profile picture uploaded successfully',
            'path': profile.profile_picture
        })
        
    return jsonify({'message': 'Invalid file type'}), 400

@profile_bp.route('/api/profile/accessibility', methods=['GET', 'POST', 'PUT', 'DELETE'])
@jwt_required()
def manage_accessibility():
    current_user_id = get_jwt_identity()
    user_profile = UserProfile.query.filter_by(user_id=current_user_id).first()
    
    if not user_profile:
        return jsonify({"error": "User profile not found"}), 404
        
    if request.method == 'GET':
        preferences = AccessibilityPreferences.query.filter_by(profile_id=user_profile.id).first()
        
        if not preferences:
            return jsonify({"message": "No accessibility preferences set"}), 404
            
        return jsonify(preferences.to_dict()), 200

    elif request.method in ['POST', 'PUT']:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['wheelchair_access', 'screen_reader', 'high_contrast']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields", "required": required_fields}), 400
            
        preferences = AccessibilityPreferences.query.filter_by(profile_id=user_profile.id).first()
        
        if not preferences:
            # Create new preferences
            preferences = AccessibilityPreferences(
                profile_id=user_profile.id,
                wheelchair_access=data['wheelchair_access'],
                screen_reader=data['screen_reader'],
                high_contrast=data['high_contrast']
            )
            db.session.add(preferences)
        else:
            # Update existing preferences
            preferences.wheelchair_access = data['wheelchair_access']
            preferences.screen_reader = data['screen_reader']
            preferences.high_contrast = data['high_contrast']
            preferences.updated_at = datetime.utcnow()
        
        try:
            db.session.commit()
            return jsonify(preferences.to_dict()), 200
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error updating accessibility preferences: {str(e)}")
            return jsonify({"error": "Failed to update accessibility preferences"}), 500

    elif request.method == 'DELETE':
        preferences = AccessibilityPreferences.query.filter_by(profile_id=user_profile.id).first()
        
        if not preferences:
            return jsonify({"message": "No accessibility preferences found"}), 404
            
        try:
            db.session.delete(preferences)
            db.session.commit()
            return jsonify({"message": "Accessibility preferences deleted successfully"}), 200
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error deleting accessibility preferences: {str(e)}")
            return jsonify({"error": "Failed to delete accessibility preferences"}), 500

@profile_bp.route('/api/profile/favorites', methods=['GET'])
@jwt_required()
def get_favorites():
    current_user_id = get_jwt_identity()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    favorites = Favorite.query.filter_by(user_id=current_user_id)\
        .order_by(Favorite.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'items': [fav.to_dict() for fav in favorites.items],
        'total': favorites.total,
        'pages': favorites.pages,
        'current_page': favorites.page
    })

@profile_bp.route('/api/profile/favorites/<int:accommodation_id>', methods=['POST', 'DELETE'])
@jwt_required()
def manage_favorite(accommodation_id):
    current_user_id = get_jwt_identity()
    
    if request.method == 'POST':
        existing = Favorite.query.filter_by(
            user_id=current_user_id,
            accommodation_id=accommodation_id
        ).first()
        
        if existing:
            return jsonify({'message': 'Already in favorites'}), 400
            
        favorite = Favorite(user_id=current_user_id, accommodation_id=accommodation_id)
        db.session.add(favorite)
        db.session.commit()
        return jsonify(favorite.to_dict()), 201
    
    # DELETE method
    favorite = Favorite.query.filter_by(
        user_id=current_user_id,
        accommodation_id=accommodation_id
    ).first()
    
    if not favorite:
        return jsonify({'message': 'Favorite not found'}), 404
        
    db.session.delete(favorite)
    db.session.commit()
    return '', 204
