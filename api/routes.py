from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import datetime
from . import db
from .models import User, Trip, Destination, Review
from .chatbot_handler import ChatbotHandler

# Create blueprints
auth_bp = Blueprint('auth', __name__)
chatbot_bp = Blueprint('chatbot', __name__)
travel_bp = Blueprint('travel', __name__)
nlp_bp = Blueprint('nlp', __name__)

# Initialize chatbot handler
chatbot = ChatbotHandler()

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not all(k in data for k in ['username', 'email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already registered'}), 400
            
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already taken'}), 400
            
        # Create new user
        user = User(
            username=data['username'],
            email=data['email'],
            password=data['password']  # Password hashing is handled in the User model
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Create access token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'Registration successful',
            'access_token': access_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"Registration error: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not all(k in data for k in ['email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Find user
        user = User.query.filter_by(email=data['email']).first()
        
        # Verify password
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401
            
        # Create access token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        })
        
    except Exception as e:
        print(f"Login error: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

@chatbot_bp.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        # Process message
        result = chatbot.process_message(message)
        
        if not result['success']:
            return jsonify({'error': result['error']}), 500
            
        return jsonify({
            'response': result['response'],
            'sentiment': result['sentiment'],
            'user_cluster': result['user_cluster']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Basic travel-related endpoints
@travel_bp.route('/destinations', methods=['GET'])
@jwt_required()
def get_destinations():
    """Get list of destinations"""
    try:
        destinations = Destination.query.all()
        return jsonify([{
            'id': d.id,
            'name': d.name,
            'country': d.country,
            'description': d.description
        } for d in destinations])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@travel_bp.route('/trips/plan', methods=['POST'])
@jwt_required()
def plan_trip():
    data = request.get_json()
    
    if not data or not all(key in data for key in ['title', 'start_date', 'end_date', 'destination_id']):
        return jsonify({'message': 'Missing required fields!'}), 400
    
    destination = Destination.query.get(data['destination_id'])
    if not destination:
        return jsonify({'message': 'Destination not found!'}), 404
    
    try:
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'message': 'Invalid date format! Use YYYY-MM-DD'}), 400
    
    if start_date >= end_date:
        return jsonify({'message': 'End date must be after start date!'}), 400
    
    new_trip = Trip(
        title=data['title'],
        start_date=start_date,
        end_date=end_date,
        preferences=data.get('preferences'),
        budget=data.get('budget'),
        user_id=get_jwt_identity(),
        destination_id=destination.id
    )
    
    db.session.add(new_trip)
    db.session.commit()
    
    return jsonify({
        'message': 'Trip planned successfully!',
        'trip': {
            'id': new_trip.id,
            'title': new_trip.title,
            'start_date': new_trip.start_date.isoformat(),
            'end_date': new_trip.end_date.isoformat(),
            'destination': {
                'id': destination.id,
                'name': destination.name,
                'country': destination.country
            }
        }
    }), 201

# NLP routes
@nlp_bp.route('/analyze', methods=['POST'])
@jwt_required()
def analyze_query():
    """Analyze a travel-related query using NLP"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400

        query = data['query']
        
        # Initialize NLP model if not already initialized
        if not hasattr(current_app, 'nlp_model'):
            from ..nlp.model import TravelAssistantNLP
            current_app.nlp_model = TravelAssistantNLP()
            current_app.nlp_model.load_model('travel_assistant_model')

        # Analyze query
        result = current_app.nlp_model.predict(query)
        
        return jsonify({
            'success': True,
            'analysis': {
                'intent': result['intent'],
                'confidence': result['confidence'],
                'entities': result['entities'],
                'sentiment': result['sentiment']
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@nlp_bp.route('/train', methods=['POST'])
@jwt_required()
def train_model():
    """Train or retrain the NLP model"""
    try:
        # Only allow admins to train the model
        user = User.query.get(get_jwt_identity())
        if not user.is_admin:
            return jsonify({'error': 'Unauthorized'}), 403

        from ..nlp.sample_data import generate_sample_data
        from ..nlp.model import TravelAssistantNLP
        
        # Generate training data
        texts, labels = generate_sample_data()
        
        # Initialize and train model
        model = TravelAssistantNLP()
        model.train(
            train_texts=texts[:40],  # Use 80% for training
            train_labels=labels[:40],
            eval_texts=texts[40:],   # Use 20% for evaluation
            eval_labels=labels[40:],
            output_dir='travel_assistant_model'
        )
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot_bp.route('/user/preferences', methods=['POST'])
@jwt_required()
def update_preferences():
    """Update user preferences for personalization"""
    try:
        data = request.get_json()
        
        # Save preferences
        chatbot.save_user_preferences(get_jwt_identity(), data)
        
        return jsonify({'message': 'Preferences updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
