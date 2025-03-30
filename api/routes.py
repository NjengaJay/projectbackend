from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import datetime
from . import db
from .models import User, Trip, Destination, Review, Accommodation
from .chatbot_handler import ChatbotHandler
from nlp.sentiment_analyzer import TripAdvisorSentimentAnalyzer
import json

# Create blueprints
auth_bp = Blueprint('auth', __name__)
chatbot_bp = Blueprint('chatbot', __name__)
travel_bp = Blueprint('travel', __name__)
nlp_bp = Blueprint('nlp', __name__)
recommendation_bp = Blueprint('recommendation', __name__)
accommodation_bp = Blueprint('accommodation', __name__)

# Initialize handlers lazily
chatbot = None
sentiment_analyzer = None

def get_chatbot():
    global chatbot
    if chatbot is None:
        chatbot = ChatbotHandler(init_recommender=False)
    return chatbot

def get_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = TripAdvisorSentimentAnalyzer()
    return sentiment_analyzer

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
        result = get_chatbot().process_message(message)
        
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
def analyze_text():
    """Analyze text using NLP models"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Analyze sentiment
        sentiment_score = get_sentiment_analyzer().analyze_sentiment(text)
        
        return jsonify({
            'sentiment': sentiment_score,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        get_chatbot().save_user_preferences(get_jwt_identity(), data)
        
        return jsonify({'message': 'Preferences updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_price_range(star_rating):
    """Calculate price range based on star rating"""
    star_rating = float(star_rating) if star_rating else 3.0
    base_price = 50 + star_rating * 25
    max_price = base_price * 2
    return {
        'min': int(base_price),
        'max': int(max_price)
    }

@accommodation_bp.route('/api/accommodations', methods=['GET'])
@jwt_required()
def get_accommodations():
    try:
        # Get query parameters
        search_term = request.args.get('search', '').lower()
        location = request.args.get('location', '')
        min_price = float(request.args.get('min_price', 0))
        max_price = float(request.args.get('max_price', float('inf')))
        
        # Get accessibility filters
        accessibility = request.args.get('accessibility', '')
        accessibility_filters = json.loads(accessibility) if accessibility else {}
        
        # Get accommodation type filters
        acc_type = request.args.get('type', '')
        type_filters = json.loads(acc_type) if acc_type else {}
        
        # Base query
        query = Accommodation.query
        
        # Apply search filter
        if search_term:
            query = query.filter(
                (Accommodation.name.ilike(f'%{search_term}%')) |
                (Accommodation.description.ilike(f'%{search_term}%'))
            )
        
        # Apply location filter
        if location:
            query = query.filter(Accommodation.city.ilike(f'%{location}%'))
        
        # Apply price filter
        query = query.filter(
            (Accommodation.price_per_night >= min_price) &
            (Accommodation.price_per_night <= max_price)
        )
        
        # Apply accessibility filters
        if accessibility_filters:
            for feature, value in accessibility_filters.items():
                if value:
                    query = query.filter(
                        Accommodation.accessibility_features.like(f'%{feature}%')
                    )
        
        # Apply type filters
        if type_filters:
            type_conditions = [
                Accommodation.type.ilike(f'%{acc_type}%')
                for acc_type, selected in type_filters.items()
                if selected
            ]
            if type_conditions:
                query = query.filter(db.or_(*type_conditions))
        
        # Execute query with pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        paginated_results = query.paginate(page=page, per_page=per_page)
        
        # Format results
        accommodations = []
        for acc in paginated_results.items:
            # Calculate price range based on star rating
            price_range = calculate_price_range(acc.star_rating)

            # Generate room types based on price range
            room_types = [
                {
                    'name': 'Standard Room',
                    'price': price_range['min'],
                    'description': 'Comfortable room with basic amenities'
                },
                {
                    'name': 'Deluxe Room',
                    'price': int((price_range['min'] + price_range['max']) / 2),
                    'description': 'Spacious room with premium amenities'
                },
                {
                    'name': 'Suite',
                    'price': price_range['max'],
                    'description': 'Luxury suite with separate living area'
                }
            ]

            acc_data = {
                'id': acc.id,
                'name': acc.name,
                'description': acc.description,
                'type': acc.type,
                'city': acc.city,
                'country': acc.country,
                'price_range': price_range,
                'room_types': room_types,
                'star_rating': float(acc.star_rating) if acc.star_rating else None,
                'review_score': float(acc.review_score) if acc.review_score else None,
                'image_url': 'https://images.unsplash.com/photo-1566073771259-6a8506099945?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
                'accessibility_features': json.loads(acc.accessibility_features) if acc.accessibility_features else {},
                'amenities': json.loads(acc.amenities) if acc.amenities else []
            }
            accommodations.append(acc_data)
        
        return jsonify({
            'accommodations': accommodations,
            'total': paginated_results.total,
            'pages': paginated_results.pages,
            'current_page': page
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@accommodation_bp.route('/api/accommodations/<int:id>', methods=['GET'])
@jwt_required()
def get_accommodation(id):
    try:
        accommodation = Accommodation.query.get_or_404(id)
        
        # Calculate price range based on star rating
        price_range = calculate_price_range(accommodation.star_rating)

        # Generate room types based on price range
        room_types = [
            {
                'name': 'Standard Room',
                'price': price_range['min'],
                'description': 'Comfortable room with basic amenities'
            },
            {
                'name': 'Deluxe Room',
                'price': int((price_range['min'] + price_range['max']) / 2),
                'description': 'Spacious room with premium amenities'
            },
            {
                'name': 'Suite',
                'price': price_range['max'],
                'description': 'Luxury suite with separate living area'
            }
        ]
        
        return jsonify({
            'id': accommodation.id,
            'name': accommodation.name,
            'description': accommodation.description,
            'type': accommodation.type,
            'city': accommodation.city,
            'country': accommodation.country,
            'price_range': price_range,
            'room_types': room_types,
            'star_rating': float(accommodation.star_rating) if accommodation.star_rating else None,
            'review_score': float(accommodation.review_score) if accommodation.review_score else None,
            'image_url': 'https://images.unsplash.com/photo-1566073771259-6a8506099945?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'accessibility_features': json.loads(accommodation.accessibility_features) if accommodation.accessibility_features else {},
            'amenities': json.loads(accommodation.amenities) if accommodation.amenities else []
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
