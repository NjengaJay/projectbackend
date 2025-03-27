import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
from models.user import User
from models.destination import Destination
from models.trip import Trip
from models import db
from api.auth import auth_bp
from api.destinations import destinations_bp
from api.trips import trips_bp
from api.chat import chat_bp
from api.sentiment import sentiment_bp
from api.nlp import nlp_bp
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_sample_data():
    """Initialize sample data in the database"""
    try:
        # Check if destinations already exist
        if Destination.query.first() is None:
            # Add sample destinations to database
            destinations = [
                {'id': 1, 'name': 'Amsterdam', 'country': 'Netherlands', 'description': 'Beautiful canals and rich history', 'image_url': 'https://example.com/amsterdam.jpg', 'budget_range': 'Medium', 'best_time_to_visit': 'Spring'},
                {'id': 2, 'name': 'Rotterdam', 'country': 'Netherlands', 'description': 'Modern architecture and vibrant culture', 'image_url': 'https://example.com/rotterdam.jpg', 'budget_range': 'Medium', 'best_time_to_visit': 'Summer'},
                {'id': 3, 'name': 'Utrecht', 'country': 'Netherlands', 'description': 'Historic university city', 'image_url': 'https://example.com/utrecht.jpg', 'budget_range': 'Medium', 'best_time_to_visit': 'Spring'},
                {'id': 4, 'name': 'Delft', 'country': 'Netherlands', 'description': 'Famous for Delft Blue pottery', 'image_url': 'https://example.com/delft.jpg', 'budget_range': 'Medium', 'best_time_to_visit': 'Summer'}
            ]
            for dest in destinations:
                destination = Destination(
                    id=dest['id'],
                    name=dest['name'],
                    country=dest['country'],
                    description=dest.get('description', ''),
                    image_url=dest.get('image_url', ''),
                    budget_range=dest.get('budget_range', ''),
                    best_time_to_visit=dest.get('best_time_to_visit', '')
                )
                db.session.add(destination)
            
            db.session.commit()
            print("Sample destinations initialized successfully")
    except Exception as e:
        print(f"Error initializing sample data: {e}")
        db.session.rollback()

def create_app():
    app = Flask(__name__)
    
    # Database configuration - Using SQLite for development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///travel_assistant.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = 'dev-secret-key'  # Change this in production
    
    # Initialize extensions
    db.init_app(app)
    jwt = JWTManager(app)
    migrate = Migrate(app, db)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Create database tables
    with app.app_context():
        db.create_all()
        init_sample_data()
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(destinations_bp, url_prefix='/api/destinations')
    app.register_blueprint(trips_bp, url_prefix='/api/trips')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(sentiment_bp, url_prefix='/api/sentiment')
    app.register_blueprint(nlp_bp, url_prefix='/api/nlp')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return jsonify({'status': 'healthy'}), 200

    # Root endpoint
    @app.route('/')
    def root():
        return jsonify({
            'message': 'Welcome to the Travel Assistant API',
            'version': '1.0',
            'endpoints': {
                'auth': '/api/auth',
                'destinations': '/api/destinations',
                'trips': '/api/trips',
                'chat': '/api/chat',
                'sentiment': '/api/sentiment',
                'nlp': '/api/nlp',
                'health': '/api/health'
            }
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=8000)

    # commented