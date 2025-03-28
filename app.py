import sys
import os
import click

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from api.models import db, User, Destination
from api.routes import auth_bp, chatbot_bp, travel_bp, recommendation_bp, nlp_bp
from api.destinations import destinations_bp

def init_sample_data():
    """Initialize sample data in the database"""
    try:
        # Check if destinations already exist
        if not Destination.query.first():
            # Add sample destinations to database
            destinations = [
                {'name': 'Amsterdam', 'country': 'Netherlands', 'description': 'Beautiful canals and rich history', 'image_url': 'https://example.com/amsterdam.jpg', 'budget_range': 'Medium', 'best_time_to_visit': 'Spring'},
                {'name': 'Rotterdam', 'country': 'Netherlands', 'description': 'Modern architecture and vibrant culture', 'image_url': 'https://example.com/rotterdam.jpg', 'budget_range': 'Medium', 'best_time_to_visit': 'Summer'},
                {'name': 'The Hague', 'country': 'Netherlands', 'description': 'Political center with beautiful beaches', 'image_url': 'https://example.com/hague.jpg', 'budget_range': 'High', 'best_time_to_visit': 'Summer'},
                {'name': 'Delft', 'country': 'Netherlands', 'description': 'Famous for Delft Blue pottery', 'image_url': 'https://example.com/delft.jpg', 'budget_range': 'Medium', 'best_time_to_visit': 'Summer'}
            ]
            for dest in destinations:
                destination = Destination(**dest)
                db.session.add(destination)
            db.session.commit()
            print("Sample destinations added successfully")
    except Exception as e:
        print(f"Error adding sample data: {str(e)}")
        db.session.rollback()

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configure database and JWT
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///travel_assistant.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = 'dev-secret-key'  # Change in production
    
    # Initialize extensions
    CORS(app)
    db.init_app(app)
    jwt = JWTManager(app)
    migrate = Migrate(app, db)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(chatbot_bp, url_prefix='/api/chat')
    app.register_blueprint(travel_bp, url_prefix='/api/travel')
    app.register_blueprint(nlp_bp, url_prefix='/api/nlp')
    app.register_blueprint(destinations_bp, url_prefix='/api/destinations')
    
    # Initialize database if it doesn't exist
    with app.app_context():
        if not os.path.exists('travel_assistant.db'):
            db.create_all()
            init_sample_data()
            print('Initialized the database.')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500

    # Health check endpoint
    @app.route('/api/health')
    def health():
        return {'status': 'healthy'}, 200

    # Root endpoint
    @app.route('/')
    def root():
        return {
            'message': 'Welcome to the Travel Assistant API',
            'version': '1.0',
            'endpoints': {
                'auth': '/api/auth',
                'destinations': '/api/destinations',
                'travel': '/api/travel',
                'chat': '/api/chat',
                'nlp': '/api/nlp',
                'health': '/api/health'
            }
        }
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=8000)