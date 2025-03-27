from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from dotenv import load_dotenv
from datetime import timedelta

# Load environment variables
load_dotenv()

# Initialize extensions
db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    
    # Configure Flask app
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change this in production
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')  # Required for sessions
    
    # Initialize extensions
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        # Import routes here to avoid circular imports
        from .routes import auth_bp, chatbot_bp, travel_bp, recommendation_bp, nlp_bp
        
        # Register blueprints
        app.register_blueprint(auth_bp, url_prefix='/api/auth')
        app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
        app.register_blueprint(travel_bp, url_prefix='/api/travel')
        app.register_blueprint(recommendation_bp, url_prefix='/api/recommendations')
        app.register_blueprint(nlp_bp, url_prefix='/api/nlp')
        
        # Create database tables
        db.create_all()

    # Handle OPTIONS requests
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    # Add root route
    @app.route('/')
    def root():
        return jsonify({
            "status": "success",
            "message": "RoamRover API Server",
            "version": "1.0.0",
            "endpoints": {
                "auth": "/api/auth",
                "destinations": "/api/destinations"
            }
        })

    # Add health check endpoint
    @app.route('/health')
    def health_check():
        try:
            # Try a simple database query to check connection
            db.session.execute('SELECT 1')
            db_status = "connected"
        except Exception as e:
            db_status = "disconnected"
        
        return jsonify({
            "status": "healthy",
            "database": db_status
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "status": "error",
            "message": "Resource not found",
            "error": str(error)
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error": str(error)
        }), 500
    
    return app
