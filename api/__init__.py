from flask import Flask
from flask_jwt_extended import JWTManager
from datetime import timedelta
from .models import db
from .routes.auth import auth_bp
from .routes.chat import chat_bp
from .routes.accommodation import accommodation_bp
from .routes.favorites import favorites_bp

def create_app():
    app = Flask(__name__)
    
    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/travel_assistant.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # JWT configuration
    app.config['JWT_SECRET_KEY'] = 'dev-secret-key'  # Change in production
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    
    # Initialize extensions
    jwt = JWTManager(app)
    db.init_app(app)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(accommodation_bp, url_prefix='/api')
    app.register_blueprint(favorites_bp, url_prefix='/api/favorites')
    
    return app
