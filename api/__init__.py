from flask import Flask
from flask_jwt_extended import JWTManager
from datetime import timedelta
from .models import db
from .routes.auth import auth_bp
from .routes.chat import chat_bp
from .routes.accommodation import accommodation_bp
from .routes.favorites import favorites_bp
from .routes.profile import profile_bp
from .routes.reservations import reservations_bp
from config import Config

def create_app():
    app = Flask(__name__)
    
    # Load configuration from Config class
    app.config.from_object(Config)
    
    # Initialize extensions
    jwt = JWTManager(app)
    db.init_app(app)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(accommodation_bp)  # URL prefix is already set in the blueprint
    app.register_blueprint(favorites_bp, url_prefix='/api/favorites')
    app.register_blueprint(profile_bp, url_prefix='/api/profile')
    app.register_blueprint(reservations_bp, url_prefix='/api/reservations')
    
    return app
