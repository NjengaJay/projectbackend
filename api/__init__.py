from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from datetime import timedelta
from .models import db
from .routes.auth import auth_bp
from .routes.chat import chat_bp
from .routes.accommodation import accommodation_bp
from .routes.favorites import favorites_bp

def create_app():
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Configure Flask-JWT-Extended
    app.config['JWT_SECRET_KEY'] = 'dev-secret-key'  # Change in production
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    jwt = JWTManager(app)

    # Configure SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///travel_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(accommodation_bp, url_prefix='/api')  # Keep this as /api since routes already include /accommodations
    app.register_blueprint(favorites_bp, url_prefix='/api/favorites')
    
    return app
