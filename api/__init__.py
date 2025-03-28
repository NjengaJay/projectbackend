from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
import os

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    
    # Configure app
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///travel_assistant.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Enable CORS
    CORS(app)
    
    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    
    # Import and register blueprints
    from .routes.auth import auth as auth_blueprint
    from .routes.trips import trips as trips_blueprint
    from .routes.destinations import destinations as destinations_blueprint
    
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(trips_blueprint)
    app.register_blueprint(destinations_blueprint)
    
    return app
