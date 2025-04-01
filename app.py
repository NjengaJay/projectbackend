import sys
import os
from datetime import timedelta
from flask import Flask, jsonify, g
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
import sqlite3

# Add the backend directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from api.models import db
from api.routes.auth import auth_bp
from api.routes.chat import chat_bp
from api.routes.accommodation import accommodation_bp
from api.routes.favorites import favorites_bp

def create_app():
    app = Flask(__name__)
    
    # Configure SQLite database with absolute path to backend/instance/travel_assistant.db
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'travel_assistant.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = 'dev-secret-key'  # Change in production
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    
    # Initialize extensions with proper CORS configuration
    CORS(app, 
         resources={
             r"/api/*": {
                 "origins": ["http://localhost:3000"],
                 "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                 "allow_headers": ["Content-Type", "Authorization"]
             }
         },
         supports_credentials=True)
    
    jwt = JWTManager(app)
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(accommodation_bp, url_prefix='/api')
    app.register_blueprint(favorites_bp, url_prefix='/api/favorites')
    
    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return jsonify({"status": "healthy"}), 200
        
    def get_db():
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
            db.execute("PRAGMA foreign_keys = ON")  # Enable foreign key support
        return db

    @app.before_request
    def before_request():
        g.db = get_db()
        
    return app

app = create_app()

if __name__ == '__main__':
    from config import Config
    app.run(debug=True, port=Config.API_PORT)