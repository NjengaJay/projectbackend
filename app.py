import sys
import os
from datetime import timedelta
from flask import Flask, jsonify, g
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from api.models import db
from api.routes.auth import auth_bp
from api.routes.chat import chat_bp
from api.routes.accommodation import accommodation_bp
from api.routes.favorites import favorites_bp
from api.chatbot_handler import ChatbotHandler
from api.travel_assistant import TravelAssistant

def create_app():
    app = Flask(__name__)
    
    # Configure CORS with credentials support
    CORS(app, 
         resources={r"/api/*": {
             "origins": ["http://localhost:3000"],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True,
             "expose_headers": ["Authorization"]
         }})
    
    # Database configuration - using absolute path
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'instance', 'travel_assistant.db'))
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # JWT configuration
    app.config['JWT_SECRET_KEY'] = 'dev-secret-key'  # Change in production
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['JWT_TOKEN_LOCATION'] = ['headers', 'cookies']
    app.config['JWT_COOKIE_CSRF_PROTECT'] = False
    app.config['JWT_COOKIE_SECURE'] = False  # Set to True in production
    
    # Initialize extensions
    jwt = JWTManager(app)
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Initialize chatbot handler
    chatbot_handler = ChatbotHandler()
    travel_assistant = TravelAssistant(chatbot_handler.recommender)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(accommodation_bp, url_prefix='/api')
    app.register_blueprint(favorites_bp, url_prefix='/api/favorites')
    
    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return jsonify({"status": "healthy"}), 200
        
    @app.before_request
    def before_request():
        g.db = db.session
        
    return app

app = create_app()

if __name__ == '__main__':
    from config import Config
    app.run(debug=True, port=Config.API_PORT)