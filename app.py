from flask import Flask, jsonify, g, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from datetime import timedelta
from api.models import db
from api.routes import register_routes
from config import Config
import logging

def create_app():
    app = Flask(__name__)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)
    
    # Configure CORS globally
    CORS(app, 
         resources={
             r"/api/*": {
                 "origins": ["http://localhost:3000"],
                 "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                 "allow_headers": ["Content-Type", "Authorization"],
                 "supports_credentials": True,
                 "max_age": 3600
             }
         },
         expose_headers=["Content-Range", "X-Content-Range"])

    # Add CORS headers to all responses
    @app.after_request
    def after_request(response):
        if request.method == 'OPTIONS':
            response = app.make_default_options_response()
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            response.headers.add('Access-Control-Max-Age', '3600')
        app.logger.info('Response Headers: %s', dict(response.headers))
        return response
    
    # Database configuration - using absolute path
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    jwt = JWTManager(app)
    
    # Register all routes
    register_routes(app)
    
    @app.before_request
    def before_request():
        g.db = db.session
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=Config.API_PORT)