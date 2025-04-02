import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

# Get the absolute path to the instance directory
BASEDIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASEDIR, 'instance')

class Config:
    # Database
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(INSTANCE_DIR, "travel_assistant.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'connect_args': {
            'check_same_thread': False
        }
    }
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_TOKEN_LOCATION = ['headers', 'cookies']
    JWT_COOKIE_CSRF_PROTECT = False
    JWT_COOKIE_SECURE = False  # Set to True in production
    
    # API
    API_PORT = int(os.getenv('API_PORT', 8000))
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    
    # CORS
    CORS_ORIGINS = ['http://localhost:3000']
    CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    CORS_HEADERS = ['Content-Type', 'Authorization']
    CORS_SUPPORTS_CREDENTIALS = True
    CORS_MAX_AGE = 3600
