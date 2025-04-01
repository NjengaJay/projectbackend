from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Import models after db is defined
from .user import User
from .profile import UserProfile, AccessibilityPreferences, Favorite
from .booking import Booking
from .trip import Trip, Destination, Review, Accommodation, ChatMessage, ChatSession

__all__ = ['db', 'User', 'Trip', 'Destination', 'Review', 'Accommodation', 
           'ChatMessage', 'ChatSession', 'UserProfile', 'AccessibilityPreferences', 
           'Favorite', 'Booking']
