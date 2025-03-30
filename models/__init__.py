from flask_sqlalchemy import SQLAlchemy
from api.models import db, User, Trip, Destination, Review, Accommodation, ChatMessage, ChatSession

__all__ = ['db', 'User', 'Trip', 'Destination', 'Review', 'Accommodation', 'ChatMessage', 'ChatSession']
