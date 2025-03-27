from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Import models after db initialization
from .user import User
from .destination import Destination
from .trip import Trip
from .review import Review
from .trip_plan import TripPlan

__all__ = ['db', 'User', 'Destination', 'Trip', 'Review', 'TripPlan']
