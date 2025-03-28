from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.sql import func
from . import db

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    preferences = db.Column(db.JSON)
    is_admin = db.Column(db.Boolean, default=False, index=True)
    
    trips = db.relationship('Trip', backref='user', lazy=True, cascade='all, delete-orphan')
    reviews = db.relationship('Review', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.set_password(password)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'preferences': self.preferences,
            'is_admin': self.is_admin
        }

class Trip(db.Model):
    __tablename__ = 'trips'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    start_date = db.Column(db.DateTime, index=True)
    end_date = db.Column(db.DateTime, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = db.Column(db.String(20), default='planned')  # planned, active, completed, cancelled
    
    # Trip details stored as JSON
    itinerary = db.Column(db.JSON)  # List of destinations with dates and activities
    route_details = db.Column(db.JSON)  # Transportation details between destinations
    accommodation_details = db.Column(db.JSON)  # Accommodation bookings
    total_cost = db.Column(db.Float)  # Estimated total cost
    notes = db.Column(db.JSON)  # User notes and reminders
    
    # Relationships
    destinations = db.relationship('Destination', secondary='trip_destinations', 
                                 backref=db.backref('trips', lazy='dynamic'))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status,
            'itinerary': self.itinerary or [],
            'route_details': self.route_details or {},
            'accommodation_details': self.accommodation_details or {},
            'total_cost': self.total_cost,
            'notes': self.notes or {},
            'destinations': [dest.to_dict() for dest in self.destinations]
        }

# Association table for many-to-many relationship between trips and destinations
trip_destinations = db.Table('trip_destinations',
    db.Column('trip_id', db.Integer, db.ForeignKey('trips.id', ondelete='CASCADE'), primary_key=True),
    db.Column('destination_id', db.Integer, db.ForeignKey('destinations.id', ondelete='CASCADE'), primary_key=True),
    db.Column('order', db.Integer),  # Order of destinations in the trip
    db.Column('duration', db.Integer)  # Duration of stay in days
)

class Destination(db.Model):
    __tablename__ = 'destinations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    country = db.Column(db.String(100), nullable=False, index=True)
    description = db.Column(db.Text)
    image_url = db.Column(db.String(500))
    budget_range = db.Column(db.String(100), index=True)
    best_time_to_visit = db.Column(db.String(200))
    accessibility_features = db.Column(db.JSON)
    activities = db.Column(db.JSON)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    trips = db.relationship('Trip', secondary='trip_destinations', 
                            backref=db.backref('destinations', lazy='dynamic'))
    reviews = db.relationship('Review', backref='destination', lazy=True, cascade='all, delete-orphan')
    accommodation = db.relationship('Accommodation', backref='destination', uselist=False, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'country': self.country,
            'description': self.description,
            'image_url': self.image_url,
            'budget_range': self.budget_range,
            'best_time_to_visit': self.best_time_to_visit,
            'accessibility_features': self.accessibility_features or [],
            'activities': self.activities or [],
            'latitude': self.latitude,
            'longitude': self.longitude,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Accommodation(db.Model):
    __tablename__ = 'accommodations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    type = db.Column(db.String(50), index=True)  # hotel, hostel, apartment, etc.
    city = db.Column(db.String(100), index=True)
    description = db.Column(db.Text)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    
    # Additional fields for detailed information
    price_range = db.Column(db.JSON)  # {'min': float, 'max': float, 'currency': str}
    room_types = db.Column(db.JSON)  # List of available room types
    amenities = db.Column(db.JSON)  # List of amenities
    star_rating = db.Column(db.Float)  # Hotel star rating (1-5)
    review_score = db.Column(db.Float)  # Average review score
    review_count = db.Column(db.Integer)  # Number of reviews
    booking_conditions = db.Column(db.JSON)  # Booking policies and conditions
    
    accessibility_features = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Link to the original POI if it exists
    destination_id = db.Column(db.Integer, db.ForeignKey('destinations.id', ondelete='SET NULL'), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'city': self.city,
            'description': self.description,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'price_range': self.price_range,
            'room_types': self.room_types or [],
            'amenities': self.amenities or [],
            'star_rating': self.star_rating,
            'review_score': self.review_score,
            'review_count': self.review_count,
            'booking_conditions': self.booking_conditions or {},
            'accessibility_features': self.accessibility_features or [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'destination_id': self.destination_id
        }

class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    destination_id = db.Column(db.Integer, db.ForeignKey('destinations.id', ondelete='CASCADE'), nullable=False, index=True)
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.Text)
    sentiment_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.CheckConstraint('rating >= 1 AND rating <= 5', name='check_valid_rating'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'destination_id': self.destination_id,
            'rating': self.rating,
            'comment': self.comment,
            'sentiment_score': self.sentiment_score,
            'created_at': self.created_at.isoformat()
        }
