from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import validates
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.sql import func

# Create the SQLAlchemy instance
db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    preferences = db.Column(db.JSON)
    is_admin = db.Column(db.Boolean, default=False, index=True)
    
    # Relationships
    trips = db.relationship('Trip', back_populates='user', lazy=True, cascade='all, delete-orphan')
    reviews = db.relationship('Review', back_populates='user', lazy=True, cascade='all, delete-orphan')
    favorites = db.relationship('Favorite', back_populates='user', lazy=True, cascade='all, delete-orphan')
    profile = db.relationship('UserProfile', back_populates='user', uselist=False, cascade='all, delete-orphan')
    chat_messages = db.relationship('ChatMessage', back_populates='user', lazy=True, cascade='all, delete-orphan')
    
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

# Association table for many-to-many relationship between trips and destinations
trip_destinations = db.Table('trip_destinations',
    db.Column('trip_id', db.Integer, db.ForeignKey('trips.id', ondelete='CASCADE'), primary_key=True),
    db.Column('destination_id', db.Integer, db.ForeignKey('destinations.id', ondelete='CASCADE'), primary_key=True)
)

class Trip(db.Model):
    __tablename__ = 'trips'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    name = db.Column(db.String(100), nullable=False, server_default='My Trip')
    description = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = db.Column(db.String(20))
    itinerary = db.Column(db.JSON)
    route_details = db.Column(db.JSON)
    accommodation_details = db.Column(db.JSON)
    total_cost = db.Column(db.Float)
    notes = db.Column(db.JSON)
    
    # Relationships
    user = db.relationship('User', back_populates='trips')
    destinations = db.relationship('Destination', secondary='trip_destinations', back_populates='trips')
    
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
    
    # Relationships
    trips = db.relationship('Trip', secondary='trip_destinations', back_populates='destinations')
    reviews = db.relationship('Review', back_populates='destination', lazy=True, cascade='all, delete-orphan')
    accommodation = db.relationship('Accommodation', back_populates='destination', uselist=False, cascade='all, delete-orphan')
    favorites = db.relationship('Favorite', back_populates='destination', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'country': self.country,
            'description': self.description,
            'image_url': self.image_url,
            'budget_range': self.budget_range,
            'best_time_to_visit': self.best_time_to_visit,
            'accessibility_features': self.accessibility_features,
            'activities': self.activities,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    destination_id = db.Column(db.Integer, db.ForeignKey('destinations.id', ondelete='CASCADE'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.Text)
    sentiment_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.CheckConstraint('rating >= 1 AND rating <= 5', name='check_valid_rating'),
    )
    
    # Relationships
    user = db.relationship('User', back_populates='reviews')
    destination = db.relationship('Destination', back_populates='reviews')
    
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
    price_range = db.Column(db.Text)  # Store as string, handle JSON in route
    room_types = db.Column(db.Text)  # Store as string, handle JSON in route
    amenities = db.Column(db.Text)  # Store as string, handle JSON in route
    star_rating = db.Column(db.Float)  # Hotel star rating (1-5)
    review_score = db.Column(db.Float)  # Average review score
    review_count = db.Column(db.Integer)  # Number of reviews
    booking_conditions = db.Column(db.Text)  # Store as string, handle JSON in route
    accessibility_features = db.Column(db.Text)  # Store as string, handle JSON in route
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Link to the original POI if it exists
    destination_id = db.Column(db.Integer, db.ForeignKey('destinations.id', ondelete='SET NULL'), nullable=True)
    
    # Relationships
    destination = db.relationship('Destination', back_populates='accommodation')
    favorites = db.relationship('Favorite', back_populates='accommodation', lazy=True)
    
    def to_dict(self):
        """Convert accommodation to dictionary with safe JSON handling"""
        from json import loads, JSONDecodeError
        
        def safe_json_loads(value, default=None):
            if not value:
                return default
            try:
                return loads(value) if isinstance(value, str) else value
            except (JSONDecodeError, TypeError):
                return default
                
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'country': 'Netherlands',  # Hardcode country since all accommodations are in Netherlands
            'city': self.city,
            'description': self.description,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'price_range': safe_json_loads(self.price_range, {}),
            'room_types': safe_json_loads(self.room_types, []),
            'amenities': safe_json_loads(self.amenities, []),
            'star_rating': self.star_rating,
            'review_score': self.review_score,
            'review_count': self.review_count,
            'booking_conditions': safe_json_loads(self.booking_conditions, []),
            'accessibility_features': safe_json_loads(self.accessibility_features, {}),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'destination_id': self.destination_id
        }

class UserProfile(db.Model):
    __tablename__ = 'user_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    profile_picture = db.Column(db.String(500))
    bio = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', back_populates='profile')
    accessibility_preferences = db.relationship('AccessibilityPreferences', back_populates='profile', uselist=False, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'profile_picture': self.profile_picture,
            'bio': self.bio,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class AccessibilityPreferences(db.Model):
    __tablename__ = 'accessibility_preferences'
    
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey('user_profiles.id', ondelete='CASCADE'), nullable=False)
    wheelchair_access = db.Column(db.Boolean)
    screen_reader = db.Column(db.Boolean)
    high_contrast = db.Column(db.Boolean)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profile = db.relationship('UserProfile', back_populates='accessibility_preferences')
    
    def to_dict(self):
        return {
            'id': self.id,
            'profile_id': self.profile_id,
            'wheelchair_access': self.wheelchair_access,
            'screen_reader': self.screen_reader,
            'high_contrast': self.high_contrast,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False, unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    context = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages = db.relationship('ChatMessage', back_populates='session', lazy=True, cascade='all, delete-orphan')

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_id = db.Column(db.String(100), db.ForeignKey('chat_sessions.session_id', ondelete='CASCADE'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    session = db.relationship('ChatSession', back_populates='messages')
    user = db.relationship('User', back_populates='chat_messages')
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'content': self.content,
            'role': self.role,
            'created_at': self.created_at.isoformat()
        }

class Favorite(db.Model):
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    destination_id = db.Column(db.Integer, db.ForeignKey('destinations.id', ondelete='CASCADE'), nullable=True)
    accommodation_id = db.Column(db.Integer, db.ForeignKey('accommodations.id', ondelete='CASCADE'), nullable=True)
    type = db.Column(db.String(20), nullable=False)  # 'destination' or 'accommodation'
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', back_populates='favorites')
    destination = db.relationship('Destination', back_populates='favorites')
    accommodation = db.relationship('Accommodation', back_populates='favorites')
    
    @validates('type')
    def validate_type(self, key, value):
        if value not in ['destination', 'accommodation']:
            raise ValueError("Type must be either 'destination' or 'accommodation'")
        return value
    
    def to_dict(self):
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'type': self.type,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        
        if self.type == 'destination':
            data['destination'] = self.destination.to_dict() if self.destination else None
            data['destination_id'] = self.destination_id
        else:
            data['accommodation'] = self.accommodation.to_dict() if self.accommodation else None
            data['accommodation_id'] = self.accommodation_id
            
        return data
    
    def __repr__(self):
        return f'<Favorite {self.id}: {self.type}>'
