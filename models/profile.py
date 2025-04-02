from datetime import datetime
from . import db

class UserProfile(db.Model):
    __tablename__ = 'user_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    profile_picture = db.Column(db.LargeBinary)  # Store image data directly in DB
    profile_picture_type = db.Column(db.String(32))  # Store mime type
    bio = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    accessibility_preferences = db.relationship('AccessibilityPreferences', backref='profile', uselist=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'has_profile_picture': self.profile_picture is not None,
            'profile_picture_type': self.profile_picture_type,
            'bio': self.bio,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class AccessibilityPreferences(db.Model):
    __tablename__ = 'accessibility_preferences'
    
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey('user_profiles.id'), nullable=False, unique=True)
    wheelchair_access = db.Column(db.Boolean, default=False)
    screen_reader = db.Column(db.Boolean, default=False)
    high_contrast = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'wheelchair_access': self.wheelchair_access,
            'screen_reader': self.screen_reader,
            'high_contrast': self.high_contrast,
            'updated_at': self.updated_at.isoformat()
        }

class Favorite(db.Model):
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    accommodation_id = db.Column(db.Integer, db.ForeignKey('destinations.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'accommodation_id', name='unique_user_favorite'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'accommodation_id': self.accommodation_id,
            'created_at': self.created_at.isoformat()
        }
