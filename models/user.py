from datetime import datetime
from . import db
from werkzeug.security import generate_password_hash, check_password_hash
import json

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    _preferences = db.Column('preferences', db.Text, default='{}')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    trips = db.relationship('Trip', backref='user', lazy=True)
    reviews = db.relationship('Review', backref='user', lazy=True)
    
    def __init__(self, username, email, password, preferences=None):
        self.username = username
        self.email = email
        self.set_password(password)
        self.preferences = preferences or {}
    
    @property
    def preferences(self):
        return json.loads(self._preferences)
    
    @preferences.setter
    def preferences(self, value):
        self._preferences = json.dumps(value)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'preferences': self.preferences,
            'created_at': self.created_at.isoformat()
        }
