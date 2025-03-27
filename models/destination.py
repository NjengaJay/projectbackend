from . import db

class Destination(db.Model):
    __tablename__ = 'destinations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    image_url = db.Column(db.String(500))
    budget_range = db.Column(db.String(100))
    best_time_to_visit = db.Column(db.String(200))
    
    # Relationships
    trips = db.relationship('Trip', backref='destination', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'country': self.country,
            'description': self.description,
            'image_url': self.image_url,
            'budget_range': self.budget_range,
            'best_time_to_visit': self.best_time_to_visit
        }
