# This file makes the routes directory a Python package
from api.routes.auth import auth_bp
from api.routes.accommodation import accommodation_bp
from api.routes.profile import profile_bp
from api.routes.reservations import reservations_bp

__all__ = ['auth_bp', 'accommodation_bp', 'profile_bp', 'reservations_bp']

def register_routes(app):
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(accommodation_bp, url_prefix='/api/accommodations')
    app.register_blueprint(profile_bp, url_prefix='/api/profile')
    app.register_blueprint(reservations_bp, url_prefix='/api/reservations')
