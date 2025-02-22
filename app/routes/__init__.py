from flask import Blueprint

# Import route blueprints
from .chat import chat_bp
from .deadline import deadline_bp
from .admin import admin_bp

# Create main Blueprint
main_bp = Blueprint("main", __name__)

# Register individual blueprints
main_bp.register_blueprint(chat_bp, url_prefix="/chat")
main_bp.register_blueprint(deadline_bp, url_prefix="/deadline")
main_bp.register_blueprint(admin_bp, url_prefix="/admin")
