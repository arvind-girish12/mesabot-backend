from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    app.url_map.strict_slashes = False
    CORS(app, 
         resources={r"/*": {"origins": "*"}},
         supports_credentials=True,
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

    # Importing routes
    from app.routes.chat import chat_bp
    from app.routes.deadline import deadline_bp
    from app.routes.admin import admin_bp

    # Register Blueprints
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(deadline_bp, url_prefix="/deadlines")
    app.register_blueprint(admin_bp, url_prefix="/admin")

    return app
