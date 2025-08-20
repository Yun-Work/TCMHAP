from flask import Flask
from app.routes.face_analysis_routes import face_analysis_bp
from app.routes.hologram_routes import hologram_bp
from app.routes.user_routes import user_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(hologram_bp, url_prefix='/api/holograms')
    app.register_blueprint(face_analysis_bp, url_prefix='/api/face')
    return app