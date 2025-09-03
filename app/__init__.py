from flask import Flask
from app.routes.face_analysis_routes import face_analysis_bp
from app.routes.history_route import history_dp
from app.routes.hologram_routes import hologram_bp
from app.routes.user_routes import user_bp
from app.routes.export_routes import export_bp
def create_app():
    app = Flask(__name__)
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(hologram_bp, url_prefix='/api/holograms')
    app.register_blueprint(face_analysis_bp, url_prefix='/api/face')
    app.register_blueprint(export_bp, url_prefix="/api/export")
    app.register_blueprint(history_dp, url_prefix="/api/history")
    return app