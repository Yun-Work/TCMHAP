from flask import Flask, jsonify

from app.routes.hologram_routes import hologram_bp
from app.routes.user_routes import user_bp
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(hologram_bp, url_prefix='/api/holograms')

    @app.get("/status")
    def status():
        return jsonify({"ok": True}), 200

    return app

