from flask import Flask
from flask_cors import CORS
import os


def create_app():
    app = Flask(__name__)
    CORS(app)

    # 匯入並註冊 Blueprint
    try:
        from app.routes.face_analysis_routes import face_analysis_bp
        app.register_blueprint(face_analysis_bp, url_prefix='/api/face')
        print("✅ face_analysis Blueprint 已註冊")
    except ImportError as e:
        print(f"❌ 匯入 face_analysis_routes 失敗: {e}")

    # 基本首頁
    @app.route('/')
    def index():
        return {
            "message": "最小可跑測試版 API",
            "version": "test-1.0",
            "endpoints": {
                "health": "/api/face/health",
                "debug_info": "/debug/info"
            }
        }

    # 除錯資訊：列出所有路由
    @app.route('/debug/info')
    def debug_info():
        return {
            "working_directory": os.getcwd(),
            "all_routes": [
                {
                    "rule": str(rule),
                    "methods": list(rule.methods)
                }
                for rule in app.url_map.iter_rules()
            ]
        }

    # 📋 啟動時列出所有路由
    print("\n📋 已註冊的路由:")
    for rule in app.url_map.iter_rules():
        methods = ",".join(sorted(rule.methods))
        print(f"   {methods:20s} {rule}")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=6060, debug=True)
