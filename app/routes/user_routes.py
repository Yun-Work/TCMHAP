import cv2
import json
import base64
from flask import Blueprint, request, jsonify, send_file
from app.services.user_service import get_all_users, add_user
from app.services.hologram_service import main

user_bp = Blueprint('user_bp', __name__)


@user_bp.route('/status', methods=['GET'])
def status():
    return "OK", 200


# 這是範本並沒有使用
@user_bp.route('/', methods=['GET'])
def list_users():
    return jsonify(get_all_users())


# 這是範本並沒有使用
@user_bp.route('/', methods=['POST'])
def create_user():
    data = request.json
    new_user = add_user(data['name'], data['email'])
    return jsonify(new_user), 201


# 臉部全息位置、清除障礙物的RestfulAPI(參考此程式)
@user_bp.route('/hologram', methods=['POST'])
def hologram():
    # 呼叫hologram_service的main
    result_img = main(request)
    # 以下是測試用
    cv2.imshow("標記結果", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return "OK", 200


# 如果你需要在這裡使用面部分析功能，可以這樣導入：
def get_face_analyzer():
    """延遲加載面部分析器，避免導入錯誤"""
    try:
        # skincolor_test.py 在 app/services/ 目錄下
        from app.services.skincolor_test import FaceSkinAnalyzer
        return FaceSkinAnalyzer()
    except ImportError as e:
        print(f"無法導入 FaceSkinAnalyzer: {e}")
        return None
    except Exception as e:
        print(f"創建 FaceSkinAnalyzer 實例時出錯: {e}")
        return None


# 示例：如果需要在用戶路由中使用面部分析
@user_bp.route('/face-test', methods=['POST'])
def face_test():
    """面部分析測試端點（示例）"""
    try:
        analyzer = get_face_analyzer()
        if analyzer is None:
            return jsonify({
                'success': False,
                'error': '面部分析器不可用'
            }), 503

        return jsonify({
            'success': True,
            'message': '面部分析器初始化成功',
            'analyzer_ready': analyzer.face_app is not None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'測試失敗: {str(e)}'
        }), 500