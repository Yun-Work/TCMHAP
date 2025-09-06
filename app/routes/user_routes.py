import cv2
from app.services.hologram_service import main
from app.services.login_service import login_user
from app.services.register_user_service import register_user
# app/routers/user_router.py

from flask import Blueprint, request, jsonify
from app.db import get_db_session  # 取得 SQLAlchemy session
from app.services.user_service import send_verification_code, verify_code, reset_password_with_code

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/status', methods=['GET'])
def status():
    return "OK", 200

# 傳送驗證碼
@user_bp.route('/send_code', methods=['POST'])
def send_code():
    data = request.json
    email = data.get("email")
    status = data.get("status")  # "1" for register, "2" for forget password

    if not email or not status:
        return jsonify({"error": "缺少必要參數"}), 400

    session = get_db_session()
    try:
        if send_verification_code(session, email, status):
            return jsonify({"message": "驗證碼已寄出"}), 200
        else:
            return jsonify({"error": "驗證碼寄送失敗"}), 400
    finally:
        session.close()

# 驗證驗證碼
@user_bp.route('/verify_code', methods=['POST'])
def verify():
    data = request.json
    email = data.get("email")
    code = data.get("code")

    if not email or not code:
        return jsonify({"error": "缺少必要參數"}), 400

    session = get_db_session()
    try:
        if verify_code(session, email, code):
            return jsonify({"message": "驗證成功"}), 200
        else:
            return jsonify({"error": "驗證碼錯誤或已過期"}), 400
    finally:
        session.close()

# 重設密碼
@user_bp.route('/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    code = data.get("code")
    new_password = data.get("new_password")

    if not email or not code or not new_password:
        return jsonify({"error": "缺少必要參數"}), 400
    if len(new_password) < 6:
        return jsonify({"error": "密碼至少 6 碼"}), 400

    session = get_db_session()
    try:
        ok = reset_password_with_code(session, email, code, new_password)
        if ok:
            return jsonify({"message": "密碼已重設"}), 200
        else:
            return jsonify({"error": "驗證碼錯誤或已過期"}), 400
    finally:
        session.close()

#註冊API
@user_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    res = register_user(
        email=data.get("email"),
        password=data.get("password"),
        name=data.get("name"),
        gender=data.get("gender"),  # 前端傳「男生 / 女生」
        birth_date_val=data.get("birth_date")  # 前端傳 "YYYY/MM/DD"
    )
    # 依結果決定 HTTP 狀態碼
    if res.get("success") is True:
        # 成功：201 Created
        return jsonify(res), 201

    # 失敗情境 → 以 message / 內容判斷更精準的狀態碼
    msg = (res.get("message") or "").lower()
    # 建議你在 service 也加上一個 code，例如 "EMAIL_TAKEN"
    code = res.get("code")

    if code == "EMAIL_TAKEN" or "已被註冊" in msg or "已存在" in msg:
        return jsonify({"success": False, "code": "EMAIL_TAKEN", "message":"已被註冊"}), 200

    if "格式" in msg or "缺少" in msg or "不合法" in msg:
        return jsonify({"success": False, "code": "BAD_REQUEST", "message": "格式錯誤"}), 200

    # 其他未分類錯誤（資料庫或內部錯誤）
    return jsonify({"success": False, "code": "SERVER_ERROR", "message": res.get("message")}), 500
#登入API
@user_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': '請輸入帳號與密碼'}), 400

    result = login_user(email, password)
    return jsonify(result), 200 if result['success'] else 401

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