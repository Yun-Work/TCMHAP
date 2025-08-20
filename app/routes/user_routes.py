import cv2
from flask import Blueprint, request, jsonify,send_file
from app.services.user_service import get_all_users, add_user
from app.services.hologram_service import main
from app.services.register_user_service import register_user
from app.services.login_service import login_user

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/status', methods=['GET'])
def status():
    return "OK",200
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


