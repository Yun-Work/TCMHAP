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