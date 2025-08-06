# app/routers/user_router.py

from flask import Blueprint, request, jsonify

from app.db import get_db_session  # 取得 SQLAlchemy session
from app.services.user_service import send_verification_code, verify_code


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
