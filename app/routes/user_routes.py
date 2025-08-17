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
        gender=data.get("gender"),  # "男生"/"女生" 或 "male"/"female"
        birth_date_val=data.get("birth_date")  # "YYYY-MM-DD"
    )
    return jsonify(res)
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


