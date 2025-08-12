import cv2
from flask import Blueprint, request, jsonify,send_file
from app.services.user_service import get_all_users, add_user
from app.services.hologram_service import main
from app.services.register_user_service import register_user
from app.services.login_service import login_user
from app.services.user_profile_service import upsert_user_profile

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

@user_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({'error': '所有欄位都必填'}), 400

    result = register_user(email, password)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result), 201
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

#輸入個人資料表
@user_bp.route('/update_profile', methods=['POST'])
def update_profile():
    data = request.get_json()
    user_id = data.get('user_id')
    full_name = data.get('full_name')
    gender = data.get('gender')
    birth_date = data.get('birth_date')

    result = upsert_user_profile(user_id, full_name, gender, birth_date)
    return jsonify(result)
