import cv2
from flask import Blueprint, request, jsonify,send_file
from app.services.user_service import get_all_users, add_user
from app.services.hologram_service import main

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
# 全息位置
@user_bp.route('/hologram', methods=['POST'])
def hologram():
    result_img = main(request)

    if result_img  is None:
        return 'No face detected', 422
    cv2.imshow("標記結果", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return "OK", 200
