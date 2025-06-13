from flask import Blueprint, request, jsonify
from app.services.user_service import get_all_users, add_user

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/status', methods=['GET'])
def status():
    return 200

@user_bp.route('/', methods=['GET'])
def list_users():
    return jsonify(get_all_users())

@user_bp.route('/', methods=['POST'])
def create_user():
    data = request.json
    new_user = add_user(data['name'], data['email'])
    return jsonify(new_user), 201
