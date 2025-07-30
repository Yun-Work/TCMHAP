import cv2
from flask import Blueprint, request, jsonify,send_file
from app.services.user_service import get_all_users, add_user
from app.services.hologram_service import main

user_bp = Blueprint('user_bp', __name__)

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
