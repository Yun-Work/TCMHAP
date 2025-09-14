# app/routers/user_router.py
import cv2
from app.services.hologram_service import main
from app.services.login_service import login_user
from app.services.register_user_service import register_user

from flask import Blueprint, request, jsonify
from app.db import get_db_session  # 取得 SQLAlchemy session
from app.services.user_service import send_verification_code, verify_code_by_user, verify_code_by_email, \
    reset_password_with_code, is_valid_password, update_user_profile, get_user_profile
from werkzeug.security import generate_password_hash
from app.models.users_model import User

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/status', methods=['GET'])
def status():
    return "OK", 200

# 傳送驗證碼
@user_bp.route('/send_code', methods=['POST'])
def send_code():
    data = request.get_json(silent=True) or {}
    status  = (data.get("status") or "").strip()   # "1" 註冊，"2" 忘記密碼
    email   = (data.get("email") or "").strip()
    user_id = data.get("user_id")                  # 忘記密碼可直接帶 user_id（可選）

    if not status:
        return jsonify({"error": "缺少必要參數"}), 400

    session = get_db_session()
    try:
        if status == "1":
            # 註冊：一定要有 email
            if not email:
                return jsonify({"error": "缺少 email"}), 400
            result = send_verification_code(session, status="1", email=email)

        elif status == "2":
            # 忘記密碼：email 或 user_id 擇一（若只給 email，後端會反查 user_id）
            if not (email or user_id):
                return jsonify({"error": "缺少 email 或 user_id"}), 400
            result = send_verification_code(
                session,
                status="2",
                email=email if email else None,
                user_id=user_id
            )
        else:
            return jsonify({"error": "未知的驗證狀態"}), 400

        if result.get("success"):
            resp = {"message": result.get("message", "驗證碼已寄出")}
            if "user_id" in result:  # 只有忘記密碼流程才會帶回
                resp["user_id"] = result["user_id"]
            return jsonify(resp), 200
        else:
            return jsonify({"error": result.get("message", "驗證碼寄送失敗")}), 400
    finally:
        session.close()

# 驗證驗證碼
@user_bp.route('/verify_code', methods=['POST'])
def verify_code_api():
    data   = request.get_json(silent=True) or {}
    status = (data.get("status") or "").strip()
    code   = (data.get("code")   or "").strip()

    if not status or not code:
        return jsonify({"error": "缺少必要參數"}), 400

    session = get_db_session()
    try:
        if status == "1":
            # 註冊：email + code
            email = (data.get("email") or "").strip()
            if not email:
                return jsonify({"error": "缺少 email"}), 400
            ok = verify_code_by_email(session, email, code)

        elif status == "2":
            # 忘記密碼：user_id + code
            user_id = data.get("user_id")
            if not user_id:
                return jsonify({"error": "缺少 user_id"}), 400
            ok = verify_code_by_user(session, int(user_id), code)

        else:
            return jsonify({"error": "未知的驗證狀態"}), 400

        if ok:
            return jsonify({"message": "驗證成功"}), 200
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
# 登入 API
@user_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    res = login_user(
        email=data.get("email"),
        password=data.get("password")
    )

    # 成功 → 200
    if res.get("success") is True:
        return jsonify(res), 200

    # 失敗情境 → 比照註冊 API 的分類
    code = res.get("code", "")
    msg  = (res.get("message") or "").lower()

    if code in ("INVALID_EMAIL", "INVALID_PASSWORD_FORMAT"):
        return jsonify({"success": False, "code": code, "message": res.get("message")}), 200

    if code == "AUTH_FAILED" or "帳號或密碼" in msg:
        return jsonify({"success": False, "code": "AUTH_FAILED", "message": "帳號或密碼錯誤"}), 200

    # 其他未分類錯誤（資料庫或內部錯誤）
    return jsonify({"success": False, "code": "SERVER_ERROR", "message": res.get("message")}), 500

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

@user_bp.route('/change_password', methods=['POST'])
def change_password():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    new_password = data.get("new_password")
    # 基本檢查
    if not user_id or not new_password:
        return jsonify({"success": False, "message": "缺少必要參數"})
    if len(new_password) < 6:
        return jsonify({"success": False, "message": "密碼至少 6 碼"})
    session = get_db_session()
    try:
        # 查找使用者
        user = session.query(User).filter(User.user_id == user_id).first()
        if not user:
            return jsonify({"success": False, "message": "找不到使用者"})
        # 更新新密碼（加密後存入）
        user.password = generate_password_hash(new_password)
        session.commit()

        return jsonify({"success": True, "message": "密碼修改成功"}), 200
    except Exception as e:
        session.rollback()
        return jsonify({"success": False, "message": f"伺服器錯誤: {str(e)}"})
    finally:
        session.close()
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

# 取得個人資料
@user_bp.route('/get_profile', methods=['POST'])
def profile_get_post():
    data = request.get_json(silent=True) or {}
    uid = data.get("user_id")
    if not uid:
        return jsonify({"error": "缺少 user_id"}), 400

    session = get_db_session()
    try:
        profile = get_user_profile(session, int(uid))
        if profile is None:
            return jsonify({"error": "找不到使用者"}), 404

        return jsonify(profile), 200
    finally:
        session.close()


# 修改個人資料（僅更新有帶的欄位）
@user_bp.route('/update_profile', methods=['POST'])
def profile_update_post():
    payload = request.get_json(silent=True) or {}
    uid = payload.get("user_id")
    if not uid:
        return jsonify({"error": "缺少 user_id"}), 400

    session = get_db_session()
    try:
        res = update_user_profile(
            session,
            user_id=int(uid),
            name=payload.get("name"),
            gender=payload.get("gender"),       # "male"/"female" 或 "男"/"女" 皆可
            birth_date_str=payload.get("birth_date")  # "YYYY-MM-DD" 或 "YYYY/MM/DD"
        )

        if res.get("success"):
            return jsonify(res["data"]), 200

        code = res.get("code")
        msg  = res.get("message", "更新失敗")
        if code == "BAD_REQUEST":
            return jsonify({"error": msg}), 400
        if code == "NOT_FOUND":
            return jsonify({"error": msg}), 404
        return jsonify({"error": msg}), 500
    finally:
        session.close()
