import re
from app.db import SessionLocal
from werkzeug.security import check_password_hash
from app.models.users_model import User


def is_valid_email(email):
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email)

def is_valid_password(password):
    return re.match(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,}$', password)

def login_user(email, password):
    # ✅ 格式驗證
    if not is_valid_email(email):
        return {
            'success': False,
            'message': 'Email 格式錯誤'
        }

    if not is_valid_password(password):
        return {
            'success': False,
            'message': '密碼需包含英文與數字，且至少6位'
        }
    # 開始查詢資料庫
    try:
        db = SessionLocal()
        user = db.query(User).filter_by(email=email).first()
        db.close()

        if user and check_password_hash(user.password, password):
            return {
                "success": True,
                "message": "登入成功",
                "user": {
                    "user_id": user.user_id,
                    "email": user.email,
                    "name": user.name,
                    "gender": user.gender,
                    "birth_date": user.birth_date.isoformat() if user.birth_date else None
                }
            }
        else:
            return {'success': False, 'message': '帳號或密碼錯誤'}

    except Exception as e:
        return {'success': False, 'message': f'資料庫錯誤：{str(e)}'}