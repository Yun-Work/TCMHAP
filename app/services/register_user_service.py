# app/services/register_user_service.py
import re
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash
from app.db import SessionLocal
from app.models.user_model import User

def is_valid_password(password):
    return (
        len(password) >= 6 and
        re.search(r'[A-Za-z]', password) and
        re.search(r'[0-9]', password)
    )

def register_user(name, email, password):
    if not is_valid_password(password):
        return {"error": "密碼格式錯誤：至少6碼，含英文與數字"}

    session = SessionLocal()
    try:
        # 檢查 email 是否已存在
        existing_user = session.query(User).filter_by(email=email).first()
        if existing_user:
            return {"error": "Email 已被註冊"}

        hashed_pw = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_pw)
        session.add(new_user)
        session.commit()
        return {"success": "註冊成功", "user_id": new_user.user_id}
    except IntegrityError:
        session.rollback()
        return {"error": "資料庫錯誤：可能是 Email 重複"}
    finally:
        session.close()
