# app/services/register_user_service.py
import re
from datetime import date
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from werkzeug.security import generate_password_hash
from app.db import SessionLocal
from app.models.user_model import User

def is_valid_password(password: str) -> bool:
    return (
        isinstance(password, str)
        and len(password) >= 6
        and re.search(r'[A-Za-z]', password)
        and re.search(r'[0-9]', password)
    )

def register_user(email: str,
                               password: str,
                               name: str | None = None,
                               gender: str | None = None,
                               birth_date_val: str | date | None = None):

    #一次性註冊 + 個人資料寫入（users 表）

    if not email or not password:
        return {"success": False, "message": "缺少 email 或 password"}

    if not is_valid_password(password):
        return {"success": False, "message": "密碼格式錯誤：至少6碼，含英文與數字"}

    # 性別正規化（可不傳）
    if gender is not None:
        gmap = {"男生": "male", "女生": "female"}
        gender_norm = gmap.get(gender, gender)
        if gender_norm not in {"male", "female"}:
            return {"success": False, "message": "gender 不合法，需為 '男生' 或 '女生'"}
    else:
        gender_norm = None

    # 生日正規化（可不傳）
    bdate = None
    if birth_date_val is not None:
        if isinstance(birth_date_val, str):
            try:
                bdate = date.fromisoformat(birth_date_val)  # 'YYYY-MM-DD'
            except ValueError:
                return {"success": False, "message": "birth_date 格式需為 YYYY-MM-DD"}
        elif isinstance(birth_date_val, date):
            bdate = birth_date_val
        else:
            return {"success": False, "message": "birth_date 需為字串 YYYY-MM-DD 或 date 物件"}

    hashed_pw = generate_password_hash(password)

    with SessionLocal() as session:
        try:
            # Email 重複檢查（也可只靠 UNIQUE + IntegrityError）
            if session.query(User).filter_by(email=email).first():
                return {"success": False, "message": "Email 已被註冊"}

            user = User(
                email=email,
                password=hashed_pw,
                name=name,
                gender=gender_norm,
                birth_date=bdate,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return {"success": True, "message": "註冊成功", "user_id": user.user_id}
        except IntegrityError:
            session.rollback()
            return {"success": False, "message": "資料庫錯誤：Email 可能已存在"}
        except SQLAlchemyError as e:
            session.rollback()
            return {"success": False, "message": f"註冊失敗：{str(e)}"}
