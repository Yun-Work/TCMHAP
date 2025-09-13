# app/services/user_service.py
import re
import random
from datetime import datetime, timedelta,date

from sqlalchemy import func
from sqlalchemy.orm import Session
from werkzeug.security import generate_password_hash

from app.models.users_model import User
from app.models.verifycode_model import VerifyCode
from app.utils.mail_util import send_email


def generate_code(length=6) -> str:
    return ''.join(random.choices('0123456789', k=length))

def get_user_by_email(session: Session, email: str):
    return session.query(User).filter_by(email=email).first()

def delete_active_codes(session: Session, *, user_id=None, email=None) -> None:
    if (user_id is None) == (email is None):
        raise ValueError("須指定 user_id 或 email 其中一個")
    q = session.query(VerifyCode).filter(VerifyCode.expires_at > func.utc_timestamp())
    q = q.filter(VerifyCode.user_id == user_id) if user_id is not None else q.filter(VerifyCode.email == email)
    q.delete(synchronize_session=False)

def send_verification_code(session: Session, *, status: str, email: str = None, user_id: int = None) -> dict:
    try:
        if status == "1":  # 註冊：用 email
            if not email:
                return {"success": False, "message": "缺少 email"}
            if get_user_by_email(session, email):
                return {"success": False, "message": "Email 已註冊，無法重複註冊"}

            delete_active_codes(session, email=email)
            code = generate_code()
            rec = VerifyCode(email=email, code=code,
                             expires_at=datetime.utcnow() + timedelta(minutes=10))
            session.add(rec); session.commit()
            send_email(email, code)
            return {"success": True, "message": "驗證碼已寄出"}

        elif status == "2":  # 忘記密碼：必須已註冊
            if not (email or user_id):
                return {"success": False, "message": "缺少 email 或 user_id"}

            if user_id is None:
                user = get_user_by_email(session, email or "")
                if not user:
                    return {"success": False, "message": "Email 尚未註冊"}
                user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
                email = user.email  # 用於寄信

            delete_active_codes(session, user_id=user_id)
            code = generate_code()
            rec = VerifyCode(user_id=user_id, code=code,
                             expires_at=datetime.utcnow() + timedelta(minutes=10))
            session.add(rec); session.commit()

            if email:
                send_email(email, code)
            return {"success": True, "message": "驗證碼已寄出", "user_id": user_id}

        else:
            return {"success": False, "message": "未知的驗證狀態"}

    except Exception as e:
        session.rollback()
        return {"success": False, "message": f"發送 Email 失敗：{e}"}

# 註冊：email + code
def verify_code_by_email(session: Session, email: str, code: str) -> bool:
    rec = (session.query(VerifyCode)
           .filter(VerifyCode.email == email,
                   VerifyCode.code == code,
                   VerifyCode.expires_at > func.utc_timestamp())
           .order_by(VerifyCode.vc_id.desc())
           .first())
    if not rec:
        return False
    rec.verified_at = func.utc_timestamp()
    session.commit()
    return True

# 忘記密碼：user_id + code
def verify_code_by_user(session: Session, user_id: int, code: str) -> bool:
    rec = (session.query(VerifyCode)
           .filter(VerifyCode.user_id == user_id,
                   VerifyCode.code == code,
                   VerifyCode.expires_at > func.utc_timestamp())
           .order_by(VerifyCode.vc_id.desc())
           .first())
    if not rec:
        return False
    rec.verified_at = func.utc_timestamp()
    session.commit()
    return True

def is_valid_password(pwd: str) -> bool:
    return bool(pwd and len(pwd) >= 6 and re.search(r'[A-Za-z]', pwd) and re.search(r'\d', pwd))

RESET_WINDOW_MINUTES = 10

def reset_password_with_code(session: Session, user_id: int, new_password: str) -> bool:
    if not is_valid_password(new_password):
        return False

    window_start = datetime.utcnow() - timedelta(minutes=RESET_WINDOW_MINUTES)

    rec = (session.query(VerifyCode)
           .filter(VerifyCode.user_id == user_id,
                   VerifyCode.verified_at.isnot(None),
                   VerifyCode.verified_at > window_start)
           .order_by(VerifyCode.vc_id.desc())
           .first())
    if not rec:
        return False

    user = session.get(User, user_id)
    if not user:
        return False

    user.password = generate_password_hash(new_password)

    session.query(VerifyCode).filter(VerifyCode.user_id == user_id).delete(synchronize_session=False)
    session.commit()
    return True


def _normalize_gender(g: str | None) -> str | None:
    if g is None:
        return None
    s = str(g).strip().lower()
    if s in ("m", "male", "男", "男生"): return "male"
    if s in ("f", "female", "女", "女生"): return "female"
    raise ValueError("性別僅允許 male/female（可傳 男/女 或 M/F）")

def _parse_birth_date(s: str | None) -> date | None:
    if not s: return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError("生日格式需為 YYYY-MM-DD 或 YYYY/MM/DD")

def get_user_profile(session: Session, user_id: int) -> dict | None:
    user = session.query(User).filter(User.user_id == user_id).first()
    if not user:
        return None
    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name or "",
        "gender": user.gender or "",  # 建議存/回 'male' / 'female'
        "birth_date": user.birth_date.strftime("%Y-%m-%d") if user.birth_date else ""
    }

def update_user_profile(
    session: Session,
    *,
    user_id: int,
    name: str | None = None,
    gender: str | None = None,
    birth_date_str: str | None = None
) -> dict:
    user = session.query(User).filter(User.user_id == user_id).first()
    if not user:
        return {"success": False, "code": "NOT_FOUND", "message": "找不到使用者"}

    try:
        if name is not None:
            name = name.strip()
            if not (2 <= len(name) <= 100):
                return {"success": False, "code": "BAD_REQUEST", "message": "使用者名稱需為 2~100 個字"}

        bd = None
        if birth_date_str is not None:
            bd = _parse_birth_date(birth_date_str)
            if bd > date.today():
                return {"success": False, "code": "BAD_REQUEST", "message": "生日不可晚於今天"}
            if bd < date(1900, 1, 1):
                return {"success": False, "code": "BAD_REQUEST", "message": "生日不可早於 1900-01-01"}

        gnorm = None
        if gender is not None:
            gnorm = _normalize_gender(gender)
    except ValueError as e:
        return {"success": False, "code": "BAD_REQUEST", "message": str(e)}

    try:
        if name is not None:
            user.name = name
        if gnorm is not None:
            user.gender = gnorm
        if birth_date_str is not None:
            user.birth_date = bd
        session.commit()
    except Exception as e:
        session.rollback()
        return {"success": False, "code": "SERVER_ERROR", "message": f"儲存失敗：{e}"}

    return {"success": True, "data": get_user_profile(session, int(user_id))}