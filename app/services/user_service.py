# app/services/user_service.py
import re
import random
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session
from werkzeug.security import generate_password_hash

from app.models.users_model import User
from app.models.verifycode_model import VerifyCode
from app.utils.mail_util import send_email


# 產生驗證碼
def generate_code(length=6) -> str:
    return ''.join(random.choices('0123456789', k=length))

# 檢查 Email 是否註冊
def get_user_by_email(session: Session, email: str):
    return session.query(User).filter_by(email=email).first()

def delete_active_codes_of_user(session: Session, user_id: int) -> None:
    """刪除該使用者未過期的驗證碼，確保僅有一筆有效碼"""
    now = datetime.utcnow()
    session.query(VerifyCode).filter(
        VerifyCode.user_id == user_id,
        VerifyCode.expires_at > now
    ).delete(synchronize_session=False)

# 發送驗證碼
def send_verification_code(session: Session, email: str, status: str) -> dict:

    user = get_user_by_email(session, email)

    if status == "1":  # 註冊流程：不允許已註冊者
        if user:
            return {"success": False, "message": "Email 已註冊，無法重複註冊"}

    elif status == "2":  # 忘記密碼流程：必須已註冊
        if not user:
            return {"success": False, "message": "Email 尚未註冊，請先註冊"}

    # 取得 user_id
    user_id = user.user_id

    # 清除舊的有效碼（避免多碼並存）
    delete_active_codes_of_user(session, user_id)

    code = generate_code()
    expires = datetime.utcnow() + timedelta(minutes=10)

    record = VerifyCode(user_id=user_id, code=code, expires_at=expires)
    session.add(record)
    session.commit()

    try:
        send_email(email, code)
        return {"success": True, "user_id": user_id}
    except Exception as e:
        session.rollback()
        return {"success": False, "message": f"發送 Email 失敗：{e}"}



#驗證使用者輸入的驗證碼是否正確且未過期
def verify_code(session: Session, user_id: int, code: str) -> bool:
    record = (
        session.query(VerifyCode)
        .filter(
            VerifyCode.user_id == user_id,
            VerifyCode.code == code,
            VerifyCode.expires_at > func.utc_timestamp()
        )
        .first()
    )
    if record :
        record.verified_at = datetime.utcnow()  # 標記已驗證
        session.commit()
        return True
    return False

def is_valid_password(pwd: str) -> bool:
    if not pwd or len(pwd) < 6:
        return False
    has_letter = re.search(r'[A-Za-z]', pwd)   # 至少一個英文字母
    has_digit  = re.search(r'\d', pwd)         # 至少一個數字
    return bool(has_letter and has_digit)

RESET_WINDOW_MINUTES = 10
# 重設密碼
def reset_password_with_code(session: Session,user_id: int,  new_password: str) -> bool:
    window_start = datetime.utcnow() - timedelta(minutes=RESET_WINDOW_MINUTES)

    rec = (session.query(VerifyCode)
           .filter(
        VerifyCode.user_id == user_id,
        VerifyCode.verified_at.isnot(None),
        VerifyCode.verified_at > window_start
    )
           .first())

    if not rec:
        return False  # 未驗證或驗證已失效

    user = session.query(User).get(user_id)
    if not user:
        return False

    user.password = generate_password_hash(new_password)

    # 讓驗證碼失效（清掉這位使用者的所有驗證碼）
    session.query(VerifyCode).filter(VerifyCode.user_id == user_id).delete(synchronize_session=False)

    session.commit()
    return True
