# app/services/user_service.py

import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.user_model import User
from app.models.verifycode_model import VerifyCode
from app.utils.mail_util import send_email


#產生驗證碼
def generate_code(length=6) -> str:
    return ''.join(random.choices('0123456789', k=length))

#檢查 Email 是否註冊
def is_email_registered(session: Session, email: str) -> bool:
    return session.query(User).filter_by(email=email).first() is not None

#發送驗證碼
def send_verification_code(session: Session, email: str, status: str) -> bool:
    if status == "1":  # 註冊流程：不允許已註冊者
        if is_email_registered(session, email):
            print("Email 已註冊，無法重複註冊")
            return False

    elif status == "2":  # 忘記密碼流程：必須已註冊
        if not is_email_registered(session, email):
            print("Email 尚未註冊，請先註冊")
            return False

    else:
        print("未知的驗證狀態")
        return False

    # 清除此 email 的舊驗證碼（無論過期與否都可刪）
    session.query(VerifyCode).filter(VerifyCode.email == email).delete()
    session.commit()

    code = generate_code()
    expires = datetime.utcnow() + timedelta(minutes=10)

    record = VerifyCode(email=email, code=code, expires_at=expires)
    session.add(record)
    session.commit()

    try:
        send_email(email, code)
        return True
    except Exception as e:
        print("發送 Email 失敗：", e)
        return False


#驗證使用者輸入的驗證碼是否正確且未過期
def verify_code(session: Session, email: str, code: str) -> bool:
    # 清除所有過期的驗證碼
    session.query(VerifyCode).filter(VerifyCode.expires_at < datetime.utcnow()).delete()
    session.commit()

    record = session.query(VerifyCode).filter_by(email=email, code=code).first()
    if record and record.expires_at > datetime.utcnow():
        session.delete(record)  # 通過後就刪除
        session.commit()
        return True
    return False
