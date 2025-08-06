# app/services/user_profile_service.py
from app.db import SessionLocal
from app.models.user_model import User

def user_profile(user_id, gender, birth_date):
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(user_id=user_id).first()
        if not user:
            return {"success": False, "message": "找不到使用者"}

        user.gender = gender
        user.birth_date = birth_date
        session.commit()
        return {"success": True, "message": "個人資料已更新"}
    except Exception as e:
        session.rollback()
        return {"success": False, "message": f"更新失敗：{str(e)}"}
    finally:
        session.close()
