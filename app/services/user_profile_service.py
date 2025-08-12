# app/services/user_profile_service.py
from datetime import date
from sqlalchemy.exc import SQLAlchemyError
from app.db import SessionLocal
from app.models.user_model import User, UserProfile

#新增或更新使用者的個人資料
def upsert_user_profile(user_id, full_name=None, gender=None, birth_date=None):
    with SessionLocal() as session:
        try:
            user = session.get(User, user_id)
            if not user:
                return {"success": False, "message": "找不到使用者"}

            # 如果還沒有 profile，就先建立一筆
            if not user.profile:
                profile = UserProfile(user_id=user_id)
                session.add(profile)
            else:
                profile = user.profile

            if full_name is not None:
                profile.full_name = full_name

            if gender is not None:
                # 定義允許的性別映射（前端傳中文 → 存英文）
                gender_map = {
                    "男生": "male",
                    "女生": "female"
                }
                if gender not in gender_map:
                    return {
                        "success": False,
                        "message": "gender 不合法，必須是 '男生' 或 '女生'"
                    }
                profile.gender = gender_map[gender]

            if birth_date is not None:
                if isinstance(birth_date, str):
                    try:
                        birth_date = date.fromisoformat(birth_date)  # 'YYYY-MM-DD'
                    except ValueError:
                        return {"success": False, "message": "birth_date 格式需為 YYYY-MM-DD"}
                profile.birth_date = birth_date

            session.commit()
            return {"success": True, "message": "個人資料已更新"}
        except SQLAlchemyError as e:
            session.rollback()
            return {"success": False, "message": f"更新失敗：{str(e)}"}
