
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()
class VerifyCode(Base):
    __tablename__ = 'verify_code'

    vc_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True)
    email = Column(String(255), nullable=True)
    code  = Column(String(6), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    verified_at = Column(DateTime, nullable=True, default=None)
