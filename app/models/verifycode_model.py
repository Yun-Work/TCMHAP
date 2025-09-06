
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()
class VerifyCode(Base):
    __tablename__ = 'VerifyCode'

    vc_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    code  = Column(String(6),   nullable=False)
    expires_at = Column(DateTime,     nullable=False)

