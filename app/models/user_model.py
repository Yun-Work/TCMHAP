from sqlalchemy import Column, Integer, String, Date, DateTime
from sqlalchemy.sql import func
from app.db import Base  # 確保 Base 有定義

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    gender = Column(String(10), nullable=True)
    password = Column(String(255), nullable=False)
    birth_date = Column(Date, nullable=True)
    created_at = Column(DateTime, nullable=True, default=func.now())
