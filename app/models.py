# app/models.py
from sqlalchemy import Column, Integer, String
from app.db import Base  # 確保 Base 有定義

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(200), nullable=False)
