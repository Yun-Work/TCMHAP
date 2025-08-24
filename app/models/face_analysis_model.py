# app/models/face_analysis_model.py
from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from app.db import Base  # 你專案裡已經有 Base

class FaceAnalysis(Base):
    __tablename__ = "face_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True)
    organs = Column(MySQLJSON, nullable=True)         # JSON 陣列，例如 ["肺","肝"]
    normal_organs = Column(MySQLJSON, nullable=True)  # JSON 陣列，例如 ["心","脾"]
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
