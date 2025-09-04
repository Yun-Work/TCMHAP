# app/models/face_analysis_model.py
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from app.db import Base  # 你專案裡已經有 Base

class FaceAnalysis(Base):
    __tablename__ = "face_analysis"

    fa_id = Column(Integer, primary_key=True, autoincrement=True)
    face = Column(String(5), nullable=False)          # 臉部位置（可直接存「右上頰」「鼻根」等）
    organ = Column(String(5), nullable=False)         # 對應臟腑（如「肺」「心」「肝」…）
    status = Column(String(5), nullable=False)        # 發紅/發黑/正常…（<=5字即可）
    message = Column(Text, nullable=True)             # 本次建議或備註
    analysis_date = Column(DateTime, server_default=func.now(), nullable=False)
