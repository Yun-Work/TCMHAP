# app/models/face_analysis_model.py
from sqlalchemy import Column, Integer, String, DateTime, func
from app.db import Base  # 你專案裡已經有 Base

class FaceAnalysis(Base):
    __tablename__ = "face_analysis"

    fa_id = Column(Integer, primary_key=True, autoincrement=True)

    # 臉部區域名稱（建議至少 32，容納「右上頰（胃區）」這種複合描述）
    face = Column(String(32), nullable=False, comment="臉部區域")

    # 臟腑描述（如「腎(生殖功能)」「心與肝交會」等，比較長，給 64 比較保險）
    organ = Column(String(64), nullable=False, comment="對應臟腑")

    # 狀態（發紅/發黑/發白/偏黃…），一般不會太長，16 已經很夠
    status = Column(String(16), nullable=False, comment="顏色或狀態")

    analysis_date = Column(DateTime, server_default=func.now(), nullable=False, comment="分析時間")

