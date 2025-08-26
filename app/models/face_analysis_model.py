# app/models/face_analysis_model.py
from sqlalchemy import Column, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from sqlalchemy.orm import relationship
from app.db import Base  # 你專案裡已經有 Base

class FaceAnalysis(Base):
    __tablename__ = "face_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )
    organs = Column(MySQLJSON, nullable=True)         # JSON 陣列，例如 ["肺","肝"]
    normal_organs = Column(MySQLJSON, nullable=True)  # JSON 陣列，例如 ["心","脾"]
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="analyses", passive_deletes=True)