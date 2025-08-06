from sqlalchemy import Column, Integer, String, DateTime,Date
from datetime import datetime

from sqlalchemy.orm import declarative_base

Base = declarative_base()
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    gender = Column(String(10), nullable=True)
    password = Column(String(255), nullable=False)
    birth_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=True)
    # updated_at = Column(DateTime, default=datetime.utcnow,
    #                     onupdate=datetime.utcnow, nullable=False)
