from sqlalchemy import Column, Integer, String, Date, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db import Base  # 確保 Base 有定義

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=True, default=func.now())

    profile = relationship(
        "UserProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )

class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id = Column(Integer,
                        ForeignKey("users.user_id", ondelete="CASCADE", onupdate="CASCADE"),
                        primary_key=True)
    full_name = Column(String(100))
    gender = Column(String(50), nullable=True)
    #gender = Column(Enum("male", "female", "other", "prefer_not_to_say", name="gender_enum"))
    birth_date = Column(Date)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="profile", uselist=False)
