# app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import MYSQL_CONFIG

DATABASE_URL = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}?charset=utf8mb4"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# 建立並回傳一個新的資料庫 Session 實例
def get_db_session():
    return SessionLocal()