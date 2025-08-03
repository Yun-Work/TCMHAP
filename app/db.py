# app/db.py
from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import MYSQL_CONFIG

DATABASE_URL = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}?charset=utf8mb4"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 測試區塊：嘗試建立連線
if __name__ == "__main__":
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ 資料庫連線成功！")
    except Exception as e:
        print("❌ 資料庫連線失敗：", e)
