# app/db.py
from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager
from typing import Generator

# 從 config.py 載入 MYSQL_CONFIG
from app.config import MYSQL_CONFIG

# 允許環境變數覆寫（例如 Docker 部署）
MYSQL_USER = os.getenv("MYSQL_USER", MYSQL_CONFIG["user"])
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", MYSQL_CONFIG["password"])
MYSQL_HOST = os.getenv("MYSQL_HOST", MYSQL_CONFIG["host"])
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", MYSQL_CONFIG["database"])

DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
    "?charset=utf8mb4"
)

# 建立 Engine（加上 pool_pre_ping 和 pool_recycle，避免斷線）
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,  # 秒數，避免 MySQL wait_timeout
    future=True,
)

# Session & Base
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()

# 建立並回傳一個新的資料庫 Session 實例
def get_db_session():
    return SessionLocal()

# 自動管理交易的 context manager
@contextmanager
def session_scope() -> Generator:
    """用法：
        with session_scope() as s:
            s.add(obj)
    自動 commit / rollback / close
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# 測試區塊：嘗試建立連線
if __name__ == "__main__":
    print(f"連線字串：{DATABASE_URL}")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ 資料庫連線成功！")
    except Exception as e:
        print("❌ 資料庫連線失敗：", e)
