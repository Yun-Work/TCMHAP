# app/config.py
import os
from flask.cli import load_dotenv
from pydantic_settings import BaseSettings

# ======================================
# 資料庫設定
# ======================================
MYSQL_CONFIG = {
    "host": "10.8.0.1",
    "user": "system",
    "password": "!QAZ2wsx#EDC",
    "database": "tcmha",
    "charset": "utf8mb4"
}

# load_dotenv()  # 讀取 .env 檔案

# ======================================
# 郵件設定 (Gmail SMTP)
# ======================================
GMAIL_SMTP_CONFIG = {
    "sender_email": "tkuim2025@gmail.com",
    "sender_password": "vcuf rchb jraj wgst",
}

# ======================================
# LLM / RAG 設定
# ======================================
class Settings(BaseSettings):
    LLM_PROVIDER: str = "ollama"

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Ollama 設定
    OLLAMA_BASE: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"   # ← 確認後替換成你本地有的

    EMBED_MODEL: str = "mxbai-embed-large"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 4

    DATA_DIR: str = "data"
    DB_DIR: str = "db"

    class Config:
        env_file = ".env"

settings = Settings()

