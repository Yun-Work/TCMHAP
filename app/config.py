# app/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# ======================
# 資料庫設定（可用環境變數覆蓋）
# ======================
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "10.8.0.1"),
    "user": os.getenv("MYSQL_USER", "system"),
    "password": os.getenv("MYSQL_PASSWORD", "!QAZ2wsx#EDC"),
    "database": os.getenv("MYSQL_DB", "tcmha"),
    "charset": "utf8mb4",
}

# ======================
# 郵件設定 (Gmail SMTP) — 從環境變數讀
# ======================
GMAIL_SMTP_CONFIG = {
    "sender_email": os.getenv("GMAIL_SENDER_EMAIL"),
    "sender_password": os.getenv("GMAIL_SENDER_PASSWORD"),
}

# ======================
# LLM / RAG 設定（pydantic-settings v2）
# ======================
class Settings(BaseSettings):
    LLM_PROVIDER: str = "ollama"

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Ollama
    OLLAMA_BASE: str = "http://163.13.202.117:11434"
    OLLAMA_MODEL: str = "cwchang/llama-3-taiwan-8b-instruct"
    EMBED_MODEL: str = "mxbai-embed-large"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 4

    DATA_DIR: str = "data"
    DB_DIR: str = "db"


    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )

settings = Settings()
