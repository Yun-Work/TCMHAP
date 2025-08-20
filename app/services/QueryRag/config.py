from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 使用 Ollama
    LLM_PROVIDER: str = "ollama"  

    # OpenAI (如果你想切換用 OpenAI，就填 API key)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Ollama 本地端 API
    OLLAMA_BASE: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:4b"

    # Embedding 模型：Ollama 提供的 mxbai-embed-large
    EMBED_MODEL: str = "mxbai-embed-large"

    # RAG 設定
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 4

    # 路徑設定
    DATA_DIR: str = "data"
    DB_DIR: str = "db"

    class Config:
        env_file = ".env"

settings = Settings()
