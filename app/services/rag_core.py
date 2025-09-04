# app/services/rag_core.py
from __future__ import annotations

"""
純 RAG 核心模組（資料表/嵌入/檢索/生成）。
⚠️ 不要在這裡 import 任何 Flask / Blueprint / 路由相關程式，避免循環匯入。
"""

__all__ = [
    "normalize_text",
    "clean_markdown",
    "split_text_iter",
    "embed_texts",
    "add_documents",
    "retrieve",
    "generate_answer",
]

import uuid
import re
from typing import List, Dict, Iterator

import numpy as np
import httpx
from sqlalchemy import Column, String, Integer, DateTime, func, Index
from sqlalchemy.orm import Session

# ✅ 你的專案的 DB 物件
from app.db import Base, engine, SessionLocal

# ------- settings 載入（帶預設值保底，避免 import 阻擋） -------
from pathlib import Path
import importlib.util

_APP = Path(__file__).resolve().parents[1]

def _load_settings():
    for p in (_APP / "utils" / "config.py", _APP / "config.py"):
        if p.exists():
            spec = importlib.util.spec_from_file_location("tcmhap_config_rc", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return getattr(mod, "settings", getattr(mod, "Settings")())
    raise ImportError("找不到 settings，請確認 app/utils/config.py 或 app/config.py")

try:
    settings = _load_settings()
except Exception as e:
    # 不讓整個模組初始化失敗：給可工作的預設值
    class _Default:
        LLM_PROVIDER = "ollama"
        OLLAMA_BASE = "http://127.0.0.1:11434"
        OLLAMA_MODEL = "llama3.1:8b"
        EMBED_MODEL = "mxbai-embed-large"
        OPENAI_API_KEY = None
        OPENAI_MODEL = "gpt-4o-mini"
    settings = _Default()
    print(f"⚠️ rag_core: settings 載入失敗，改用預設值：{e}")

# ------- ORM 定義 -------
# 在 MySQL 用 LONGTEXT/JSON；若你改用其他 DB，可自行切換型別
from sqlalchemy.dialects.mysql import LONGTEXT, JSON as MySQLJSON

class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String(1024), nullable=False)
    chunk_idx = Column(Integer, nullable=False, default=0)
    text = Column(LONGTEXT, nullable=False)       # 長文本
    embedding = Column(MySQLJSON, nullable=False) # 向量用 JSON 儲存
    created_at = Column(DateTime, server_default=func.now())

# 索引（避免 InnoDB 長度限制，對 source 做前綴）
Index("ix_rag_chunks_source", RagChunk.source, mysql_length=255)
Index("ix_rag_chunks_idx", RagChunk.chunk_idx)

# 建表：不要阻擋 import
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"⚠️ rag_core: 建表略過（稍後初始化亦可），原因：{e}")

# ------- 文本前處理/切塊 -------
_whitespace_re = re.compile(r"\s+")
_md_symbol_re = re.compile(r"[#*`>\-\d\.]+")

def normalize_text(t: str) -> str:
    return _whitespace_re.sub(" ", str(t)).strip()

def clean_markdown(text: str) -> str:
    t = _md_symbol_re.sub(" ", text or "")
    return _whitespace_re.sub(" ", t).strip()

def split_text_iter(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    if not text:
        return
    t = normalize_text(text)
    n = len(t)
    if chunk_size <= 0:
        yield t
        return
    overlap = max(0, min(overlap, max(0, chunk_size - 1)))
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        yield t[start:end]
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

# ------- Embeddings -------
async def embed_texts(texts: List[str]) -> List[List[float]]:
    provider = (getattr(settings, "LLM_PROVIDER", "ollama") or "ollama").lower()
    cleaned = [clean_markdown(t) for t in texts if t and str(t).strip()]
    if not cleaned:
        return []

    try:
        if provider == "openai":
            if not getattr(settings, "OPENAI_API_KEY", None):
                raise RuntimeError("缺少 OPENAI_API_KEY")
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {"model": settings.EMBED_MODEL, "input": cleaned}
            async with httpx.AsyncClient(timeout=300) as client:
                r = await client.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                return [d["embedding"] for d in data["data"]]

        # Ollama embeddings
        base = (getattr(settings, "OLLAMA_BASE", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").rstrip("/")
        results: List[List[float]] = []
        async with httpx.AsyncClient(timeout=300) as client:
            for t in cleaned:
                payload = {"model": settings.EMBED_MODEL, "prompt": t}
                r = await client.post(f"{base}/api/embeddings", json=payload)
                if r.status_code == 404:
                    raise RuntimeError(
                        f"Ollama embeddings 404（{base}/api/embeddings）。請先拉模型：`ollama pull {settings.EMBED_MODEL}`"
                    )
                r.raise_for_status()
                data = r.json()
                emb = data.get("embedding") or (data.get("embeddings", [None])[0])
                if not emb:
                    raise RuntimeError("Ollama 回傳空 embedding")
                results.append(emb)
        return results

    except Exception as e:
        raise RuntimeError(f"嵌入失敗：{e}") from e

# ------- 資料寫入/Session -------
def _session() -> Session:
    return SessionLocal()

def add_documents(chunks: List[Dict]) -> int:
    """chunks: [{'source','chunk','text','embedding'}]"""
    s = _session()
    inserted = 0
    try:
        for c in chunks:
            s.add(RagChunk(
                source=c["source"],
                chunk_idx=int(c["chunk"]),
                text=c["text"],
                embedding=c["embedding"],
            ))
            inserted += 1
        s.commit()
        return inserted
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

# ------- 檢索與生成 -------
async def retrieve(query: str, top_k: int = 4) -> List[Dict]:
    """相似度排序檢索，回傳 [{text, metadata, score}]"""
    q_embs = await embed_texts([query])
    if not q_embs:
        return []
    q = np.asarray(q_embs[0], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)

    s = _session()
    try:
        rows: List[RagChunk] = s.query(RagChunk).all()
        mat, texts, metas = [], [], []
        for r in rows:
            emb = r.embedding
            if not emb:
                continue
            v = np.asarray(emb, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            mat.append(v)
            texts.append(r.text)
            metas.append({"source": r.source, "chunk": r.chunk_idx})
    finally:
        s.close()

    if not mat:
        return []

    M = np.vstack(mat)
    sim = M @ q
    k = min(max(1, top_k), len(sim))
    idx = np.argpartition(-sim, k - 1)[:k]
    idx = idx[np.argsort(-sim[idx])]

    hits: List[Dict] = []
    for i in idx:
        hits.append({
            "text": texts[i],
            "metadata": metas[i],
            "score": float(sim[i]),
        })
    return hits

async def generate_answer(query: str, contexts: List[Dict]) -> str:
    """根據 contexts 生成回答（繁中）。"""
    provider = (getattr(settings, "LLM_PROVIDER", "ollama") or "ollama").lower()
    context_block = "\n\n".join([normalize_text(c.get("text", "")) for c in contexts if c.get("text")])

    system = (
        "你是一位臉色觀察與臟腑對應的助理。必須以繁體中文回答；"
        "若文件沒有直接答案，回覆「資料未明確指出」，並提供最接近的中醫方向。"
    )
    user = f"使用者問題：{query}\n\n（以下是文件內容，僅能依此作答）\n{context_block}"

    if provider == "openai":
        if not getattr(settings, "OPENAI_API_KEY", None):
            raise RuntimeError("缺少 OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

    # Ollama
    base = (getattr(settings, "OLLAMA_BASE", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").rstrip("/")
    prompt = f"System: {system}\n\nUser: {user}\nAssistant:"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{base}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
