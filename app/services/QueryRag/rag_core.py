import re
import uuid
from typing import List, Dict, Iterator

import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings

# -------- util --------
_whitespace_re = re.compile(r"\s+")

def normalize_text(t: str) -> str:
    return _whitespace_re.sub(" ", t).strip()


# -------- Chroma client --------
chroma_client = chromadb.PersistentClient(
    path=settings.DB_DIR,
    settings=ChromaSettings()
)
collection = chroma_client.get_or_create_collection(name="docs")


# -------- Embeddings --------
import re

_md_symbol_re = re.compile(r"[#*`>\-\d\.]+")  # 簡單去除 Markdown 標記
_whitespace_re = re.compile(r"\s+")

def clean_markdown(text: str) -> str:
    t = _md_symbol_re.sub(" ", text)
    return _whitespace_re.sub(" ", t).strip()


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    回傳每段文字的向量。
    - OpenAI：/v1/embeddings
    - Ollama：/api/embeddings
    """
    provider = (settings.LLM_PROVIDER or "ollama").lower()

    # 🔹 預先清理文字，避免 Ollama embeddings 回傳空
    cleaned_texts = [clean_markdown(t) for t in texts if t.strip()]
    if not cleaned_texts:
        raise RuntimeError("No valid text to embed after cleaning.")

    try:
        if provider == "openai":
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {"model": settings.EMBED_MODEL, "input": cleaned_texts}
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                return [item["embedding"] for item in data["data"]]

        # 默認走 Ollama
        base = settings.OLLAMA_BASE.rstrip("/")
        results: List[List[float]] = []
        async with httpx.AsyncClient(timeout=300.0) as client:
            for t in cleaned_texts:
                payload = {"model": settings.EMBED_MODEL, "prompt": t}
                r = await client.post(f"{base}/api/embeddings", json=payload)
                r.raise_for_status()
                data = r.json()
                emb = data.get("embedding") or (data.get("embeddings")[0] if "embeddings" in data else None)
                if not emb:
                    raise RuntimeError(f"Ollama returned empty embedding for text: {t[:50]}...")
                results.append(emb)
        return results

    except Exception as e:
        print(f"[EMBED ERROR] {e}")
        raise

# -------- Indexing --------
async def add_documents(chunks: List[Dict]):
    texts = [c["text"] for c in chunks]
    embeddings = await embed_texts(texts)
    ids = [c.get("id") or str(uuid.uuid4()) for c in chunks]
    metadatas = [{"source": c.get("source", "unknown"), "chunk": c.get("chunk", 0)} for c in chunks]
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)


# -------- Streaming Split --------
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


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    out: List[str] = []
    for i, piece in enumerate(split_text_iter(text, chunk_size, overlap)):
        if i >= 2000:
            break
        out.append(piece)
    return out


# -------- Retrieval --------
async def retrieve(query: str, top_k: int) -> List[Dict]:
    q_emb = (await embed_texts([query]))[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=max(1, top_k),
        include=["documents", "metadatas", "distances"],  # ✅ 移除 "ids"
    )
    hits: List[Dict] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i, doc in enumerate(docs):
        hits.append({
            "id": f"chunk_{i}",  # ✅ 自行產生 id
            "text": doc,
            "metadata": metas[i],
            "score": dists[i],
        })
    return hits


# -------- Generation --------
async def generate_answer(query: str, contexts: List[Dict]) -> str:
    provider = (settings.LLM_PROVIDER or "ollama").lower()
    system = (
        "You are a helpful assistant. Answer the user using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know."
    )
    context_block = "\n\n".join([f"[Chunk {i+1}] {c['text']}" for i, c in enumerate(contexts)])
    user = f"Question: {query}\n\nContext:\n{context_block}"

    if provider == "openai":
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

    base = settings.OLLAMA_BASE.rstrip("/")
    prompt = f"System: {system}\n\nUser: {user}\nAssistant:"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(f"{base}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()


async def rag_answer(query: str) -> Dict:
    hits = await retrieve(query, settings.TOP_K)
    answer = await generate_answer(query, hits)
    return {"answer": answer, "sources": hits}
