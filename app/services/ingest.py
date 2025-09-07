# app/services/ingest.py
from __future__ import annotations
from pathlib import Path
import sys, os, json, asyncio, importlib.util
from typing import List, Dict

# 路徑與 settings
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
APP  = ROOT / "app"
for p in (str(ROOT), str(APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

def _load_settings():
    for p in (APP / "utils" / "config.py", APP / "config.py"):
        if p.exists():
            spec = importlib.util.spec_from_file_location("tcmhap_config_ing", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return getattr(mod, "settings", getattr(mod, "Settings")())
    raise ImportError("找不到 settings，請確認 app/utils/config.py 或 app/config.py")
settings = _load_settings()

# 核心函式（MySQL）
from app.services.rag_core import split_text_iter, embed_texts, normalize_text, add_documents

SUPPORT_EXT = {".md", ".txt"}
BATCH = 36

# 解析資料夾（相對於 app/）
_data = Path(getattr(settings, "DATA_DIR", "data"))
DATA_DIR = _data if _data.is_absolute() else (APP / _data)

def _iter_files(dirpath: Path):
    for p in dirpath.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORT_EXT:
            yield p

def _read_text_auto(p: Path) -> str:
    for enc in ("utf-8-sig","utf-8","big5","cp950","gb18030","cp1252","latin-1"):
        try:
            s = p.read_text(encoding=enc, errors="ignore")
            if s:
                break
        except Exception:
            continue
    else:
        s = p.read_text(errors="ignore")
    return normalize_text(s)

async def _process_file(p: Path) -> int:
    text = _read_text_auto(p)
    if not text:
        return 0
    chunk_size = int(getattr(settings, "CHUNK_SIZE", 1000))
    overlap = int(getattr(settings, "CHUNK_OVERLAP", 100))

    chunks = list(split_text_iter(text, chunk_size, overlap))
    if not chunks:
        return 0

    total = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        embs = await embed_texts(batch)
        payload = []
        for j, emb in enumerate(embs):
            if not emb:
                continue
            payload.append({
                "source": str(p).replace("\\","/"),
                "chunk": i + j,
                "text": batch[j],
                "embedding": emb,
            })
        if payload:
            total += add_documents(payload)
    return total

async def main():
    if not DATA_DIR.exists():
        print(f"[WARN] Data dir not found: {DATA_DIR}")
        return
    paths = sorted(_iter_files(DATA_DIR))
    print(f"[INGEST] Found {len(paths)} file(s) under {DATA_DIR}.")

    total_chunks = 0
    for idx, p in enumerate(paths, 1):
        rel = p.relative_to(DATA_DIR) if str(p).startswith(str(DATA_DIR)) else p.name
        print(f"[INGEST] ({idx}/{len(paths)}) {rel}")
        try:
            n = await _process_file(p)
            print(f"[INGEST] {p.name}: appended {n} chunks.")
            total_chunks += n
        except Exception as e:
            print(f"[ERROR] {p} failed: {e}")
    print(f"[DONE] Ingested total {total_chunks} chunks into MySQL (table: rag_chunks).")

if __name__ == "__main__":
    asyncio.run(main())
