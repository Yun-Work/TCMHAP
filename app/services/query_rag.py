# app/services/query_rag.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import asyncio
import sys
import re
import argparse
import importlib.util

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_APP  = _ROOT / "app"
for p in (str(_ROOT), str(_APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

def _load_settings():
    for p in (_APP / "utils" / "config.py", _APP / "config.py"):
        if p.exists():
            spec = importlib.util.spec_from_file_location("tcmhap_config_qr", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return getattr(mod, "settings", getattr(mod, "Settings")())
    raise ImportError("找不到 settings，請確認 app/utils/config.py 或 app/config.py")
settings = _load_settings()

from app.services.rag_core import retrieve, generate_answer

def _to_single_paragraph(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

async def search(query: str, top_k: int | None = None) -> List[Dict]:
    top_k = top_k or getattr(settings, "TOP_K", 4)
    hits = await retrieve(query, top_k)
    results: List[Dict] = []
    for h in hits:
        md = h.get("metadata") or {}
        results.append({
            "score": float(h.get("score", 0.0)),
            "source": md.get("source"),
            "chunk": md.get("chunk"),
            "text": h.get("text"),
        })
    return results

async def answer_with_context(query: str, ctx_texts: List[str]) -> str:
    contexts = [{"text": t} for t in ctx_texts]
    # 保守用語 + 必須中文
    system_prompt = (
        "你是一位臉色觀察與臟腑對應的助理。請務必以繁體中文回答，"
        "病機與症狀敘述使用保守語氣（如：可能提示、或與…相關、可能伴隨…）。"
        "若無明確答案，請回覆『資料未明確指出』並提供最接近方向。"
    )
    # 直接把 system 包進 query
    ans = await generate_answer(system_prompt + "\n\n問題：" + query, contexts)
    return _to_single_paragraph(ans)

async def answer_multi(queries: List[str]) -> str:
    joined_questions = "；".join([f"問題{i+1}：{q}" for i, q in enumerate(queries)])
    context_block = ""  # 讓 generate_answer 內自行處理
    prompt = (
        "你是一位臉色觀察與臟腑對應的助理。請務必以繁體中文回答，避免英文；"
        "請在一個連貫段落中依序回答所有問題，不要換行、不要列點、不要標題；"
        "病機與症狀使用保守語氣（可能提示、或與…相關、可能伴隨…）；"
        "若無明確答案請用『資料未明確指出』。\n\n"
        f"需要回答的所有問題：{joined_questions}\n\n"
        "請輸出最終答案："
    )
    ans = await generate_answer("多題整合回答", [{"text": prompt}])
    return _to_single_paragraph(ans)

# -------- CLI --------
def run_sync(coro):
    return asyncio.run(coro)

def _run_single_query_flow(query: str, k: int | None):
    res = run_sync(search(query, k))
    if not res:
        print("沒有找到結果。")
        return
    texts = [r["text"] for r in res]
    ans = run_sync(answer_with_context(query, texts))
    print(ans)

def _run_multi_query_flow(queries: List[str]):
    if not queries:
        print("未提供任何問題。")
        return
    ans = run_sync(answer_multi(queries))
    print(ans)

def main():
    parser = argparse.ArgumentParser(description="RAG (MySQL) semantic search / multi-qa")
    parser.add_argument("query", type=str, nargs="*", help="單題模式：直接輸入問題；多題請用 --multi")
    parser.add_argument("--k", type=int, default=None, help="Top-K（預設 settings.TOP_K）")
    parser.add_argument("--multi", type=str, default=None, help="多題字串，用 '||' 分隔")
    args = parser.parse_args()

    if args.multi:
        queries = [q.strip() for q in args.multi.split("||") if q.strip()]
        _run_multi_query_flow(queries)
        return

    if args.query:
        q = " ".join(args.query).strip()
        _run_single_query_flow(q, args.k)
        return

    print("💬 互動模式（單題），輸入問題並按 Enter，Ctrl+C 離開")
    try:
        while True:
            q = input("\n> 問題：").strip()
            if not q:
                continue
            _run_single_query_flow(q, args.k)
    except KeyboardInterrupt:
        print("\n👋 Bye")

if __name__ == "__main__":
    main()
