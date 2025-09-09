# app/services/rag_service.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import asyncio

# 保留的核心：async 介面
from app.services.rag_core import retrieve, generate_answer


# ---------- 公用：執行一次 RAG（檢索→生成） ----------
async def _run_async(query: str, top_k: int = 6) -> Tuple[str, List[Dict]]:
    ctx = await retrieve(query, top_k=top_k)
    ans = await generate_answer(query=query, contexts=ctx)
    # 整理來源（可回前端）
    sources: List[Dict] = []
    for c in ctx or []:
        meta = c.get("metadata", {}) or {}
        sources.append({
            "source": meta.get("source"),
            "chunk": meta.get("chunk"),
            "score": c.get("score"),
        })
    return (ans or "").strip(), sources


def _extras_to_line(overall_color: Optional[Dict], has_moles: bool, has_beard: bool) -> str:
    hexv = (overall_color or {}).get("hex") if isinstance(overall_color, dict) else None
    return f"整體HEX={hexv or '未知'}；痣={'有' if has_moles else '無'}；鬍鬚={'有' if has_beard else '無'}"


# ---------- 1) 逐區域結構化建議 ----------
def _region_prompt(region: str, color: str,
                   overall_color: Optional[Dict] = None,
                   has_moles: bool = False,
                   has_beard: bool = False) -> str:
    """
    產生「辨識區域+異常顏色 → 為什麼造成 → 可能症狀」的限制式提示（繁體中文）
    僅根據 data/face_map.md（與你 ingest 後的知識）來回答，不做疾病診斷。
    """
    extras = _extras_to_line(overall_color, has_moles, has_beard)
    return f"""你是中醫知識助理，請以繁體中文回答，僅根據提供的文件內容作答（例如 face_map.md）。
對於觀察到的臉部區域與顏色，請用**三段式**輸出，每段獨立一行、避免醫療診斷用語：

1) 辨識：〈{region}〉－〈{color}〉
2) 可能原因：以作息、飲食、情緒、環境與臟腑對應的方向解釋為主（簡潔 1~2 句）
3) 可能伴隨症狀：列出常見的非疾病描述症狀（1 句內），若文件沒有依據請寫「資料未明確指出」

整體觀測：{extras}
若文件無明確依據，請誠實說明不要臆測。"""


def advise_for_regions(region_results: Dict[str, str],
                       overall_color: Optional[Dict] = None,
                       has_moles: bool = False,
                       has_beard: bool = False
                       ) -> Tuple[str, Dict[str, str], List[Dict]]:
    """
    給一個 {區域: 顏色} 的 map，依序問 RAG，回傳：
    - combined_text：把每個區域的三段式答案拼成一段
    - per_region    ：{區域: 三段式答案}（前端可逐項顯示）
    - sources_all   ：彙整的檢索來源
    """
    per_region: Dict[str, str] = {}
    sources_all: List[Dict] = []

    for region, color in (region_results or {}).items():
        prompt = _region_prompt(region, color, overall_color, has_moles, has_beard)
        ans, srcs = asyncio.run(_run_async(prompt, top_k=6))
        per_region[region] = ans or "（此區域未取得建議）"
        if srcs:
            sources_all.extend(srcs)

    combined = "\n\n".join([f"【{r}：{c}】\n{per_region[r]}" for r, c in region_results.items()]).strip()
    return combined, per_region, sources_all


# ---------- 2) 歷史回顧：自由對話 ----------
def chat_freeform(question: str,
                  context_text: str = "",
                  system_hint: str = ""
                  ) -> Tuple[str, List[Dict]]:
    """
    給使用者問題 + 可選的 7 天病歷摘要（字串），回傳像 GPT 一樣的對答。
    - system_hint 可加：語氣/風格/是否允許追問
    """
    system = system_hint or (
        "你是友善的中醫知識助理，請用繁體中文，先根據提供的背景資料回答，"
        "若資料不足要誠實說明並可提出 1 個簡短澄清問題。"
        "避免醫療診斷與病名，多用日常可行建議。"
    )
    full_q = (
        f"{system}\n\n"
        f"【可用背景（近七天病歷節錄，可能為空）】\n{context_text or '（無）'}\n\n"
        f"【使用者問題】\n{question}"
    )
    return asyncio.run(_run_async(full_q, top_k=8))
