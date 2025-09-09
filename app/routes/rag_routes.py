# app/routes/rag_routes.py
from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional

from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest

# --- 封裝好的服務層 ---
# 逐區域「一次性」建議（你已在 service 內部合併成一次 RAG 呼叫）
from app.services.region_advice_service import get_region_advice_all_in_one as advise_for_regions
# 歷史回顧自由聊天
from app.services.rag_service import chat_freeform

bp = Blueprint("rag", __name__)

# --- RAG Core 檢查（僅顯示用，不阻擋啟動） ---
try:
    from app.services.rag_core import retrieve, generate_answer  # noqa: F401
    _rag_ready = True
except Exception as _e:
    print(f"⚠️ RAG Core 不可用：{_e}")
    _rag_ready = False


# ---------------------------------------
# 小工具：統一錯誤/保底訊息
# ---------------------------------------
_DEF_EMPTY_HINT = "（知識庫未提供明確依據，請換個方式描述或縮小問題範圍）"

def _safe_answer(text: Optional[str]) -> str:
    t = (text or "").strip()
    return t if t else _DEF_EMPTY_HINT

def _json_err(msg: str, code: int = 400):
    return jsonify({"success": False, "error": msg}), code


@bp.route("/rag/health", methods=["GET"])
def rag_health():
    """簡單健康檢查（是否找得到 RAG 核心函式）"""
    return jsonify({"rag_core_ready": _rag_ready}), 200


# =========================================================
# ✅ 逐區域建議（臉部分析頁）
# =========================================================
@bp.route("/rag/regions", methods=["POST"])
def rag_regions():
    """
    請求 JSON：
    {
      "region_results": {"額頭(心)":"發紅","鼻頭(脾)":"發黑"},
      "overall_color": {"hex":"#AC5D44"},   // 可省略
      "has_moles": true,                     // 可省略
      "has_beard": false                     // 可省略
    }

    回應：
    {
      "diagnoses": { "額頭(心)": "三段式答案", ... },
      "diagnosis_text": "【區域+顏色 → 原因 → 症狀】各區塊以空行分隔",
      "advice_sources": [ ... ]
    }
    """
    data = request.get_json(silent=True) or {}
    region_results = data.get("region_results") or {}
    if not isinstance(region_results, dict) or not region_results:
        raise BadRequest("缺少欄位：region_results（非空物件）")

    if not _rag_ready:
        # 不中斷，回前端清楚訊息
        return jsonify({
            "diagnoses": {k: "（RAG 核心未啟動或不可用）" for k in region_results.keys()},
            "diagnosis_text": "（RAG 核心未啟動或不可用，無法生成建議）",
            "advice_sources": [],
            "note": "請檢查資料庫/模型設定或服務連線"
        }), 200

    t0 = time.time()
    print(f"[RAG REGIONS] ▶ items={len(region_results)}")

    try:
        combined, per_region, sources = advise_for_regions(
            region_results=region_results,
            overall_color=data.get("overall_color") or {},
            has_moles=bool(data.get("has_moles") or False),
            has_beard=bool(data.get("has_beard") or False),
        )
    except Exception as e:
        print(f"[RAG REGIONS] ✗ error: {e}")
        return jsonify({
            "diagnoses": {k: f"（此區域建議生成失敗：{e}）" for k in region_results.keys()},
            "diagnosis_text": _DEF_EMPTY_HINT,
            "advice_sources": []
        }), 200

    # 保底：每區域至少回一段提示；combined 也做保底
    per_region = {k: _safe_answer(v) for k, v in (per_region or {}).items()}
    combined = _safe_answer(combined)

    dt = time.time() - t0
    print(f"[RAG REGIONS] ✓ done in {dt:.2f}s, combined_len={len(combined)}, per_regions={len(per_region)}")

    return jsonify({
        "diagnoses": per_region,
        "diagnosis_text": combined,
        "advice_sources": sources or []
    }), 200


# =========================================================
# ✅ 歷史回顧自由對話（像 GPT）
# =========================================================
@bp.route("/rag/chat", methods=["POST"])
def rag_chat():
    """
    請求 JSON：
    {
      "question": "…",
      "context": "（選填）近七天病歷摘要字串",
      "system_hint": "（選填）覆蓋系統語氣"
    }
    回應：
    { "answer": "...", "advice_sources": [ ... ] }
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        raise BadRequest("缺少欄位：question")

    if not _rag_ready:
        return jsonify({"answer": "（RAG 核心未啟動或不可用，暫無法回覆）", "advice_sources": []}), 200

    context_text = data.get("context") or ""
    system_hint = data.get("system_hint") or ""

    t0 = time.time()
    print(f"[RAG CHAT] ▶ q='{question[:60]}{'...' if len(question)>60 else ''}' ctx_len={len(context_text)}")

    try:
        answer, sources = chat_freeform(
            question=question,
            context_text=context_text,
            system_hint=system_hint
        )
    except Exception as e:
        print(f"[RAG CHAT] ✗ error: {e}")
        return jsonify({"answer": f"（分析時發生錯誤：{e}）", "advice_sources": []}), 200

    answer = _safe_answer(answer)
    dt = time.time() - t0
    print(f"[RAG CHAT] ✓ done in {dt:.2f}s, answer_len={len(answer)}")

    return jsonify({"answer": answer, "advice_sources": sources or []}), 200


# =========================================================
# 🧩 相容端點：舊前端不改也能用
# =========================================================
@bp.route("/rag/ask", methods=["POST"])
def rag_ask_compat():
    """
    舊版 /rag/ask 相容：
    仍接受 organ / start_date / end_date，但改走自由聊天，
    並把條件拼進 context。
    回傳格式：{ "answer": "..." }
    """
    data = request.get_json(silent=True)
    if not data:
        raise BadRequest("請以 application/json 傳送資料")

    question   = (data.get("question") or "").strip()
    organ      = (data.get("organ") or "").strip()
    start_date = (data.get("start_date") or "").strip()
    end_date   = (data.get("end_date") or "").strip()

    if not question:
        raise BadRequest("缺少欄位：question")

    if not _rag_ready:
        return jsonify({"answer": "（RAG 核心未啟動或不可用，暫無法回覆）"}), 200

    conds = []
    if organ:
        conds.append(f"相關臟腑/區域：{organ}")
    if start_date or end_date:
        conds.append(f"日期範圍：{start_date or '未知'} ~ {end_date or '未知'}")
    ctx = ("；".join(conds)) if conds else ""

    t0 = time.time()
    print(f"[RAG ASK] ▶ q='{question[:60]}{'...' if len(question)>60 else ''}', organ='{organ}', range={start_date}~{end_date}")

    try:
        answer, _sources = chat_freeform(question=question, context_text=ctx)
    except Exception as e:
        print(f"[RAG ASK] ✗ error: {e}")
        return jsonify({"answer": f"分析時發生錯誤：{str(e)}"}), 200

    answer = _safe_answer(answer)
    dt = time.time() - t0
    print(f"[RAG ASK] ✓ done in {dt:.2f}s, answer_len={len(answer)}")

    return jsonify({"answer": answer}), 200


@bp.route("/rag/region-advice", methods=["POST"])
def rag_region_advice_compat():
    """
    相容單一區域問答：
    {
      "area_label": "額頭(心)",
      "status": "發紅",
      "overall_color": {"hex":"#f2dede"}, // 可省略
      "has_moles": false, "has_beard": false // 可省略
    }
    回應：{ "area_label": "...", "advice": "三段式答案", "advice_sources": [...] }
    """
    data = request.get_json(silent=True) or {}
    area_label = (data.get("area_label") or "").strip()
    status = (data.get("status") or "").strip()
    if not area_label or not status:
        raise BadRequest("缺少欄位：area_label 或 status")

    if not _rag_ready:
        return jsonify({
            "area_label": area_label,
            "advice": "（RAG 核心未啟動或不可用）",
            "advice_sources": []
        }), 200

    try:
        _, per_region, sources = advise_for_regions(
            region_results={area_label: status},
            overall_color=data.get("overall_color") or {},
            has_moles=bool(data.get("has_moles") or False),
            has_beard=bool(data.get("has_beard") or False),
        )
        advice = _safe_answer(per_region.get(area_label))
    except Exception as e:
        print(f"[RAG REGION ONE] ✗ error: {e}")
        advice, sources = f"（此區域建議生成失敗：{e}）", []

    return jsonify({"area_label": area_label, "advice": advice, "advice_sources": sources}), 200


@bp.route("/rag/region-advice/batch", methods=["POST"])
def rag_region_advice_batch_compat():
    """
    相容批次端點：
    {
      "items": [
        {"area_label":"額頭(心)","status":"發紅"},
        {"area_label":"鼻頭(脾)","status":"發黑"}
      ],
      "overall_color": {"hex":"#f2dede"}, // 可省略
      "has_moles": false, "has_beard": false // 可省略
    }
    回應：{ "results": [ {"area_label":"...","advice":"..."}, ... ], "advice_sources":[...] }
    """
    data = request.get_json(silent=True) or {}
    items = data.get("items") or []
    if not isinstance(items, list) or not items:
        raise BadRequest("缺少欄位：items（非空陣列）")

    if not _rag_ready:
        return jsonify({
            "results": [{"area_label": (it.get('area_label') or ''), "advice": "（RAG 核心未啟動或不可用）"} for it in items],
            "advice_sources": []
        }), 200

    common_overall = data.get("overall_color") or {}
    common_has_moles = bool(data.get("has_moles") or False)
    common_has_beard = bool(data.get("has_beard") or False)

    results: List[Dict] = []
    sources_all: List[Dict] = []

    for it in items:
        area_label = (it.get("area_label") or "").strip()
        status = (it.get("status") or "").strip()
        if not area_label or not status:
            results.append({"area_label": area_label or "(未提供)", "advice": "（缺少必要欄位）"})
            continue

        overall = it.get("overall_color") or common_overall
        has_moles = bool(it.get("has_moles") if "has_moles" in it else common_has_moles)
        has_beard = bool(it.get("has_beard") if "has_beard" in it else common_has_beard)

        try:
            _, per_region, srcs = advise_for_regions(
                region_results={area_label: status},
                overall_color=overall,
                has_moles=has_moles,
                has_beard=has_beard,
            )
            advice = _safe_answer(per_region.get(area_label))
            results.append({"area_label": area_label, "advice": advice})
            if srcs:
                sources_all.extend(srcs)
        except Exception as e:
            print(f"[RAG REGION BATCH] ✗ item '{area_label}' error: {e}")
            results.append({"area_label": area_label, "advice": f"（此區域建議生成失敗：{e}）"})

    return jsonify({"results": results, "advice_sources": sources_all}), 200