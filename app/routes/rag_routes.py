# app/routes/rag_routes.py — 只處理 /rag/regions
from __future__ import annotations

from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest

# 只保留區域建議所需的服務
from app.services.region_advice_service import get_region_advice_all_in_one as advise_for_regions

bp = Blueprint("rag", __name__)

def _ok(payload: dict, code: int = 200):
    return jsonify({"success": True, **payload}), code

def _safe(text: str | None, fallback: str = "（目前沒有足夠依據可回答）") -> str:
    t = (text or "").strip()
    return t if t else fallback


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
      "success": true,
      "diagnoses": { "額頭(心)": "三段式答案", ... },
      "diagnosis_text": "彙整版（多段落）",
      "advice_sources": [ ... ]
    }
    """
    data = request.get_json(silent=True) or {}
    region_results = data.get("region_results") or {}
    if not isinstance(region_results, dict) or not region_results:
        raise BadRequest("缺少欄位：region_results（非空物件）")

    try:
        combined, per_region, sources = advise_for_regions(
            region_results=region_results,
            overall_color=data.get("overall_color") or {},
            has_moles=bool(data.get("has_moles") or False),
            has_beard=bool(data.get("has_beard") or False),
        )
        per_region = {k: _safe(v) for k, v in (per_region or {}).items()}
        return _ok({
            "diagnoses": per_region,
            "diagnosis_text": _safe(combined),
            "advice_sources": sources or []
        })
    except Exception as e:
        # 不中斷：每區域給保底提示
        return _ok({
            "diagnoses": {k: f"（此區域建議生成失敗：{e}）" for k in region_results.keys()},
            "diagnosis_text": "（目前沒有足夠依據可彙整）",
            "advice_sources": []
        })
