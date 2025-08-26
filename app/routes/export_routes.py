# app/routes/export_routes.py
from flask import Blueprint, request, send_file, jsonify
from datetime import datetime
from app.services.export_service import (
    build_symptom_history_excel,
    build_symptom_history_excel_base64,
)

export_bp = Blueprint("export", __name__)  # url_prefix 交由 create_app() 統一指定

def _parse_date(s: str | None):
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d")

@export_bp.post("/ping")
def ping():
    return jsonify({"status": "ok"}), 200

@export_bp.post("/symptom-history")
def export_symptom_history_file():
    """
    以檔案方式下載 Excel（Content-Type: .xlsx）
    Body(JSON) 可選：{ "user_id": 1, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "filename": "xxx.xlsx" }
    """
    data = request.get_json(silent=True) or {}
    try:
        user_id = data.get("user_id")
        start = _parse_date(data.get("start"))
        end   = _parse_date(data.get("end"))
    except ValueError:
        return jsonify({"error": "日期格式錯誤，請使用 YYYY-MM-DD"}), 400

    bio = build_symptom_history_excel(user_id=user_id, start=start, end=end)
    filename = data.get("filename") or "symptom_history.xlsx"

    return send_file(
        bio,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
        max_age=0
    )

@export_bp.post("/symptom-history-base64")
def export_symptom_history_base64_api():
    """
    回傳 Excel 的 base64 字串（包在 JSON 內）
    Body(JSON) 可選：{ "user_id": 1, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" }
    回傳：{ "file_base64": "UEsDBBQABgAIAAAAIQ..." }
    """
    data = request.get_json(silent=True) or {}
    try:
        user_id = data.get("user_id")
        start = _parse_date(data.get("start"))
        end   = _parse_date(data.get("end"))
    except ValueError:
        return jsonify({"error": "日期格式錯誤，請使用 YYYY-MM-DD"}), 400

    excel_b64 = build_symptom_history_excel_base64(user_id=user_id, start=start, end=end)
    return jsonify({"file_base64": excel_b64}), 200