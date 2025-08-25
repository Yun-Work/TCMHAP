# app/routes/export_routes.py
from flask import Blueprint, request, send_file, jsonify
from datetime import datetime
from app.services.export_service import build_symptom_history_excel

export_bp = Blueprint("export", __name__, url_prefix="/api/export")

def _parse_date(s: str | None):
    if not s:
        return None
    # 接受 YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d")

@export_bp.get("/ping")
def ping():
    return jsonify({"status": "ok"}), 200

@export_bp.get("/symptom-history")
def export_symptom_history():
    """
    下載 Excel（彩色月份×器官表）
    可選參數：
      - user_id: int
      - start: YYYY-MM-DD
      - end:   YYYY-MM-DD（含當天，後端會自動 +1 天做上限）
    """
    user_id = request.args.get("user_id", type=int)
    start = _parse_date(request.args.get("start"))
    end = _parse_date(request.args.get("end"))

    bio = build_symptom_history_excel(user_id=user_id, start=start, end=end)

    return send_file(
        bio,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="symptom_history.xlsx",
        max_age=0
    )