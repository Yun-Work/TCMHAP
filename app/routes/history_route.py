from flask import Blueprint, request, jsonify
from datetime import datetime
from app.services.history_service import get_organ_color_series
from app.db import get_db_session

history_dp = Blueprint('history_dp', __name__)

@history_dp.route("/status-bar", methods=["GET", "POST"])
def status_bar():
    """
    入參：organ, start, end, user_id（YYYY-MM-DD），可選 mode=multi|dominant（預設 multi）
    回傳（multi）：{ organ, start, end, x: [...], series: {發紅:[...], 發黑:[...], ...} }
    回傳（dominant）另含：dominant: ["正常"|"發紅"|...]
    """
    payload = request.get_json(silent=True) or {}

    organ = request.args.get("organ") or payload.get("organ")
    start = request.args.get("start") or payload.get("start")
    end   = request.args.get("end")   or payload.get("end")
    mode  = (request.args.get("mode") or payload.get("mode") or "multi").lower()

    # 讀 user_id：優先 header，再 query，再 body
    uid_str = (
        request.headers.get("X-User-Id")
        or request.args.get("user_id")
        or payload.get("user_id")
    )

    if not organ or not start or not end or uid_str is None:
        return jsonify({"error": "請提供 organ、start、end、user_id（YYYY-MM-DD）"}), 400

    try:
        start_d = datetime.strptime(start, "%Y-%m-%d").date()
        end_d   = datetime.strptime(end, "%Y-%m-%d").date()
        uid     = int(uid_str)
    except ValueError:
        return jsonify({"error": "日期需 YYYY-MM-DD，且 user_id 必須為整數"}), 400

    if uid <= 0:
        return jsonify({"error": "user_id 不合法"}), 400
    if (end_d - start_d).days > 370:
        return jsonify({"error": "時間區間過長，請縮短至一年內"}), 400

    session = get_db_session()
    try:
        res = get_organ_color_series(session, organ, start_d, end_d, user_id=uid, mode=mode)
        if "error" in res:
            return jsonify(res), 404
        return jsonify(res), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as ex:
        return jsonify({"error": f"Server error: {ex}"}), 500
    finally:
        session.close()
