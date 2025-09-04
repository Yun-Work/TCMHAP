from flask import Blueprint, request, jsonify
from datetime import datetime
from app.services.history_service import get_status_bar
from app.db import get_db_session  # 回傳 SQLAlchemy Session

history_dp = Blueprint('history_dp', __name__)  # 若有 url_prefix 請在註冊時加

@history_dp.route("/status-bar", methods=["GET", "POST"])
def status_bar():
    """
    單一路由：
    GET/POST /api/history/status-bar  (或你註冊的 url_prefix)
    Body/Query 需帶 { organ, start, end }，日期格式 YYYY-MM-DD
    回傳：
    { "categories": [...], "data": [...] }
    """
    payload = request.get_json(silent=True) or {}
    organ = request.args.get("organ") or payload.get("organ")
    start = request.args.get("start") or payload.get("start")
    end   = request.args.get("end")   or payload.get("end")

    if not organ or not start or not end:
        return jsonify({"error": "請提供 organ、start、end（YYYY-MM-DD）"}), 400

    try:
        start_d = datetime.strptime(start, "%Y-%m-%d").date()
        end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "日期格式錯誤，需 YYYY-MM-DD"}), 400

    session = get_db_session()
    try:
        res = get_status_bar(session, organ, start_d, end_d)
        if "error" in res:
            return jsonify(res), 404

        categories = res.get("categories", [])
        data = res.get("data", [])

         # 固定顯示順序
        desired_order = ["發紅", "發黑", "發黃", "發白", "發青"]
        counts = dict(zip(categories, data))
        categories, data = desired_order, [int(counts.get(k, 0)) for k in desired_order]

        return jsonify({"categories": categories, "data": data}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as ex:
        return jsonify({"error": f"Server error: {ex}"}), 500
    finally:
        session.close()
