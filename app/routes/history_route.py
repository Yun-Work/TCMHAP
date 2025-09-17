from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from sqlalchemy import text  # ✅ 新增
from app.services.history_service import get_organ_color_series
from app.db import get_db_session

history_dp = Blueprint('history_dp', __name__)

# ✅ 新增：查「這位使用者、這段期間、這個臟腑」實際出現過的全息位置名稱
def _query_detected_locations(session, organ_name, start_date, end_date, user_id):
    """
    回傳 List[str]，內容為 sys_code(code_type='face').code_name，
    僅包含在該使用者該期間針對該臟腑實際出現過的 face。
    """
    # 讓時間上界是「隔天 00:00」(半開區間)，避免 datetime 比對的邊界誤差
    end_plus1 = end_date + timedelta(days=1)

    # 一次性 join 取得 face 名稱（避免先查 organ_id 再二次查）
    sql = text("""
        SELECT DISTINCT sc_face.code_name
        FROM face_analysis fa
        JOIN sys_code sc_organ
          ON sc_organ.code_type = 'organ'
         AND sc_organ.code_name = :organ_name
         AND fa.organ = sc_organ.code_id
        JOIN sys_code sc_face
          ON sc_face.code_type = 'face'
         AND sc_face.code_id = fa.face
        WHERE fa.user_id = :uid
          AND fa.analysis_date >= :start_dt
          AND fa.analysis_date <  :end_dt
          AND fa.face IS NOT NULL
    """)
    rows = session.execute(sql, {
        "organ_name": organ_name,
        "uid": user_id,
        "start_dt": start_date,
        "end_dt": end_plus1
    }).scalars().all()

    # 依 face 的 code_id 排序想更漂亮可再做一次排序（此處已 DISTINCT）
    return rows or []

@history_dp.route("/status-bar", methods=["GET", "POST"])
def status_bar():
    """
    入參：organ, start, end, user_id（YYYY-MM-DD），可選 mode=multi|dominant（預設 multi）
    回傳（multi）：{ organ, start, end, x: [...], series: {發紅:[...], 發黑:[...], ...},
                    locations_detected: ["顴骨內","鼻樑中段",...] }
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

        # ✅ 新增：把「本區間實際偵測到的全息位置」一起回傳
        res["locations_detected"] = _query_detected_locations(session, organ, start_d, end_d, uid)

        return jsonify(res), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as ex:
        return jsonify({"error": f"Server error: {ex}"}), 500
    finally:
        session.close()
