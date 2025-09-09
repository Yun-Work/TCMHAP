# app/routes/history_routes.py
from __future__ import annotations

from flask import Blueprint, request, jsonify
from datetime import datetime, date
from typing import Tuple, Optional, Dict, List

from app.services.history_service import get_status_bar
from app.db import get_db_session  # 回傳 SQLAlchemy Session

# 建議在 app/__init__.py 用：app.register_blueprint(history_dp, url_prefix="/api/history")
history_dp = Blueprint("history_dp", __name__)

# ---- 顏色標籤正規化 ----
_DESIRED_ORDER: List[str] = ["發紅", "發黑", "發黃", "發白", "發綠"]  # 固定順序
# 同義詞/別名：歷史資料若用到舊稱，統一映射
_COLOR_ALIASES: Dict[str, str] = {
    "紅": "發紅",
    "黑": "發黑",
    "黃": "發黃",
    "白": "發白",
    "青": "發綠",
    "發青": "發綠",
    "綠": "發綠",
}

def _norm_color(label: str) -> Optional[str]:
    if not label:
        return None
    t = str(label).strip()
    t = _COLOR_ALIASES.get(t, t)
    return t if t in _DESIRED_ORDER else None


# ---- 參數解析/日期處理 ----
def _parse_params() -> Tuple[str, Optional[str], Optional[str]]:
    """
    同時支援 query string 與 JSON body，鍵名支援：
    - organ
    - start / end
    - start_date / end_date
    """
    payload = request.get_json(silent=True) or {}
    get = request.args.get

    organ = (get("organ") or payload.get("organ") or "").strip()

    start = get("start") or payload.get("start") or payload.get("start_date")
    end   = get("end")   or payload.get("end")   or payload.get("end_date")

    return organ, (start or None), (end or None)


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _normalize_dates(start: str, end: str) -> Tuple[Optional[date], Optional[date], Optional[str]]:
    try:
        d1, d2 = _parse_ymd(start), _parse_ymd(end)
        # 若使用者誤把順序顛倒 -> 自動交換
        if d1 > d2:
            d1, d2 = d2, d1
        return d1, d2, None
    except Exception:
        return None, None, "日期格式錯誤，需 YYYY-MM-DD"


# ---- 組裝回應 ----
def _build_response(session, organ: str, start_d: date, end_d: date):
    """
    呼叫 history_service 取得資料，並做：
    1) 顏色標籤正規化
    2) 依固定順序排列
    3) 即使資料為空也回傳 0 計數，避免前端崩潰
    """
    # 後端服務預期：空 organ 代表「全部」
    organ_for_query = None if (not organ or organ == "全部") else organ

    try:
        res = get_status_bar(session, organ_for_query, start_d, end_d) or {}
    except Exception as e:
        # 發生例外也不要讓前端炸掉，回零值
        res = {"error": f"{e}"}

    # 可能的格式一：{"categories":[...],"data":[...]}
    categories = list(res.get("categories") or [])
    data = list(res.get("data") or [])

    # 若服務回傳的是 dict 或其它格式，這裡可再擴充；先處理最常見型態
    counts_map: Dict[str, int] = {}

    if categories and data and len(categories) == len(data):
        for k, v in zip(categories, data):
            key = _norm_color(k)
            if not key:
                continue
            try:
                counts_map[key] = counts_map.get(key, 0) + int(v)
            except Exception:
                # 忽略不能轉 int 的值
                pass
    else:
        # 後端回傳空或含 error -> 全部 0
        counts_map = {}

    # 固定輸出順序與補 0
    out_data = [int(counts_map.get(k, 0)) for k in _DESIRED_ORDER]

    payload = {
        "success": True,
        "organ": organ or "全部",
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "categories": _DESIRED_ORDER,
        "data": out_data,
    }

    # 若服務有錯，額外附帶 error 字串（但仍 200，讓前端可顯示圖表為 0）
    if "error" in res:
        payload["note"] = f"後端來源回傳錯誤：{res.get('error')}（已以 0 填補）"

    return jsonify(payload), 200


# -------- 路由定義 --------
# 1) 原先的：/api/history/status-bar  (GET/POST 都支援)
@history_dp.route("/status-bar", methods=["GET", "POST"])
def status_bar():
    organ, start, end = _parse_params()
    if not organ or not start or not end:
        return jsonify({"success": False, "error": "請提供 organ、start/end（或 start_date/end_date）"}), 400

    start_d, end_d, err = _normalize_dates(start, end)
    if err:
        return jsonify({"success": False, "error": err}), 400

    session = get_db_session()
    try:
        return _build_response(session, organ, start_d, end_d)
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as ex:
        return jsonify({"success": False, "error": f"Server error: {ex}"}), 500
    finally:
        session.close()


# 2) 相容前端的：/api/history  (你的 Android 使用這個)
@history_dp.route("/", methods=["POST", "GET"])
def history_root():
    organ, start, end = _parse_params()
    if not organ or not start or not end:
        return jsonify({"success": False, "error": "請提供 organ、start/end（或 start_date/end_date）"}), 400

    start_d, end_d, err = _normalize_dates(start, end)
    if err:
        return jsonify({"success": False, "error": err}), 400

    session = get_db_session()
    try:
        return _build_response(session, organ, start_d, end_d)
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as ex:
        return jsonify({"success": False, "error": f"Server error: {ex}"}), 500
    finally:
        session.close()
