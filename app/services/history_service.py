# app/services/history_service.py
from datetime import datetime, date, time
from typing import Dict, List, Set, DefaultDict, Optional
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import text

def _start_of_day(d: date) -> datetime:
    return datetime.combine(d, time.min)

def _end_of_day(d: date) -> datetime:
    return datetime.combine(d, time.max)

ABNORMAL_ORDER = ["發紅", "發黑", "發黃", "發白", "發青"]
ALL_STATUSES   = ABNORMAL_ORDER + ["正常"]

def get_organ_color_series(session: Session,organ_name: str,start: date,end: date,user_id: int,                 # ← 新增：必帶 user_id
    mode: str = "multi") -> Dict:

    if end < start:
        raise ValueError("end 不能早於 start")
    if not isinstance(user_id, int) or user_id <= 0:
        return {"error": "缺少或不合法的 user_id"}

    # 1) 器官中文名 -> code_id
    organ_id = session.execute(
        text("""
            SELECT code_id
            FROM sys_code
            WHERE code_type='organ' AND code_name=:name
            LIMIT 1
        """),
        {"name": organ_name}
    ).scalar()
    if organ_id is None:
        return {"error": f"找不到器官：{organ_name}"}
    organ_id_str = str(organ_id)

    # 2) 取出 status 對照（一次查好）
    status_rows = session.execute(
        text("""
            SELECT code_id, code_name
            FROM sys_code
            WHERE code_type='status'
        """)
    ).mappings().all()
    id_to_name = {str(r["code_id"]): r["code_name"] for r in status_rows}

    # 3) 撈該使用者在區間內的 (日期, 狀態)
    rows = session.execute(
        text("""
            SELECT DATE(fa.analysis_date) AS d, fa.status AS s
            FROM face_analysis fa
            WHERE fa.user_id = :uid
              AND fa.organ   = :organ_id
              AND fa.analysis_date BETWEEN :start_dt AND :end_dt
              AND fa.status IS NOT NULL
            GROUP BY DATE(fa.analysis_date), fa.status
            ORDER BY d ASC
        """),
        {
            "uid": user_id,
            "organ_id": organ_id_str,
            "start_dt": _start_of_day(start),
            "end_dt": _end_of_day(end),
        }
    ).mappings().all()

    # 4) 整理每天有哪些顏色（狀態名）
    day_to_statuses: DefaultDict[date, Set[str]] = defaultdict(set)
    for r in rows:
        d = r["d"]
        s = str(r["s"])
        name = id_to_name.get(s)
        if name:
            day_to_statuses[d].add(name)

    # 5) 準備 X 軸日期
    days: List[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur = date.fromordinal(cur.toordinal() + 1)
    x = [d.strftime("%Y-%m-%d") for d in days]

    # 6A) 多資料列（每個顏色一條 0/1）
    series = {k: [] for k in ABNORMAL_ORDER}
    for d in days:
        statuses = day_to_statuses.get(d, set())
        for color in ABNORMAL_ORDER:
            series[color].append(1 if color in statuses else 0)

    result = {
        "organ": organ_name,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "x": x,
        "series": series,
    }

    # 6B) 若要主色
    if mode == "dominant":
        dominant: List[str] = []
        for d in days:
            statuses = day_to_statuses.get(d, set())
            picked = "正常"
            for c in ABNORMAL_ORDER:
                if c in statuses:
                    picked = c
                    break
            dominant.append(picked)
        result["dominant"] = dominant

    return result
