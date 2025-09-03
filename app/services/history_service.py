# app/services/history_service.py
from datetime import datetime, date, time
from typing import Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import text

def _start_of_day(d: date) -> datetime:
    return datetime.combine(d, time.min)

def _end_of_day(d: date) -> datetime:
    return datetime.combine(d, time.max)

def get_status_bar(session: Session, organ_name: str, start: date, end: date) -> Dict:
    if end < start:
        raise ValueError("end 不能早於 start")

    # 1) 器官中文名 -> code_id（int），之後轉成字串去比對 face_analysis.organ (varchar)
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

    organ_id_str = str(organ_id)  # 這行是關鍵：用字串去比對 varchar 欄位

    # 2) 取狀態清單（X 軸順序以 sys_code 排序為準）
    status_rows = session.execute(
        text("""
            SELECT code_id, code_name
            FROM sys_code
            WHERE code_type='status'
            ORDER BY code_id
        """)
    ).mappings().all()

    # 3) 匯總各狀態次數（status 在 face_analysis 也是 varchar，要用字串處理）
    rows = session.execute(
        text("""
            SELECT fa.status AS s, COUNT(*) AS c
            FROM face_analysis fa
            WHERE fa.organ = :organ_id
              AND fa.analysis_date BETWEEN :start_dt AND :end_dt
            GROUP BY fa.status
        """),
        {
            "organ_id": organ_id_str,
            "start_dt": _start_of_day(start),
            "end_dt": _end_of_day(end),
        }
    ).mappings().all()

    # 用字串鍵（避免型別不一致）
    cnt_map = {str(r["s"]): int(r["c"]) for r in rows}

    categories: List[str] = [r["code_name"] for r in status_rows]
    data: List[int] = [cnt_map.get(str(r["code_id"]), 0) for r in status_rows]

    return {
        "organ": organ_name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "categories": categories,   # 例：["發紅","發黃","發青","發白","發黑"]
        "data": data                # 對應各分類的次數
    }
