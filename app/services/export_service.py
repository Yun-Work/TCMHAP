# app/services/export_service.py
import io
import pandas as pd
from sqlalchemy import text
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter
from datetime import datetime
from zoneinfo import ZoneInfo
from app.db import engine

SQL = text("""
SELECT
  DATE_FORMAT(fa.created_at, '%Y-%m-01') AS month,
  jt.organ AS organ,
  COUNT(*) AS symptom_count
FROM face_analysis fa
JOIN JSON_TABLE(
  CAST(fa.organs AS JSON), '$[*]' COLUMNS (organ VARCHAR(50) PATH '$')
) jt
WHERE JSON_VALID(fa.organs)
  AND (:user_id IS NULL OR fa.user_id = :user_id)
  AND (:start   IS NULL OR fa.created_at >= :start)
  AND (:end     IS NULL OR fa.created_at < DATE_ADD(:end, INTERVAL 1 DAY))
GROUP BY month, organ
ORDER BY month, organ;
""")

def build_symptom_history_excel(user_id=None, start=None, end=None) -> io.BytesIO:
    # 1) 查資料
    with engine.begin() as conn:
        df = pd.read_sql(SQL, conn, params={"user_id": user_id, "start": start, "end": end})

    # 沒有資料
    if df.empty:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            pd.DataFrame([{"訊息": "查無資料"}]).to_excel(writer, index=False, sheet_name="History")
        bio.seek(0)
        return bio

    # 2) 轉「器官 × 月份」矩陣
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")
    pivot = df.pivot_table(index="organ", columns="month", values="symptom_count", fill_value=0)
    full_months = pd.period_range(pivot.columns.min(), pivot.columns.max(), freq="M")
    pivot = pivot.reindex(columns=full_months, fill_value=0).sort_index()
    pivot.columns = pivot.columns.astype(str)  # 讓欄名在 Excel 顯示更直覺

    # 3) 寫入 Excel
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:

        df_out = pivot.reset_index()
        df_out.to_excel(writer, sheet_name="History", index=False)
        ws = writer.sheets["History"]

        # 數字區範圍（B2 起 → 到最後一個月份欄與資料列）
        nrows, ncols = df_out.shape
        start_row, start_col = 2, 2  # B2
        end_row = nrows
        end_col = ncols
        data_range = f"{get_column_letter(start_col)}{start_row}:{get_column_letter(end_col)}{end_row}"

        # 三色條件式
        green  = PatternFill(start_color="4CAF50", end_color="4CAF50", fill_type="solid")
        yellow = PatternFill(start_color="FFD54F", end_color="FFD54F", fill_type="solid")
        red    = PatternFill(start_color="E53935", end_color="E53935", fill_type="solid")
        ws.conditional_formatting.add(data_range, CellIsRule(operator='equal', formula=['0'], fill=green))
        ws.conditional_formatting.add(data_range, CellIsRule(operator='between', formula=['1','3'], fill=yellow))
        ws.conditional_formatting.add(data_range, CellIsRule(operator='greaterThan', formula=['3'], fill=red))

        # 右側色碼 + 說明
        legend_col = end_col + 2
        lc, lc2 = get_column_letter(legend_col), get_column_letter(legend_col + 1)
        ws[f"{lc}{start_row+1}"].fill = green
        ws[f"{lc2}{start_row+1}"] = "綠色(0次):無症狀紀錄，狀態穩定"
        ws[f"{lc}{start_row+2}"].fill = yellow
        ws[f"{lc2}{start_row+2}"] = "黃色(1–3次):輕度症狀出現，建議觀察追蹤"
        ws[f"{lc}{start_row+3}"].fill = red
        ws[f"{lc2}{start_row+3}"] = "紅色(≥4次):症狀較頻繁，建議深入檢查或諮詢醫師"

        # 4) 匯出日期時間
        ts_row = max(end_row, start_row + 3) + 2
        ts = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
        ts_cell = f"A{ts_row}"
        ws[ts_cell] = f"匯出日期時間：{ts}"
        ws[ts_cell].font = Font(bold=True)
        ws[ts_cell].alignment = Alignment(horizontal="left", vertical="top")

    bio.seek(0)
    return bio
