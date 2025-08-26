# app/services/export_service.py
import io
import base64
import pandas as pd
from sqlalchemy import text
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter
from datetime import datetime
from zoneinfo import ZoneInfo
from app.db import engine

# 展開 organs(JSON 陣列) → 月份×器官 次數
SQL = text("""
SELECT
  DATE_FORMAT(fa.created_at, '%Y-%m-01') AS month,
  jt.organ AS organ,
  COUNT(*) AS symptom_count
FROM face_analysis fa
JOIN JSON_TABLE(
  fa.organs, '$[*]' COLUMNS (organ VARCHAR(50) PATH '$')
) jt
GROUP BY month, organ
ORDER BY month, organ;
""")

def _make_excel_stream(df: pd.DataFrame) -> io.BytesIO:
    """把查詢後的長表資料 -> 轉成 Excel 並回傳 BytesIO。"""
    bio = io.BytesIO()

    if df.empty:
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            pd.DataFrame([{"訊息": "查無資料"}]).to_excel(writer, index=False, sheet_name="History")
        bio.seek(0)
        return bio

    # 器官 × 月份矩陣；補齊連續月份
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")
    pivot = df.pivot_table(index="organ", columns="month", values="symptom_count", fill_value=0)
    full_months = pd.period_range(pivot.columns.min(), pivot.columns.max(), freq="M")
    pivot = pivot.reindex(columns=full_months, fill_value=0).sort_index()
    pivot.columns = pivot.columns.astype(str)

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_out = pivot.reset_index()                      # ["organ", "YYYY-MM",...]
        df_out.to_excel(writer, sheet_name="History", index=False)
        ws = writer.sheets["History"]

        # 數字區範圍（B2 起 → 到最後一個月份欄與最後一列）
        nrows, ncols = df_out.shape                       # 含表頭
        start_row, start_col = 2, 2                       # B2
        end_row = nrows
        end_col = ncols                                   # 最右欄是最後一個月份
        data_range = f"{get_column_letter(start_col)}{start_row}:{get_column_letter(end_col)}{end_row}"

        # 三色條件式（順序：紅 → 黃 → 綠）
        green  = PatternFill(start_color="4CAF50", end_color="4CAF50", fill_type="solid")
        yellow = PatternFill(start_color="FFD54F", end_color="FFD54F", fill_type="solid")
        red    = PatternFill(start_color="E53935", end_color="E53935", fill_type="solid")
        ws.conditional_formatting.add(data_range, CellIsRule(operator='greaterThan', formula=['3'], fill=red))
        ws.conditional_formatting.add(data_range, CellIsRule(operator='between',    formula=['1','3'], fill=yellow))
        ws.conditional_formatting.add(data_range, CellIsRule(operator='equal',      formula=['0'],     fill=green))

        # 右側色碼 + 說明（與資料上緣對齊）
        legend_col = end_col + 2
        lc, lc2 = get_column_letter(legend_col), get_column_letter(legend_col + 1)
        ws[f"{lc}{start_row+1}"].fill = green
        ws[f"{lc2}{start_row+1}"] = "綠色(0次):無症狀紀錄，狀態穩定"
        ws[f"{lc}{start_row+2}"].fill = yellow
        ws[f"{lc2}{start_row+2}"] = "黃色(1–3次):輕度症狀出現，建議觀察追蹤"
        ws[f"{lc}{start_row+3}"].fill = red
        ws[f"{lc2}{start_row+3}"] = "紅色(≥4次):症狀較頻繁，建議深入檢查或諮詢醫師"

        # --- 匯出日期時間：放在表格最下方
        ts = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")

        footer_row = end_row + 2  # 資料最後一列再往下兩列，避開條件格式區
        footer_start_col = 1  # A 欄
        footer_end_col = end_col  # 最後一個月份欄
        footer_range = f"{get_column_letter(footer_start_col)}{footer_row}:{get_column_letter(footer_end_col)}{footer_row}"

        # 合併儲存格並寫入文字
        ws.merge_cells(footer_range)
        ws[f"A{footer_row}"] = f"匯出日期時間：{ts}"
        ws[f"A{footer_row}"].font = Font(bold=True)
        ws[f"A{footer_row}"].alignment = Alignment(horizontal="left", vertical="center")

    bio.seek(0)
    return bio

def build_symptom_history_excel(user_id=None, start=None, end=None) -> io.BytesIO:
    """回傳 Excel 的 BytesIO（給 send_file 下載）。"""
    with engine.begin() as conn:
        df = pd.read_sql(SQL, conn, params={"user_id": user_id, "start": start, "end": end})
    return _make_excel_stream(df)

def build_symptom_history_excel_base64(user_id=None, start=None, end=None) -> str:
    """回傳 Excel 的 base64 字串（給 JSON 回傳）。"""
    with engine.begin() as conn:
        df = pd.read_sql(SQL, conn, params={"user_id": user_id, "start": start, "end": end})
    bio = _make_excel_stream(df)
    return base64.b64encode(bio.getvalue()).decode("utf-8")
