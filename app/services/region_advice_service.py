# app/services/region_advice_service.py
from __future__ import annotations
import asyncio
import importlib
import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
import time

# ================================
# 0) 常數與正規化
# ================================
COLOR_FALLBACK: Dict[str, Tuple[str, List[str]]] = {
    "發紅": ("偏熱/火旺；可與作息壓力或辛辣上火相關", ["口乾易煩", "睡不好", "容易上火"]),
    "發黑": ("腎精/腎陽偏虛或疲勞、睡眠不足相關", ["疲倦乏力", "腰痠", "黑眼圈"]),
    "發白": ("氣血不足或偏寒", ["頭暈", "手腳冰冷", "精神不濟"]),
    "發黃": ("脾胃運化欠佳或濕熱", ["食慾差", "痞悶倦怠", "舌苔厚膩"]),
    "發綠": ("氣機鬱結或寒滯（原青）", ["脹悶", "緊繃", "偶有隱痛"]),
}

ORGANS = {"心", "肝", "脾", "肺", "腎", "胃", "膽", "大腸", "小腸", "膀胱", "胞宮"}

PAREN = re.compile(r"[()（）]")
WS = re.compile(r"\s+")
BULLET_RE = re.compile(r"^\s*-\s*\*\*(發紅|發黑|發白|發黃|發綠)\*\*\s*→\s*(.*?)\s*→\s*(.*)\s*$")
AREA_H3_RE = re.compile(r"^\s*###\s*(.+?)\s*$")

def _strip_no_ws(s: str) -> str:
    return WS.sub("", (s or "").strip())

def normalize_status(s: str) -> str:
    t = str(s or "").strip()
    if "紅" in t or "赤" in t:
        return "發紅"
    if "黑" in t:
        return "發黑"
    if "白" in t or "蒼白" in t:
        return "發白"
    if "黃" in t:
        return "發黃"
    if "綠" in t or "青" in t:
        return "發綠"
    return t

def _extract_face_and_organ(label: str) -> tuple[str, str]:
    """同時兼容：'下頰，生殖功能(腎)' 與 '下頰，腎(生殖功能)' 等寫法。"""
    if not label:
        return "", ""
    # 先取逗號前真正門牌
    face = label.split("，")[0].strip()
    organ = ""
    # 先抓括號內
    in_parens = re.findall(r"[（(]([^（）()]*)[)）]", label)
    # 候選池：括號內 + 全字串（用來抓括號外的臟腑字）
    pools = in_parens + [label]
    for seg in pools:
        toks = re.split(r"[，,/\s]", seg)
        for t in toks:
            tt = t.strip()
            if tt in ORGANS:
                organ = tt
                break
        if organ:
            break
    return face, organ

def normalize_face_organ(area_label: str) -> tuple[str, str]:
    face, organ = _extract_face_and_organ(area_label)
    face_n = _strip_no_ws(face)
    organ_n = _strip_no_ws(organ)
    # 去除註語
    organ_n = organ_n.replace("兼肝交會", "")
    return face_n, organ_n

# 統一輸出格式：【區域(器官)】顏色 → 可能機理 → 常見表徵
def _fmt_line(area: str, status: str, why: str | None, symptoms_list: list[str] | None) -> str:
    area = (area or "").strip()
    status = (status or "").strip()
    why_out = (why or "").strip()
    # 若模型回了「資料未明確指出」，視為空白
    if "資料未明確指出" in why_out:
        why_out = ""
    sym_out = "、".join([str(s).strip() for s in (symptoms_list or []) if str(s).strip()])
    return f"【{area}】{status} → {why_out} → {sym_out}"

# ================================
# 1) 解析 face_map.md → 索引
# ================================
FACE_MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "face_map.md"
_FACE_CACHE: Dict[str, object] = {"mtime": 0.0, "index": {}}  # type: ignore

# 索引結構：index[(face_n, organ_n, status)] = (why, [symptoms...])
def _load_face_map_index(md_text: str) -> Dict[Tuple[str, str, str], Tuple[str, List[str]]]:
    index: Dict[Tuple[str, str, str], Tuple[str, List[str]]] = {}
    current_area = None

    for line in md_text.splitlines():
        # 區域標題
        m_area = AREA_H3_RE.match(line)
        if m_area:
            current_area = m_area.group(1).strip()  # 例：下巴(腎總區) / 下頰，腎(生殖功能) 等
            continue

        # 子彈條目
        m_b = BULLET_RE.match(line)
        if m_b and current_area:
            status = normalize_status(m_b.group(1))
            why = (m_b.group(2) or "").strip()
            sym_raw = (m_b.group(3) or "").strip()
            # 症狀以「、」「；」「;」「，」切開
            syms = [s.strip() for chunk in re.split(r"[；;]", sym_raw) for s in chunk.split("、")]
            syms = [s for s in syms if s]
            # 建立索引鍵
            face_n, organ_n = normalize_face_organ(current_area)
            key = (face_n, organ_n, status)
            index[key] = (why, syms)

    return index

def _get_face_map_index() -> Dict[Tuple[str, str, str], Tuple[str, List[str]]]:
    try:
        st = os.stat(FACE_MAP_PATH)
        if st.st_mtime != _FACE_CACHE["mtime"]:
            # 重新讀檔建索引
            text = FACE_MAP_PATH.read_text(encoding="utf-8")
            _FACE_CACHE["index"] = _load_face_map_index(text)
            _FACE_CACHE["mtime"] = st.st_mtime
        return _FACE_CACHE["index"]  # type: ignore
    except FileNotFoundError:
        # 找不到 md 也不要讓流程中斷
        return {}

def _lookup_from_face_map(area_label: str, status: str) -> Tuple[Optional[str], Optional[List[str]]]:
    idx = _get_face_map_index()
    face_n, organ_n = normalize_face_organ(area_label)
    st = normalize_status(status)

    # 完整鍵
    key = (face_n, organ_n, st)
    if key in idx:
        return idx[key]

    # 沒有臟腑的容錯（例如標題沒標或辨識沒抓到）
    key2 = (face_n, "", st)
    if key2 in idx:
        return idx[key2]

    # 反向容錯：只靠臟腑（不太建議，但盡量對上）
    for (f, o, s), val in idx.items():
        if s == st and o == organ_n and (f in face_n or face_n in f):
            return val

    return None, None

# ================================
# 2) RAG 檢索 + 生成（只處理缺少的）
# ================================
async def run_async_retr_then_gen(retr_query: str, gen_query: str, top_k: int = 24):
    m = importlib.import_module("app.services.rag_core")
    retrieve = getattr(m, "retrieve")
    generate_answer = getattr(m, "generate_answer")
    ctx = await retrieve(retr_query, top_k=top_k)
    answer = await generate_answer(query=gen_query, contexts=ctx)
    return answer, ctx

def sources_from_ctx(ctx: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for c in ctx or []:
        meta = c.get("metadata", {})
        out.append({"source": meta.get("source"), "chunk": meta.get("chunk"), "score": c.get("score")})
    return out

def build_retrieval_query(region_results: Dict[str, str]) -> str:
    lines = []
    # 強化檢索：把「### 區域」頭與「- **發色**」關鍵詞當成查詢提示
    for area, status in (region_results or {}).items():
        face, organ = _extract_face_and_organ(area)
        st = normalize_status(status)
        header = f"### {face}({organ})" if organ else f"### {face}"
        lines.append(f"{header}\n- **{st}**")
    # 補上通用詞
    lines.append("臉部 望色 臟腑 發紅 發黑 發白 發黃 發綠 額部 鼻翼 鼻頭 鼻根 右上頰 左上頰 顴內 顴外 下巴 下頰 人中 眼白")
    return "\n".join(lines)

def all_in_one_query(region_results: Dict[str, str],
                     overall_color: Optional[Dict],
                     has_moles: bool, mole_analysis: dict,
                     has_beard: bool, beard_analysis: dict) -> str:
    items = [{"area": k, "status": normalize_status(v)} for k, v in (region_results or {}).items()]
    overall_hex = (overall_color or {}).get("hex") or "未知"
    return f"""
你是中醫知識助理，使用繁體中文。**請只輸出 JSON**，不要有任何多餘文字。

對【觀察項目】中的每一筆都必須回應，且：
- `area` **必須完整複製**觀察項目裡的字串（包含括號與臟腑），**不可改寫/刪減**。
- `status` 僅能是這五個之一：["發紅","發黑","發白","發黃","發綠"]。
- `why` 請依據文件（如 face_map.md）中**對應部位與臟腑**的條目解釋；無直接條目時，依同臟腑或同色的相近條目**合理推斷**，以「可能」語氣描述。
- `symptoms` 為精簡列點（0~3 個）；若無依據請回空陣列 []，**不要**回「資料未明確指出」。

輸出 JSON 形如：
{{
  "regions": [
    {{"area":"〈原樣〉","status":"發紅/發黑/發白/發黃/發綠","why":"…","symptoms":["…","…"]}}
  ],
  "summary": ""
}}

【觀察項目】:
{json.dumps(items, ensure_ascii=False)}

【補充資訊】:
- 整體臉色 HEX: {overall_hex}
- 痣: {"有" if has_moles else "無"}（{json.dumps(mole_analysis or {}, ensure_ascii=False)}）
- 鬍鬚: {"有" if has_beard else "無"}（{json.dumps(beard_analysis or {}, ensure_ascii=False)}）
"""

# ================================
# 3) 對外入口：一次搞定全部
# ================================
def get_region_advice_all_in_one(
    region_results: Dict[str, str],
    overall_color: Optional[Dict] = None,
    has_moles: bool = False, mole_analysis: dict | None = None,
    has_beard: bool = False, beard_analysis: dict | None = None,
) -> Tuple[str, Dict[str, str], List[Dict]]:
    """
    回傳：
    - diagnosis_text：每個區域一行（段落間空一行）「【區域(臟腑)】顏色 → 可能機理 → 常見表徵」
    - per_region    ：{原始區域key: 單行文字}
    - advice_sources：檢索來源
    """
    if not region_results:
        return ("", {}, [])

    per_region: Dict[str, str] = {}
    missing: Dict[str, str] = {}

    # 1) 先用 md 索引逐一命中
    for area, status in (region_results or {}).items():
        why, syms = _lookup_from_face_map(area, status)
        if why is not None:  # 命中（就算 why 空字串也算命中）
            per_region[area] = _fmt_line(area, normalize_status(status), why, syms or [])
        else:
            missing[area] = status  # 交給 RAG 或 fallback

    sources: List[Dict] = []

    # 2) 針對 miss 的才丟給 RAG（降低成本&延遲）
    if missing:
        retr_q = build_retrieval_query(missing)
        gen_q  = all_in_one_query(
            missing, overall_color,
            has_moles, mole_analysis or {},
            has_beard, beard_analysis or {}
        )
        print(f"[RAG] ALL-IN-ONE start, regions={len(missing)}")
        txt, ctx = asyncio.run(run_async_retr_then_gen(retr_q, gen_q, top_k=24))
        print(f"[RAG] ALL-IN-ONE done")

        sources = sources_from_ctx(ctx)

        # 解析 JSON
        def _extract_json(s: str) -> dict | None:
            if not s:
                return None
            m = re.search(r"\{.*\}", s, re.S)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

        data = _extract_json(txt)
        seen_keys: set[str] = set()

        if isinstance(data, dict) and isinstance(data.get("regions"), list):
            # 建 key 索引把模型回的 area 對回「原始 key」
            key_index: Dict[Tuple[str, str], str] = {}
            for k in region_results.keys():
                f_n, o_n = normalize_face_organ(k)
                key_index[(f_n, o_n)] = k
                if (f_n, "") not in key_index:
                    key_index[(f_n, "")] = k

            for item in data["regions"]:
                raw_area = (item.get("area") or "").strip()
                if not raw_area:
                    continue
                f_n, o_n = normalize_face_organ(raw_area)
                orig_key = key_index.get((f_n, o_n)) or key_index.get((f_n, ""))
                if not orig_key:
                    # 退一步：臉部名稱包含匹配
                    for (ff, oo), kk in key_index.items():
                        if f_n and ff and (f_n in ff or ff in f_n):
                            orig_key = kk
                            break
                if not orig_key:
                    continue

                st = normalize_status(item.get("status") or missing.get(orig_key, ""))
                why = (item.get("why") or "").strip()
                syms = [str(s).strip() for s in (item.get("symptoms") or []) if str(s).strip()]
                if "資料未明確指出" in why:
                    why = ""  # 讓後面 fallback/留白處理

                if not why:
                    fb = COLOR_FALLBACK.get(st)
                    if fb:
                        why, syms = fb[0], fb[1]

                per_region[orig_key] = _fmt_line(orig_key, st, why, syms)
                seen_keys.add(orig_key)

        # 3) RAG 還是沒覆蓋到的，用顏色 fallback
        for area, status in missing.items():
            if area not in per_region:
                st = normalize_status(status)
                fb = COLOR_FALLBACK.get(st)
                if fb:
                    why, syms = fb
                    per_region[area] = _fmt_line(area, st, why, syms)
                else:
                    per_region[area] = _fmt_line(area, st, "", [])

    # 4) 排序輸出：維持原輸入順序，段落間空一行
    ordered_lines = [per_region[k] for k in region_results.keys() if k in per_region]
    diagnosis_text = "\n\n".join(ordered_lines).strip()

    # 如果完全沒進 RAG，但有用到 md，補上一個來源提示
    if not sources and _get_face_map_index():
        sources = [{"source": str(FACE_MAP_PATH), "chunk": -1, "score": 1.0}]

    return diagnosis_text, per_region, sources

# 相容舊名稱
def get_region_advice_batch(*args, **kwargs):
    return get_region_advice_all_in_one(*args, **kwargs)
