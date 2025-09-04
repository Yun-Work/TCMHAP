from flask import Blueprint, request, jsonify
import traceback
from datetime import datetime
import base64
import numpy as np
import cv2
import json  # 🆕 for prompt/sources serialization
from app.db import SessionLocal
from app.models.face_analysis_model import FaceAnalysis
import importlib
import asyncio
import traceback as _tb
from typing import List, Dict, Tuple   # 若你的 Py 版本 < 3.9 必用；>=3.9 也可保留

# 嘗試載入 RAG 服務（query_rag -> rag_core）
try:  # 🆕
    from app.services.rag_core import retrieve, generate_answer
    _rag_ready = True
except Exception as _e:  # 🆕
    print(f"⚠️ Blueprint: RAG 服務不可用：{_e}")
    _rag_ready = False

# 創建Blueprint
face_analysis_bp = Blueprint('face_analysis', __name__)

# 導入服務
analysis_service = None
diagnosis_service = None
mole_detection_service = None

try:
    from app.services.facialskincoloranalysis_service import analyze_face_from_base64, FaceSkinAnalyzer
    analysis_service = analyze_face_from_base64
    print("✅ Blueprint: 成功導入分析服務")
except ImportError as e:
    print(f"⚠️ Blueprint: 分析服務導入失敗: {e}")

try:
    from app.services.mole_detection_service import MoleDetectionService, remove_beard_from_image
    mole_detection_service = MoleDetectionService()
    print("✅ Blueprint: 成功導入痣檢測服務")
except ImportError as e:
    print(f"⚠️ Blueprint: 痣檢測服務導入失敗: {e}")


def base64_to_image(base64_string):
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Base64轉圖像失敗: {e}")
        return None


def image_to_base64(image):
    try:
        _, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"圖像轉Base64失敗: {e}")
        return None


def detect_beard_features(image_data):
    try:
        if not mole_detection_service:
            return {'has_beard': False, 'beard_count': 0}
        image = base64_to_image(image_data)
        if image is None:
            return {'has_beard': False, 'beard_count': 0}
        has_beard, beards, _ = mole_detection_service.detect_beard_hair(image)
        return {
            'has_beard': has_beard,
            'beard_count': len(beards) if beards else 0
        }
    except Exception as e:
        print(f"鬍鬚檢測錯誤: {e}")
        return {'has_beard': False, 'beard_count': 0}


def process_beard_removal(image_data):
    try:
        if not mole_detection_service:
            return None
        image = base64_to_image(image_data)
        if image is None:
            return None
        processed_image, beard_info = remove_beard_from_image(image)
        processed_base64 = image_to_base64(processed_image)
        return processed_base64
    except Exception as e:
        print(f"鬍鬚移除錯誤: {e}")
        return None


# =========================
# 🆕 RAG 提示建構 & Fallback
# =========================
def _build_advice_prompt(region_results: dict, all_region_results: dict, overall_color: dict | None,
                         has_moles: bool, mole_analysis: dict, has_beard: bool, beard_analysis: dict) -> str:
    abnormal_pairs = [f"{k}：{v}" for k, v in (region_results or {}).items()]
    overall_hex = overall_color.get("hex") if isinstance(overall_color, dict) else None
    prompt = f"""你是一位中醫知識助理，請根據臉部觀察資料，輸出「非醫療診斷」的健康建議，口吻中立客觀，避免病名，使用「可能/建議/可留意」等字眼。輸出為短段落（120~220字），繁體中文。

【觀察資料】
- 異常區域（位置→顏色）：{ "、".join(abnormal_pairs) if abnormal_pairs else "無" }
- 整體臉色 (HEX)：{overall_hex or "未知"}
- 痣：{"有" if has_moles else "無"}（統計：{json.dumps(mole_analysis or {}, ensure_ascii=False)}）
- 鬍鬚：{"有" if has_beard else "無"}（統計：{json.dumps(beard_analysis or {}, ensure_ascii=False)}）

【輸出格式】
直接回覆建議段落，不要加標題或項目符號；避免下診斷、避免指示性治療；可包含作息、飲食、情緒與運動等一般性建議。"""
    return prompt


def _split_area_label(area: str) -> tuple[str, str]:
    if not area:
        return "", ""
    area = area.strip()
    if "(" in area and area.endswith(")"):
        base, organ = area.split("(", 1)
        return base.strip()[:5], organ[:-1].strip()[:5]
    return area[:5], ""


def _build_area_advice_prompt(area_label: str, status: str,
                              overall_color: dict | None,
                              has_moles: bool, mole_analysis: dict,
                              has_beard: bool, beard_analysis: dict) -> str:
    hexv = overall_color.get("hex") if isinstance(overall_color, dict) else None
    return (
        "你是一位臉色觀察與臟腑對應的助理。請務必以繁體中文回答，避免英文；"
        "請在一個連貫段落中依序回答所有問題，不要換行、不要列點、不要標題；"
        "病機與症狀使用保守語氣（可能提示、或與…相關、可能伴隨…）；"
        "若無明確答案請用『資料未明確指出』。\n\n"
        "字數約 80～160 字。\n\n"
        f"【單一觀察重點】\n- 區域：{area_label}\n- 現象：{status}\n"
        f"- 整體臉色 HEX：{hexv or '未知'}\n"
        f"- 痣：{'有' if has_moles else '無'}（統計：{json.dumps(mole_analysis or {}, ensure_ascii=False)}）\n"
        f"- 鬍鬚：{'有' if has_beard else '無'}（統計：{json.dumps(beard_analysis or {}, ensure_ascii=False)}）\n\n"
        "請輸出一段建議文字（不要條列清單、不要標題、不要引用來源）。"
    )


# ==== 只用 RAG 的同步包裝 ====
def _run_rag_sync(prompt: str) -> Tuple[str, List[Dict]]:
    try:
        m = importlib.import_module("app.services.rag_core")
        _retrieve = getattr(m, "retrieve", None)
        _generate_answer = getattr(m, "generate_answer", None)
        if not callable(_retrieve) or not callable(_generate_answer):
            raise RuntimeError("rag_core 缺少 retrieve/generate_answer")

        async def _flow():
            ctx = await _retrieve(prompt, top_k=4)
            txt = await _generate_answer(
                query="請根據觀察資料給非醫療建議（120-220字、繁中）。",
                contexts=ctx
            )
            return txt, ctx

        txt, ctx = asyncio.run(_flow())
        sources: List[Dict] = []
        for c in ctx or []:
            meta = c.get("metadata", {})
            sources.append({
                "source": meta.get("source"),
                "chunk": meta.get("chunk"),
                "score": c.get("score"),
            })
        return (txt or "").strip(), sources
    except Exception:
        _tb.print_exc()
        return "", []


def _get_rag_advice_text(region_results, all_region_results, overall_color,
                         has_moles, mole_analysis, has_beard, beard_analysis) -> tuple[str, list[dict]]:
    try:
        prompt = _build_advice_prompt(
            region_results, all_region_results, overall_color,
            has_moles, mole_analysis, has_beard, beard_analysis
        )
        advice_text, sources = _run_rag_sync(prompt)
        return (advice_text or "").strip(), (sources or [])
    except Exception as e:
        print(f"⚠️ 產生建議失敗：{e}")
        return "", []


# =========================
# 🆕 新增：把「整體建議」存成 FaceAnalysis 的一筆「總結」列，並可讀最新總結
# =========================
def _save_summary_to_db(db, summary_text: str):
    """把整體建議保存為一筆 FaceAnalysis（face='總結', organ='', status='總結'）。"""
    try:
        if not summary_text:
            return None
        row = FaceAnalysis(
            face="總結",
            organ="",
            status="總結",
            message=summary_text[:2000],  # 防超長
            analysis_date=datetime.utcnow()
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    except Exception as e:
        db.rollback()
        print(f"⚠️ 保存總結失敗：{e}")
        return None


def _fetch_latest_summary(db):
    """讀取最新一筆 face='總結' 的 message 當作 diagnosis_text。"""
    try:
        q = db.query(FaceAnalysis).filter(FaceAnalysis.face == "總結").order_by(FaceAnalysis.analysis_date.desc())
        row = q.first()
        return (row.message if row else "") or ""
    except Exception as e:
        print(f"⚠️ 讀取總結失敗：{e}")
        return ""


@face_analysis_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "analysis_service": analysis_service is not None,
        "diagnosis_service": diagnosis_service is not None,
        "mole_detection_service": mole_detection_service is not None,
        "rag_service": _rag_ready
    })


@face_analysis_bp.route('/upload', methods=['POST'])
def upload_and_analyze():
    print("\n" + "=" * 60)
    print("🔍 Blueprint: 收到新的分析請求")

    try:
        # 驗證
        if not request.is_json:
            return jsonify({
                "success": False, "error": "請求格式錯誤",
                "abnormal_count": 0, "overall_color": None,
                "all_region_results": {}, "region_results": {},
                "diagnoses": {}, "diagnosis_text": "",
                "has_moles": False, "has_beard": False,
                "mole_analysis": None, "beard_analysis": None
            }), 400

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False, "error": "缺少image字段",
                "abnormal_count": 0, "overall_color": None,
                "all_region_results": {}, "region_results": {},
                "diagnoses": {}, "diagnosis_text": "",
                "has_moles": False, "has_beard": False,
                "mole_analysis": None, "beard_analysis": None
            }), 400

        image_data = data['image']
        remove_moles = data.get('remove_moles', False)
        remove_beard = data.get('remove_beard', False)

        print(f"🔷 Blueprint: 圖片數據長度: {len(image_data)}")
        print(f"🔷 Blueprint: 移除痣: {remove_moles}, 移除鬍鬚: {remove_beard}")

        if analysis_service is None:
            print("❌ Blueprint: 分析服務不可用")
            return jsonify({
                "success": False, "error": "分析服務未正確安裝",
                "abnormal_count": 0, "overall_color": None,
                "all_region_results": {}, "region_results": {},
                "diagnoses": {}, "diagnosis_text": "",
                "has_moles": False, "has_beard": False,
                "mole_analysis": None, "beard_analysis": None
            })

        # 圖像預處理（鬍鬚移除）
        processed_image_data = image_data
        if remove_beard:
            print("🧔 Blueprint: 開始移除鬍鬚...")
            beard_removed_image = process_beard_removal(image_data)
            if beard_removed_image:
                processed_image_data = beard_removed_image
                print("✅ Blueprint: 鬍鬚移除完成")
            else:
                print("⚠️ Blueprint: 鬍鬚移除失敗，使用原始圖像")

        # 主分析
        try:
            analyzer = FaceSkinAnalyzer()
            result = analyzer.analyze_from_base64(processed_image_data, remove_moles, remove_beard)
        except Exception as analyzer_error:
            print(f"⚠️ Blueprint: 使用新分析器失敗，回退到舊分析器: {analyzer_error}")
            result = analysis_service(processed_image_data)
            result["has_moles"] = False
            result["has_beard"] = False
            result["mole_analysis"] = {"mole_count": 0, "total_moles": 0}
            result["beard_analysis"] = {"beard_count": 0}
            result["moles_removed"] = False
            result["beard_removed"] = False

        print(f"✅ Blueprint: 分析完成，成功: {result.get('success', False)}")
        if not result.get('success', False):
            return jsonify({
                "success": False,
                "error": result.get('error', '分析失敗'),
                "abnormal_count": 0, "overall_color": None,
                "all_region_results": {}, "region_results": {},
                "diagnoses": {}, "diagnosis_text": "",
                "has_moles": False, "has_beard": False,
                "mole_analysis": None, "beard_analysis": None
            })

        # 鬍鬚檢測（若未移除）
        beard_detection_result = {'has_beard': False, 'beard_count': 0}
        if not remove_beard:
            print("🔍 Blueprint: 檢測鬍鬚特徵...")
            beard_detection_result = detect_beard_features(image_data)
            print(f"🧔 Blueprint: 鬍鬚檢測結果: {beard_detection_result}")

        # 產生整體建議（RAG）
        advice_text, advice_sources = _get_rag_advice_text(
            region_results=result.get("region_results", {}),
            all_region_results=result.get("all_region_results", {}),
            overall_color=result.get("overall_color"),
            has_moles=result.get("has_moles", False),
            mole_analysis=result.get("mole_analysis", {}),
            has_beard=beard_detection_result.get("has_beard", False) if not remove_beard else False,
            beard_analysis={
                "beard_count": beard_detection_result.get("beard_count", 0) if not remove_beard else 0,
                "has_beard": beard_detection_result.get("has_beard", False) if not remove_beard else False
            }
        )

        # 準備回傳骨架
        response_data = {
            "success": True,
            "error": None,
            "abnormal_count": result.get("abnormal_count", 0),
            "overall_color": result.get("overall_color", None),
            "all_region_results": result.get("all_region_results", {}),
            "region_results": result.get("region_results", {}),
            "diagnoses": {},
            "diagnosis_text": "",  # 先占位，稍後用「DB 最新總結」覆蓋
            "has_moles": result.get("has_moles", False),
            "mole_analysis": result.get("mole_analysis", {"mole_count": 0, "total_moles": 0}),
            "moles_removed": result.get("moles_removed", remove_moles),
            "has_beard": beard_detection_result['has_beard'] if not remove_beard else False,
            "beard_analysis": {
                "beard_count": beard_detection_result['beard_count'] if not remove_beard else 0,
                "has_beard": beard_detection_result['has_beard'] if not remove_beard else False
            },
            "beard_removed": remove_beard,
            "advice_sources": advice_sources,
        }

        print("📊 Blueprint: 準備寫 DB（總結 + 各區域）")
        # =========================
        # 🆕 寫入「總結」＋「各異常區域」到 DB，並以 DB 讀回的總結回填 diagnosis_text
        # =========================
        try:
            area_map = result.get("region_results") or {}
            now = datetime.utcnow()
            has_beard = (beard_detection_result.get("has_beard", False) if not remove_beard else False)
            beard_ana = {
                "beard_count": beard_detection_result.get("beard_count", 0) if not remove_beard else 0,
                "has_beard": has_beard
            }
            has_moles = result.get("has_moles", False)
            mole_ana = result.get("mole_analysis", {})
            area_advices = {}
            written = 0

            with SessionLocal() as db:
                # 1) 先把整體建議存成一筆「總結」
                _save_summary_to_db(db, advice_text)

                # 2) 逐區域 RAG → 寫入
                for area_label, status_str in area_map.items():
                    prompt = _build_area_advice_prompt(
                        area_label=area_label,
                        status=status_str,
                        overall_color=result.get("overall_color"),
                        has_moles=has_moles,
                        mole_analysis=mole_ana,
                        has_beard=has_beard,
                        beard_analysis=beard_ana
                    )
                    per_text, per_sources = _run_rag_sync(prompt)
                    face_val, organ_val = _split_area_label(area_label)
                    status_val = (status_str or "未知")[:5]
                    row = FaceAnalysis(
                        face=face_val,
                        organ=organ_val,
                        status=status_val,
                        message=per_text,
                        analysis_date=now
                    )
                    db.add(row)
                    db.commit()
                    db.refresh(row)
                    written += 1
                    area_advices[area_label] = {
                        "advice": per_text,
                        "sources": per_sources,
                        "fa_id": getattr(row, "fa_id", None)
                    }

                # 3) 用 DB 讀回最新的「總結」填入 diagnosis_text（➡️ 給前端 fromJson 使用）
                latest_summary = _fetch_latest_summary(db)
                response_data["diagnosis_text"] = latest_summary

            print(f"🗄️ Blueprint: face_analysis 已寫入：總結 1 筆 + 區域 {written} 筆")
            response_data["diagnoses"] = area_advices

        except Exception as e:
            print(f"⚠️ Blueprint: 寫入/讀取 DB 失敗：{e}")
            # 如果 DB 有問題，至少把 RAG 產出的整體建議直接回傳
            response_data["diagnosis_text"] = advice_text or ""

        # 回傳
        print(f"   - 建議文字(回傳)：{response_data['diagnosis_text'][:80]}{'...' if len(response_data['diagnosis_text'])>80 else ''}")
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Blueprint: 整體錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"服務器錯誤: {str(e)}",
            "abnormal_count": 0,
            "overall_color": None,
            "all_region_results": {},
            "region_results": {},
            "diagnoses": {},
            "diagnosis_text": "",
            "has_moles": False,
            "has_beard": False,
            "mole_analysis": None,
            "beard_analysis": None
        }), 500


@face_analysis_bp.route('/analyze', methods=['POST'])
def analyze():
    """與 /upload 相同的功能，為了兼容性"""
    return upload_and_analyze()


@face_analysis_bp.route('/analyze_face', methods=['POST'])
def analyze_face():
    """專門處理包含痣和鬍鬚檢測的分析請求"""
    try:
        data = request.get_json()
        base64_image = data.get('image')
        remove_moles = data.get('remove_moles', False)
        remove_beard = data.get('remove_beard', False)

        if not base64_image:
            return jsonify({'success': False, 'error': '缺少圖像數據'}), 400

        print(f"🔍 analyze_face: 移除痣={remove_moles}, 移除鬍鬚={remove_beard}")

        processed_image_data = base64_image
        if remove_beard:
            print("🧔 analyze_face: 開始移除鬍鬚...")
            beard_removed_image = process_beard_removal(base64_image)
            if beard_removed_image:
                processed_image_data = beard_removed_image
                print("✅ analyze_face: 鬍鬚移除完成")

        try:
            analyzer = FaceSkinAnalyzer()
            result = analyzer.analyze_from_base64(processed_image_data, remove_moles, remove_beard)
        except Exception as e:
            print(f"❌ analyze_face: 新分析器執行失敗: {e}")
            if analysis_service:
                result = analysis_service(processed_image_data)
                result["has_moles"] = False
                result["has_beard"] = False
                result["mole_analysis"] = {"mole_count": 0, "total_moles": 0}
                result["beard_analysis"] = {"beard_count": 0}
                result["moles_removed"] = False
                result["beard_removed"] = False
            else:
                return jsonify({'success': False, 'error': '分析服務不可用'}), 500

        if not remove_beard and result.get('success', False):
            beard_detection_result = detect_beard_features(base64_image)
            result["has_beard"] = beard_detection_result['has_beard']
            result["beard_analysis"] = {
                "beard_count": beard_detection_result['beard_count'],
                "has_beard": beard_detection_result['has_beard']
            }
        else:
            result["has_beard"] = False
            result["beard_analysis"] = {"beard_count": 0, "has_beard": False}

        result["beard_removed"] = remove_beard

        # 產生整體建議（RAG）
        advice_text, advice_sources = _get_rag_advice_text(
            region_results=result.get("region_results", {}),
            all_region_results=result.get("all_region_results", {}),
            overall_color=result.get("overall_color"),
            has_moles=result.get("has_moles", False),
            mole_analysis=result.get("mole_analysis", {}),
            has_beard=result.get("has_beard", False),
            beard_analysis=result.get("beard_analysis", {})
        )

        # 🆕：把「整體建議」寫 DB 並用 DB 的最新總結回填 diagnosis_text
        with SessionLocal() as db:
            _save_summary_to_db(db, advice_text)
            latest_summary = _fetch_latest_summary(db)
        result["diagnosis_text"] = latest_summary
        result["advice_sources"] = advice_sources

        return jsonify(result)

    except Exception as e:
        print(f"❌ analyze_face錯誤: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'分析失敗: {str(e)}'}), 500


@face_analysis_bp.route('/detect_features', methods=['POST'])
def detect_features():
    try:
        data = request.get_json()
        base64_image = data.get('image')
        if not base64_image:
            return jsonify({'success': False, 'error': '缺少圖像數據'}), 400

        mole_result = {'has_moles': False, 'mole_count': 0}
        if mole_detection_service:
            try:
                image = base64_to_image(base64_image)
                if image is not None:
                    analysis_result = mole_detection_service.comprehensive_mole_analysis(image)
                    mole_result = {
                        'has_moles': analysis_result.get('has_dark_areas', False),
                        'mole_count': analysis_result.get('summary', {}).get('spot_count', 0)
                    }
            except Exception as e:
                print(f"痣檢測失敗: {e}")

        beard_result = detect_beard_features(base64_image)

        return jsonify({
            'success': True,
            'mole_detection': mole_result,
            'beard_detection': beard_result,
            'has_features': mole_result['has_moles'] or beard_result['has_beard']
        })

    except Exception as e:
        print(f"❌ detect_features錯誤: {e}")
        return jsonify({'success': False, 'error': f'特徵檢測失敗: {str(e)}'}), 500