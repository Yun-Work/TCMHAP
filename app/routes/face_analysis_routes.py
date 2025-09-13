# app/routes/face_analysis_routes.py
import base64
import traceback
from datetime import datetime
from typing import Dict
from sqlalchemy import text
import re
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from app.db import SessionLocal
from app.models.face_analysis_model import FaceAnalysis
# ✅ 改為統一呼叫新的 RAG 服務（同步介面，內部自己處理 async）
from app.services.region_advice_service import get_region_advice_batch as advise_for_regions

# ✅ 要這一行（使用我們剛剛的 all-in-one 包裝）

# 嘗試載入 RAG 服務檢查（僅供 /health 顯示用）
try:
    from app.services.rag_core import retrieve, generate_answer  # 若缺少不影響主流程
    _rag_ready = True
except Exception as _e:
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


def base64_to_image(base64_string: str):
    """將base64字符串轉換為OpenCV圖像"""
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


def image_to_base64(image) -> str | None:
    """將OpenCV圖像轉換為base64字符串"""
    try:
        _, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"圖像轉Base64失敗: {e}")
        return None


def detect_beard_features(image_data: str) -> Dict:
    """檢測圖像中的鬍鬚特徵"""
    try:
        if not mole_detection_service:
            return {'has_beard': False, 'beard_count': 0}

        image = base64_to_image(image_data)
        if image is None:
            return {'has_beard': False, 'beard_count': 0}

        # 使用痣檢測服務來檢測鬍鬚
        has_beard, beards, _ = mole_detection_service.detect_beard_hair(image)
        return {
            'has_beard': has_beard,
            'beard_count': len(beards) if beards else 0
        }
    except Exception as e:
        print(f"鬍鬚檢測錯誤: {e}")
        return {'has_beard': False, 'beard_count': 0}


def process_beard_removal(image_data: str) -> str | None:
    """處理鬍鬚移除"""
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
def load_code_maps(session):
    """
    從 sys_code 一次撈出 face/organ/status 的 name→code_id 對照表（字典）
    回傳: (face_map, organ_map, status_map)；value 直接轉為字串，便於寫入你現有的 VARCHAR 欄位
    """
    rows = session.execute(text("""
        SELECT code_type, code_id, code_name
        FROM sys_code
    """)).fetchall()

    face_map, organ_map, status_map = {}, {}, {}
    for code_type, code_id, code_name in rows:
        name = (code_name or "").strip()
        if code_type == "face":
            face_map[name] = str(code_id)
        elif code_type == "organ":
            organ_map[name] = str(code_id)
        elif code_type == "status":
            status_map[name] = str(code_id)
    return face_map, organ_map, status_map


# 只把「最後一段括號」視為臟腑；前面全部保留給權息點名稱
# 例： "鼻根(心與肝交會)(心)" → face="鼻根(心與肝交會)" , organ="心"
_SPLIT_PATTERN = re.compile(r"^(?P<face>.+)\((?P<organ>[^()]+)\)$")

def split_area_label_exact(area: str) -> tuple[str, str]:
    """
    將像 '鼻翼(胃)'、'鼻根(心與肝交會)(心)'、'下頰(腎(生殖功能))' 這類字串
    分成 (face_name, organ_name)。能處理巢狀括號。
    """
    if not area:
        return "", ""
    s = area.strip()
    if not s.endswith(')'):
        return s, ""  # 沒有以 ) 結尾，視為無 organ

    # 從尾端回掃，找到與最後一個 ')' 對應的 '('
    depth = 0
    open_idx = -1
    for i in range(len(s) - 1, -1, -1):
        c = s[i]
        if c == ')':
            depth += 1
        elif c == '(':
            depth -= 1
            if depth == 0:
                open_idx = i
                break

    if open_idx == -1:
        # 括號不成對，當成無 organ
        return s, ""

    face_name = s[:open_idx].strip()
    organ_name = s[open_idx + 1:-1].strip()
    return face_name, organ_name

def _get_user_id_from_request(data) -> int | None:
    """從 JSON 或 Header 取得 user_id"""
    # 先看 JSON body
    uid = data.get("user_id")
    if uid is None:
        # 再看 header
        uid = request.headers.get("X-User-Id")
    try:
        return int(uid) if uid is not None and str(uid).strip() != "" else None
    except Exception:
        return None


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
        # 正確的請求驗證
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "請求格式錯誤",
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
            }), 400

        data = request.get_json()
        user_id = _get_user_id_from_request(data)
        print(f"user_id = {user_id}")
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "缺少image字段",
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
            }), 400

        image_data = data['image']
        remove_moles = data.get('remove_moles', False)
        remove_beard = data.get('remove_beard', False)

        print(f"🔷 Blueprint: 圖片數據長度: {len(image_data)}")
        print(f"🔷 Blueprint: 移除痣: {remove_moles}, 移除鬍鬚: {remove_beard}")

        # 檢查分析服務
        if analysis_service is None:
            print("❌ Blueprint: 分析服務不可用")
            return jsonify({
                "success": False,
                "error": "分析服務未正確安裝",
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
            })

        # 執行分析
        try:
            print("🚀 Blueprint: 開始分析...")
            processed_image_data = image_data

            # 可選：鬍鬚移除
            if remove_beard:
                print("🧔 Blueprint: 開始移除鬍鬚...")
                beard_removed_image = process_beard_removal(image_data)
                if beard_removed_image:
                    processed_image_data = beard_removed_image
                    print("✅ Blueprint: 鬍鬚移除完成")
                else:
                    print("⚠️ Blueprint: 鬍鬚移除失敗，使用原始圖像")

            # 優先使用新分析器
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
                print(f"⚠️ Blueprint: 分析失敗: {result.get('error', '未知錯誤')}")
                return jsonify({
                    "success": False,
                    "error": result.get('error', '分析失敗'),
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
                })

            # 沒移除鬍鬚就做檢測
            beard_detection_result = {'has_beard': False, 'beard_count': 0}
            if not remove_beard:
                print("🔍 Blueprint: 檢測鬍鬚特徵...")
                beard_detection_result = detect_beard_features(image_data)
                print(f"🧔 Blueprint: 鬍鬚檢測結果: {beard_detection_result}")

            # ✅ 取得逐區域建議（批次）：由新 service 封裝（三段式）
            diagnosis_text, per_region_advices, advice_sources = advise_for_regions(
                region_results=result.get("region_results", {}),
                overall_color=result.get("overall_color"),
                has_moles=result.get("has_moles", False),
                has_beard=beard_detection_result.get("has_beard", False) if not remove_beard else False
            )

            # 準備響應數據
            response_data = {
                "success": True,
                "error": None,
                "abnormal_count": result.get("abnormal_count", 0),
                "overall_color": result.get("overall_color", None),
                "all_region_results": result.get("all_region_results", {}),
                "region_results": result.get("region_results", {}),
                "diagnoses": per_region_advices,                # ← 每個區域一段建議（三段式）
                "diagnosis_text": diagnosis_text,               # ← 拼接總結
                # 痣檢測相關欄位
                "has_moles": result.get("has_moles", False),
                "mole_analysis": result.get("mole_analysis", {"mole_count": 0, "total_moles": 0}),
                "moles_removed": result.get("moles_removed", remove_moles),
                # 鬍鬚檢測相關欄位
                "has_beard": beard_detection_result['has_beard'] if not remove_beard else False,
                "beard_analysis": {
                    "beard_count": beard_detection_result['beard_count'] if not remove_beard else 0,
                    "has_beard": beard_detection_result['has_beard'] if not remove_beard else False
                },
                "beard_removed": remove_beard,
                # 參考來源（前端需要可顯示）
                "advice_sources": advice_sources,
            }

            print(f"📊 Blueprint: 響應數據準備完成")
            print(f"   - 異常區域: {response_data['abnormal_count']}")
            print(f"   - 檢測到痣: {response_data['has_moles']}")
            print(f"   - 檢測到鬍鬚: {response_data['has_beard']}")
            print(f"   - 建議文字: {response_data['diagnosis_text'][:80]}{'...' if len(response_data['diagnosis_text'])>80 else ''}")

            # =========================
            # 寫入 face_analysis（逐區域一列）
            # =========================
            try:
                # 只存異常；若你要「全部區域都存」，把下一行改成 all_region_results
                area_map = result.get("region_results") or {}
                now = datetime.utcnow()

                if area_map:
                    with SessionLocal() as db:
                        face_map, organ_map, status_map = load_code_maps(db)

                        rows = []
                        for area, status_str in area_map.items():
                            face_name, organ_name = split_area_label_exact(area)  # 精準拆分
                            status_name = (status_str or "未知").strip()

                            face_code = face_map.get(face_name)
                            organ_code = organ_map.get(organ_name)
                            status_code = status_map.get(status_name)

                            # 找不到代碼時，採寬鬆回退策略並印出警告
                            if not face_code:
                                print(f"[warn] face 對不到 code_id：{face_name}")
                            if not organ_code:
                                print(f"[warn] organ 對不到 code_id：{organ_name}")
                            if not status_code:
                                print(f"[warn] status 對不到 code_id：{status_name}")

                            rows.append(FaceAnalysis(
                                user_id=user_id,
                                face=face_code or face_name[:5],  # 先寫 code_id（字串）；對不到則寫名稱前 5 字
                                organ=organ_code or organ_name[:5],
                                status=status_code or status_name[:5],
                                analysis_date=now
                            ))

                        if rows:
                            db.add_all(rows)
                            db.commit()
                            print(f"🗄️ Blueprint: face_analysis 已寫入 {len(rows)} 筆（已轉成 sys_code code_id）")
                else:
                    print("ℹ️ Blueprint: 本次沒有可寫入的區域資料。")

            except Exception as e:
                print(f"⚠️ Blueprint: 寫入 face_analysis 失敗：{e}")

            return jsonify(response_data)

        except Exception as e:
            print(f"❌ Blueprint: 分析執行錯誤: {e}")
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"分析失敗: {str(e)}",
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
            })

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

        # 嘗試新分析器
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

        # 鬍鬚檢測
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

        # ✅ RAG 批次建議（三段式）
        diagnosis_text, per_region_advices, advice_sources = advise_for_regions(
            region_results=result.get("region_results", {}),
            overall_color=result.get("overall_color"),
            has_moles=result.get("has_moles", False),
            has_beard=result.get("has_beard", False)
        )
        result["diagnosis_text"] = diagnosis_text
        result["diagnoses"] = per_region_advices
        result["advice_sources"] = advice_sources

        return jsonify(result)

    except Exception as e:
        print(f"❌ analyze_face錯誤: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'分析失敗: {str(e)}'}), 500


@face_analysis_bp.route('/detect_features', methods=['POST'])
def detect_features():
    """專門的特徵檢測端點（只檢測，不分析膚色）"""
    try:
        data = request.get_json()
        user_id = _get_user_id_from_request(data)
        print(f"analyze_face user_id = {user_id}")
        base64_image = data.get('image')

        if not base64_image:
            return jsonify({'success': False, 'error': '缺少圖像數據'}), 400

        # 檢測痣
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

        # 檢測鬍鬚
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
