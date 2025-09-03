from flask import Blueprint, request, jsonify
import traceback
import base64
import numpy as np
import cv2
from app.db import SessionLocal
from app.models.face_analysis_model import FaceAnalysis

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


def image_to_base64(image):
    """將OpenCV圖像轉換為base64字符串"""
    try:
        _, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"圖像轉Base64失敗: {e}")
        return None


def detect_beard_features(image_data):
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


def process_beard_removal(image_data):
    """處理鬍鬚移除"""
    try:
        if not mole_detection_service:
            return None

        image = base64_to_image(image_data)
        if image is None:
            return None

        # 移除鬍鬚
        processed_image, beard_info = remove_beard_from_image(image)

        # 將處理後的圖像轉換回base64
        processed_base64 = image_to_base64(processed_image)

        return processed_base64

    except Exception as e:
        print(f"鬍鬚移除錯誤: {e}")
        return None


@face_analysis_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "analysis_service": analysis_service is not None,
        "diagnosis_service": diagnosis_service is not None,
        "mole_detection_service": mole_detection_service is not None
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
        remove_beard = data.get('remove_beard', False)  # 新增鬍鬚移除參數

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

        # 執行分析（使用新的分析器）
        try:
            print("🚀 Blueprint: 開始分析...")

            # 處理圖像預處理
            processed_image_data = image_data

            # 如果需要移除鬍鬚，先處理鬍鬚
            if remove_beard:
                print("🧔 Blueprint: 開始移除鬍鬚...")
                beard_removed_image = process_beard_removal(image_data)
                if beard_removed_image:
                    processed_image_data = beard_removed_image
                    print("✅ Blueprint: 鬍鬚移除完成")
                else:
                    print("⚠️ Blueprint: 鬍鬚移除失敗，使用原始圖像")

            # 使用FaceSkinAnalyzer的新方法
            try:
                analyzer = FaceSkinAnalyzer()
                result = analyzer.analyze_from_base64(processed_image_data, remove_moles, remove_beard)
            except Exception as analyzer_error:
                print(f"⚠️ Blueprint: 使用新分析器失敗，回退到舊分析器: {analyzer_error}")
                # 回退到舊版本分析
                result = analysis_service(processed_image_data)
                # 手動添加痣和鬍鬚檢測相關欄位
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

            # 如果沒有移除鬍鬚，檢測鬍鬚特徵
            beard_detection_result = {'has_beard': False, 'beard_count': 0}
            if not remove_beard:
                print("🔍 Blueprint: 檢測鬍鬚特徵...")
                beard_detection_result = detect_beard_features(image_data)
                print(f"🧔 Blueprint: 鬍鬚檢測結果: {beard_detection_result}")

            # 準備響應數據（包含新欄位）
            response_data = {
                "success": True,
                "error": None,
                "abnormal_count": result.get("abnormal_count", 0),
                "overall_color": result.get("overall_color", None),
                "all_region_results": result.get("all_region_results", {}),
                "region_results": result.get("region_results", {}),
                "diagnoses": {},
                "diagnosis_text": "",

                # 痣檢測相關欄位
                "has_moles": result.get("has_moles", False),
                "mole_analysis": result.get("mole_analysis", {
                    "mole_count": 0,
                    "total_moles": 0
                }),
                "moles_removed": result.get("moles_removed", remove_moles),

                # 新增：鬍鬚檢測相關欄位
                "has_beard": beard_detection_result['has_beard'] if not remove_beard else False,
                "beard_analysis": {
                    "beard_count": beard_detection_result['beard_count'] if not remove_beard else 0,
                    "has_beard": beard_detection_result['has_beard'] if not remove_beard else False
                },
                "beard_removed": remove_beard,

                # 圖片相關欄位
                #"annotated_image": result.get("annotated_image"),
                #"original_image": result.get("original_image"),
            }

            print(f"📊 Blueprint: 響應數據準備完成")
            print(f"   - 異常區域: {response_data['abnormal_count']}")
            print(f"   - 檢測到痣: {response_data['has_moles']}")
            print(f"   - 檢測到鬍鬚: {response_data['has_beard']}")
            print(f"   - 痣已移除: {response_data['moles_removed']}")
            print(f"   - 鬍鬚已移除: {response_data['beard_removed']}")

            # =========================
            # 分析成功後，寫入 face_analysis 資料表
            # =========================
            try:
                # 取得「異常器官」清單（region_results 已是只有異常）
                abnormal_map = result.get("region_results") or {}
                abnormal_organs = list(abnormal_map.keys())  # 例: ["肺","肝","胃"]

                # 取得「正常器官」清單（用 all_region_results 扣掉異常），若沒有就存 None
                all_map = result.get("all_region_results") or {}
                normal_organs = None
                if all_map:
                    normal_organs = [k for k in all_map.keys() if k not in abnormal_map]

                # 從請求取得 user_id（沒有登入就存 None；可依你實作改來源）
                raw_uid = request.headers.get("X-User-Id")  # 例如 App 端送上來
                user_id = int(raw_uid) if raw_uid and raw_uid.isdigit() else None

                # 寫入 MySQL（organs/normal_organs 都是 JSON 欄位，直接塞 list）
                if abnormal_organs or normal_organs:
                    with SessionLocal() as db:
                        row = FaceAnalysis(
                            user_id=user_id,
                            organs=abnormal_organs if abnormal_organs else [],  # 至少存空陣列，避免 NULL
                            normal_organs=normal_organs  # 允許 None
                        )
                        db.add(row)
                        db.commit()
                        print(f"🗄️ Blueprint: 已寫入 face_analysis，id={row.id}")
                else:
                    print("ℹ️ Blueprint: 本次沒有可寫入的器官清單（異常/正常皆空）。")

            except Exception as e:
                # 寫庫失敗時只記錄，不阻擋 API 回傳
                print(f"⚠️ Blueprint: 寫入 face_analysis 失敗：{e}")
            # =========================

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
        remove_beard = data.get('remove_beard', False)  # 新增鬍鬚移除參數

        if not base64_image:
            return jsonify({'success': False, 'error': '缺少圖像數據'}), 400

        print(f"🔍 analyze_face: 移除痣={remove_moles}, 移除鬍鬚={remove_beard}")

        # 處理圖像預處理
        processed_image_data = base64_image

        # 如果需要移除鬍鬚，先處理鬍鬚
        if remove_beard:
            print("🧔 analyze_face: 開始移除鬍鬚...")
            beard_removed_image = process_beard_removal(base64_image)
            if beard_removed_image:
                processed_image_data = beard_removed_image
                print("✅ analyze_face: 鬍鬚移除完成")

        # 嘗試使用新的分析器
        try:
            analyzer = FaceSkinAnalyzer()
            result = analyzer.analyze_from_base64(processed_image_data, remove_moles, remove_beard)
        except Exception as e:
            print(f"❌ analyze_face: 新分析器執行失敗: {e}")
            # 如果新分析器失敗，使用舊的分析方式
            if analysis_service:
                result = analysis_service(processed_image_data)
                # 手動添加痣和鬍鬚檢測相關欄位
                result["has_moles"] = False
                result["has_beard"] = False
                result["mole_analysis"] = {"mole_count": 0, "total_moles": 0}
                result["beard_analysis"] = {"beard_count": 0}
                result["moles_removed"] = False
                result["beard_removed"] = False
            else:
                return jsonify({'success': False, 'error': '分析服務不可用'}), 500

        # 如果沒有移除鬍鬚，檢測鬍鬚特徵
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

        # 添加處理狀態標記
        result["beard_removed"] = remove_beard

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