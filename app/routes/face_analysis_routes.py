from flask import Blueprint, request, jsonify
import traceback
from app.db import SessionLocal
from app.models.face_analysis_model import FaceAnalysis

# 創建Blueprint
face_analysis_bp = Blueprint('face_analysis', __name__)

# 導入服務
analysis_service = None
diagnosis_service = None

try:
    from app.services.facialskincoloranalysis_service import analyze_face_from_base64
    analysis_service = analyze_face_from_base64
    print("✅ Blueprint: 成功導入分析服務")
except ImportError as e:
    print(f"⚠️ Blueprint: 分析服務導入失敗: {e}")

try:
    from app.routes.diagnosis_mapping_route import (
        get_all_diagnoses,
        format_diagnosis_text
    )
    diagnosis_service = {
        'get_all_diagnoses': get_all_diagnoses,
        'format_diagnosis_text': format_diagnosis_text
    }
    print("✅ Blueprint: 成功導入診斷服務")
except ImportError as e:
    print(f"⚠️ Blueprint: 診斷服務導入失敗: {e}")


@face_analysis_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "analysis_service": analysis_service is not None,
        "diagnosis_service": diagnosis_service is not None
    })


@face_analysis_bp.route('/upload', methods=['POST'])
def upload_and_analyze():
    print("\n" + "=" * 60)
    print("🔍 Blueprint: 收到新的分析請求")

    try:
        # ✅ 正確的請求驗證 - 與工作版本保持一致
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "請求格式錯誤",
                "abnormal_count": 0,
                "overall_color": None,
                "all_region_results": {},
                "region_results": {},
                "diagnoses": {},
                "diagnosis_text": ""
            }), 400

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "缺少image字段",  # ✅ 與工作版本一致的錯誤信息
                "abnormal_count": 0,
                "overall_color": None,
                "all_region_results": {},
                "region_results": {},
                "diagnoses": {},
                "diagnosis_text": ""
            }), 400

        image_data = data['image']
        print(f"📷 Blueprint: 圖片數據長度: {len(image_data)}")

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
                "diagnosis_text": ""
            })

        # 執行分析 - 與工作版本完全相同的邏輯
        try:
            print("🚀 Blueprint: 開始分析...")
            result = analysis_service(image_data)
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
                    "diagnosis_text": ""
                })

            # 準備響應數據 - 與工作版本完全相同
            response_data = {
                "success": True,
                "error": None,
                "abnormal_count": result.get("abnormal_count", 0),
                "overall_color": result.get("overall_color", None),
                "all_region_results": result.get("all_region_results", {}),
                "region_results": result.get("region_results", {}),
                "diagnoses": {},
                "diagnosis_text": "",
                "original_image": None,
                "annotated_image": None,
                "abnormal_only_image": None,
                "grid_analysis": {
                    "grid_image": None,
                    "dark_blocks_image": None
                }
            }

            # 添加診斷（與工作版本相同的邏輯）
            region_results = result.get("region_results", {})
            if diagnosis_service and region_results:
                try:
                    print(f"🔬 Blueprint: 開始診斷，異常區域: {list(region_results.keys())}")
                    diagnoses = diagnosis_service['get_all_diagnoses'](region_results)
                    diagnosis_text = diagnosis_service['format_diagnosis_text'](diagnoses)

                    response_data["diagnoses"] = diagnoses
                    response_data["diagnosis_text"] = diagnosis_text
                    print(f"📋 Blueprint: 診斷完成，生成 {len(diagnoses)} 個診斷")
                except Exception as e:
                    print(f"❌ Blueprint: 診斷失敗: {e}")
                    response_data["diagnosis_text"] = "診斷服務暫時不可用"
            elif not region_results:
                response_data["diagnosis_text"] = "所有檢測區域膚色狀態正常，身體狀況良好"
            else:
                print("⚠️ Blueprint: 診斷服務不可用")
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
                "diagnosis_text": ""
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
            "diagnosis_text": ""
        }), 500


@face_analysis_bp.route('/analyze', methods=['POST'])
def analyze():
    """與 /upload 相同的功能，為了兼容性"""
    return upload_and_analyze()


@face_analysis_bp.route('/test-diagnosis', methods=['POST'])
def test_diagnosis():
    """測試診斷服務功能"""
    if not diagnosis_service:
        return jsonify({
            "success": False,
            "error": "診斷服務不可用"
        })

    try:
        data = request.get_json()
        if not data or 'region_results' not in data:
            test_regions = {"心": "發紅", "肺": "發黑", "肝": "發綠"}
        else:
            test_regions = data['region_results']

        diagnoses = diagnosis_service['get_all_diagnoses'](test_regions)
        diagnosis_text = diagnosis_service['format_diagnosis_text'](diagnoses)

        return jsonify({
            "success": True,
            "input_regions": test_regions,
            "diagnoses": diagnoses,
            "diagnosis_text": diagnosis_text
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"測試失敗: {str(e)}"
        }), 500