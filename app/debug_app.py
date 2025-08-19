# simple_app.py - 簡化版啟動文件

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback

# 🔧 確保路徑正確
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 🔧 導入診斷映射服務
diagnosis_service_available = False
try:
    from app.routes.diagnosis_mapping_route import (
        get_diagnosis,
        get_all_diagnoses,
        format_diagnosis_text
    )

    print("✅ 成功導入診斷映射服務")
    diagnosis_service_available = True
except ImportError:
    try:
        from diagnosis_mapping_route import (
            get_diagnosis,
            get_all_diagnoses,
            format_diagnosis_text
        )

        print("✅ 成功導入診斷映射服務 (方法2)")
        diagnosis_service_available = True
    except ImportError as e:
        print(f"⚠️ 診斷映射服務導入失敗: {e}")

# 🔧 導入分析服務
analysis_service_available = False
try:
    from app.services.facialskincoloranalysis_service import analyze_face_from_base64

    print("✅ 成功導入分析服務")
    analysis_service_available = True
except ImportError as e:
    print(f"⚠️ 分析服務導入失敗: {e}")


@app.route('/')
def index():
    return {
        "message": "人臉膚色分析 API",
        "version": "2.0",
        "status": "running",
        "services": {
            "analysis": analysis_service_available,
            "diagnosis": diagnosis_service_available
        },
        "endpoints": {
            "health": "/api/face/health",
            "upload": "/api/face/upload",
            "analyze": "/api/face/analyze"
        }
    }


@app.route('/api/face/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "analysis_service": analysis_service_available,
        "diagnosis_service": diagnosis_service_available
    })


@app.route('/api/face/upload', methods=['POST'])
def upload_and_analyze():
    print("\n" + "=" * 60)
    print("🔍 收到新的分析請求")

    try:
        # 基礎驗證
        if not request.is_json:
            return jsonify({"success": False, "error": "請求格式錯誤"}), 400

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "缺少image字段"}), 400

        image_data = data['image']
        print(f"📷 圖片數據長度: {len(image_data)}")

        # 檢查分析服務
        if not analysis_service_available:
            return jsonify({
                "success": False,
                "error": "分析服務不可用"
            }), 503

        # 執行分析
        try:
            print("🚀 開始分析...")
            result = analyze_face_from_base64(image_data)
            print(f"✅ 分析完成，成功: {result.get('success', False)}")

            if result.get('success', False):
                print(f"📊 異常區域數量: {result.get('abnormal_count', 0)}")

                # 🎯 添加診斷服務處理
                region_results = result.get("region_results", {})
                diagnoses = {}
                diagnosis_text = ""

                if diagnosis_service_available and region_results:
                    print(f"🔬 開始處理診斷，異常區域: {list(region_results.keys())}")

                    try:
                        diagnoses = get_all_diagnoses(region_results)
                        diagnosis_text = format_diagnosis_text(diagnoses)

                        print(f"📋 診斷完成，生成了 {len(diagnoses)} 個診斷")
                        print(f"📝 診斷文字長度: {len(diagnosis_text)}")
                    except Exception as diag_error:
                        print(f"❌ 診斷處理失敗: {diag_error}")
                        diagnosis_text = "診斷服務暫時不可用"

                elif not diagnosis_service_available:
                    print("⚠️ 診斷服務不可用，跳過診斷步驟")
                    diagnosis_text = "診斷服務暫時不可用"
                else:
                    print("ℹ️ 沒有異常區域，無需診斷")
                    diagnosis_text = "所有檢測區域膚色狀態正常，身體狀況良好"

                # 構建響應數據
                response_data = {
                    "success": True,
                    "error": None,
                    "abnormal_count": result.get("abnormal_count", 0),
                    "overall_color": result.get("overall_color", None),
                    "all_region_results": result.get("all_region_results", {}),
                    "region_results": region_results,
                    "diagnoses": diagnoses,
                    "diagnosis_text": diagnosis_text,
                    # 不返回圖片數據以避免傳輸問題
                    "original_image": None,
                    "annotated_image": None,
                    "abnormal_only_image": None,
                    "grid_analysis": None
                }

                return jsonify(response_data)

            else:
                print(f"⚠️ 分析失敗: {result.get('error', '未知錯誤')}")
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

        except Exception as analysis_error:
            print(f"❌ 分析執行失敗: {str(analysis_error)}")
            print("📜 錯誤詳情:")
            traceback.print_exc()

            return jsonify({
                "success": False,
                "error": f"分析失敗: {str(analysis_error)}",
                "error_type": type(analysis_error).__name__
            }), 500

    except Exception as e:
        print(f"❌ 整體錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"服務器錯誤: {str(e)}"
        }), 500


@app.route('/api/face/test-diagnosis', methods=['POST'])
def test_diagnosis():
    """測試診斷服務功能"""
    try:
        data = request.get_json()
        if not data or 'region_results' not in data:
            return jsonify({
                "success": False,
                "error": "請提供 region_results 數據進行測試"
            }), 400

        region_results = data['region_results']

        if diagnosis_service_available:
            diagnoses = get_all_diagnoses(region_results)
            diagnosis_text = format_diagnosis_text(diagnoses)

            return jsonify({
                "success": True,
                "message": "診斷服務測試成功",
                "input_regions": region_results,
                "diagnoses": diagnoses,
                "diagnosis_text": diagnosis_text
            })
        else:
            return jsonify({
                "success": False,
                "error": "診斷服務不可用"
            })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"測試失敗: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("🚀 啟動人臉分析 API")
    print("📍 端口: 6060")
    print(f"🔬 診斷服務: {'可用' if diagnosis_service_available else '不可用'}")
    print(f"🔍 分析服務: {'可用' if analysis_service_available else '不可用'}")

    # 測試診斷服務
    if diagnosis_service_available:
        try:
            test_regions = {"心": "發紅", "肺": "發黑"}
            test_diagnoses = get_all_diagnoses(test_regions)
            test_text = format_diagnosis_text(test_diagnoses)
            print("🧪 診斷服務測試:")
            print(f"   輸入: {test_regions}")
            print(f"   診斷: {len(test_diagnoses)} 個結果")
            print(f"   文字: {test_text[:100]}...")
        except Exception as e:
            print(f"⚠️ 診斷服務測試失敗: {e}")

    app.run(host='0.0.0.0', port=6060, debug=True)