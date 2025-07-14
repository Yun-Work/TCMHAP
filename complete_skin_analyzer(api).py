import cv2
import numpy as np
from PIL import Image
import base64
import io
from enum import Enum
from typing import Dict, Tuple, List, Optional, Union
import json


#使用方法
    #1. 完整分析：TCMFaceAnalysisIntegrator().analyze_patient_face(base64_image)
    #2. 簡化分析：SimpleFaceAnalysisAPI().analyze(base64_image)
    #3. 基礎分析：FaceSkinAnalyzerAPI().analyze_face_from_base64(base64_image)


class FaceRegion(Enum):
    """面部區域定義"""
    FOREHEAD_TOP = "額頭上區"
    FOREHEAD_MIDDLE = "額頭中區"
    FOREHEAD_BOTTOM = "額頭下區"
    LEFT_CHEEK = "左臉頰"
    RIGHT_CHEEK = "右臉頰"
    CHIN = "下巴"
    NOSE = "鼻尖"
    PHILTRUM = "人中"
    RIGHT_NOSE_WING = "右鼻翼"
    LEFT_NOSE_WING = "左鼻翼"
    NOSE_TIP = "鼻頭(脾)"
    NOSE_ROOT = "鼻根(肺)"


class SkinCondition(Enum):
    """膚色狀態定義"""
    NORMAL = "正常"
    DARK = "發黑"
    RED = "發紅"
    PALE = "發白"
    YELLOW = "發黃"
    CYAN = "發青"


class FaceSkinAnalyzerAPI:
    """面部膚色分析API類別"""

    def __init__(self):
        self.face_cascade = None
        self.diagnosis_results = {}
        self.face_regions = {}
        self.current_face_rect = None
        self.init_face_detector()

    def init_face_detector(self):
        """初始化人臉檢測器"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("無法載入人臉檢測模型")
            return True
        except Exception as e:
            print(f"人臉檢測器初始化失敗: {e}")
            return False

    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """將base64字符串轉換為OpenCV圖像"""
        try:
            # 移除data:image前綴（如果存在）
            if 'base64,' in base64_string:
                base64_string = base64_string.split('base64,')[1]

            # 解碼base64
            image_data = base64.b64decode(base64_string)

            # 轉換為PIL圖像
            pil_image = Image.open(io.BytesIO(image_data))

            # 轉換為RGB格式（如果是RGBA）
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')

            # 轉換為numpy陣列
            image_array = np.array(pil_image)

            # 轉換為BGR格式（OpenCV格式）
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            return image_bgr

        except Exception as e:
            raise ValueError(f"base64圖像解碼失敗: {str(e)}")

    def image_to_base64(self, image: np.ndarray, format: str = 'JPEG') -> str:
        """將OpenCV圖像轉換為base64字符串"""
        try:
            # 轉換顏色格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 轉換為PIL圖像
            pil_image = Image.fromarray(image_rgb)

            # 儲存到BytesIO
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=95)

            # 編碼為base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return f"data:image/{format.lower()};base64,{image_base64}"

        except Exception as e:
            raise ValueError(f"圖像轉base64編碼失敗: {str(e)}")

    def detect_faces(self, image):
        """檢測人臉"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(100, 100)
        )

        return faces

    def get_largest_face(self, faces):
        """獲取最大的人臉區域"""
        if len(faces) == 0:
            return None

        largest_face = faces[0]
        max_area = largest_face[2] * largest_face[3]

        for face in faces:
            area = face[2] * face[3]
            if area > max_area:
                max_area = area
                largest_face = face

        return largest_face

    def define_face_regions(self, face_rect):
        """定義面部各個區域"""
        x, y, w, h = face_rect
        regions = {}

        # 額頭區域 - 分三等份
        forehead_height = h // 3
        forehead_segment_height = forehead_height // 3

        # 額頭上部
        regions[FaceRegion.FOREHEAD_TOP] = (
            x + w // 4, y, w // 2, forehead_segment_height
        )

        # 額頭中部
        regions[FaceRegion.FOREHEAD_MIDDLE] = (
            x + w // 4, y + forehead_segment_height, w // 2, forehead_segment_height
        )

        # 額頭下部
        regions[FaceRegion.FOREHEAD_BOTTOM] = (
            x + w // 4, y + 2 * forehead_segment_height, w // 2, forehead_segment_height
        )

        # 改進的臉頰區域定位
        cheek_y_start = y + h // 3 + h // 8
        cheek_height = h // 4
        cheek_width = w // 4

        # 左臉頰
        regions[FaceRegion.LEFT_CHEEK] = (
            x + w // 8, cheek_y_start, cheek_width, cheek_height
        )

        # 右臉頰
        regions[FaceRegion.RIGHT_CHEEK] = (
            x + w - w // 8 - cheek_width, cheek_y_start, cheek_width, cheek_height
        )

        # 鼻尖
        nose_size = w // 5
        regions[FaceRegion.NOSE] = (
            x + w // 2 - nose_size // 2, y + h // 2 - nose_size // 2, nose_size, nose_size
        )

        # 人中區域
        philtrum_width = w // 5
        philtrum_height = h // 11
        philtrum_y = y + int(h * 0.7)
        regions[FaceRegion.PHILTRUM] = (
            x + w // 2 - philtrum_width // 2, philtrum_y,
            philtrum_width, philtrum_height
        )

        # 下巴
        chin_y_start = y + int(h * 0.9)
        chin_height = h - (chin_y_start - y)
        chin_width = w // 3
        chin_x_start = x + w // 2 - chin_width // 2

        regions[FaceRegion.CHIN] = (
            chin_x_start, chin_y_start, chin_width, chin_height
        )

        # 右鼻翼
        regions[FaceRegion.RIGHT_NOSE_WING] = (
            x + w // 2, y + h // 2 - nose_size // 2, nose_size // 2, nose_size // 2
        )

        # 左鼻翼
        regions[FaceRegion.LEFT_NOSE_WING] = (
            x + w // 2 - nose_size // 2, y + h // 2 - nose_size // 2, nose_size // 2, nose_size // 2
        )

        # 鼻頭(脾)
        regions[FaceRegion.NOSE_TIP] = (
            x + w // 2 - nose_size // 4, y + h // 2, nose_size // 2, nose_size // 2
        )

        # 鼻根(肺)
        regions[FaceRegion.NOSE_ROOT] = (
            x + w // 2 - nose_size // 4, y + h // 3, nose_size // 2, nose_size // 2
        )

        return regions

    def analyze_skin_color_for_region(self, image, region_rect):
        """分析特定區域的膚色"""
        x, y, w, h = region_rect

        # 確保區域在圖像範圍內
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)

        if w <= 0 or h <= 0:
            return (153, 134, 117)

        region = image[y:y + h, x:x + w]

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(region, (7, 7), 0)

        # 轉換到多個色彩空間進行分析
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 計算平均亮度
        avg_brightness = np.mean(hsv[:, :, 2])

        # 改進的膚色範圍檢測
        if avg_brightness < 50:
            lower_skin = np.array([0, 5, 20])
            upper_skin = np.array([40, 255, 255])
        else:
            lower_skin = np.array([0, 10, 40])
            upper_skin = np.array([40, 255, 255])

        # 進行膚色檢測
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 形態學處理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # 如果膚色檢測失敗，使用更寬鬆的條件
        if cv2.countNonZero(skin_mask) < (w * h * 0.1):
            lower_skin_loose = np.array([0, 8, 30])
            upper_skin_loose = np.array([50, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin_loose, upper_skin_loose)

            if cv2.countNonZero(skin_mask) == 0:
                skin_mask = np.ones((h, w), dtype=np.uint8) * 255

        # 提取膚色區域並計算平均顏色
        skin_region = cv2.bitwise_and(blurred, blurred, mask=skin_mask)
        mean_color = cv2.mean(skin_region, skin_mask)

        return mean_color[:3]

    def diagnose_skin_condition(self, mean_color):
        """根據RGB值判斷膚色狀態"""
        b, g, r = mean_color

        # 計算亮度和色彩特徵
        brightness = (r + g + b) / 3.0
        max_color = max(r, g, b)
        min_color = min(r, g, b)

        # 計算色彩飽和度
        saturation = (max_color - min_color) / max_color if max_color > 0 else 0

        # 計算各顏色分量比例
        total_color = r + g + b
        if total_color > 0:
            red_ratio = r / total_color
            green_ratio = g / total_color
            blue_ratio = b / total_color
        else:
            red_ratio = green_ratio = blue_ratio = 0.33

        # 判斷膚色狀態
        if brightness < 70:
            return SkinCondition.DARK
        elif brightness > 200 and min_color > 150 and saturation < 0.1:
            return SkinCondition.PALE
        elif red_ratio > 0.42 and r > 150:
            return SkinCondition.RED
        elif (green_ratio > red_ratio and green_ratio > blue_ratio and
              green_ratio > 0.38 and g > 130):
            if red_ratio > 0.3:
                return SkinCondition.YELLOW
        elif (blue_ratio > red_ratio and blue_ratio > green_ratio and
              blue_ratio > 0.38 and b > 130):
            if green_ratio > 0.25:
                return SkinCondition.CYAN
        else:
            return SkinCondition.NORMAL

    def get_organ_diagnosis(self, region, condition):
        """根據面部區域和狀態提供相關器官健康診斷"""
        diagnosis_map = {
            FaceRegion.FOREHEAD_TOP: {
                SkinCondition.DARK: "額頭上區發黑：可能與腦部供血、膽囊功能相關",
                SkinCondition.RED: "額頭上區發紅：可能與腦部血管擴張、頭痛相關",
                SkinCondition.PALE: "額頭上區發白：可能與腦部供血不足相關",
                SkinCondition.YELLOW: "額頭上區發黃：可能與膽囊功能異常、黃疸相關",
                SkinCondition.CYAN: "額頭上區發青：可能與氧氣供應不足、循環不良相關"
            },
            FaceRegion.FOREHEAD_MIDDLE: {
                SkinCondition.DARK: "額頭中區發黑：可能與肝膽溼熱、腸胃功能相關",
                SkinCondition.RED: "額頭中區發紅：可能與心火上炎、高血壓相關",
                SkinCondition.PALE: "額頭中區發白：可能與氣血不足、疲勞相關",
                SkinCondition.YELLOW: "額頭中區發黃：可能與肝膽功能異常相關",
                SkinCondition.CYAN: "額頭中區發青：可能與心肺功能不良相關"
            },
            FaceRegion.FOREHEAD_BOTTOM: {
                SkinCondition.DARK: "額頭下區發黑：可能與小腸機能、消化系統相關",
                SkinCondition.RED: "額頭下區發紅：可能與心臟負擔較重相關",
                SkinCondition.PALE: "額頭下區發白：可能與免疫力下降相關",
                SkinCondition.YELLOW: "額頭下區發黃：可能與消化系統功能異常相關",
                SkinCondition.CYAN: "額頭下區發青：可能與心臟功能不良相關"
            },
            FaceRegion.LEFT_CHEEK: {
                SkinCondition.DARK: "左臉頰發黑：可能與肝臟功能相關",
                SkinCondition.RED: "左臉頰發紅：可能與肝火上升相關",
                SkinCondition.PALE: "左臉頰發白：可能與肝臟血流不足相關",
                SkinCondition.YELLOW: "左臉頰發黃：可能與肝臟功能異常、黃疸相關",
                SkinCondition.CYAN: "左臉頰發青：可能與肝血循環不良相關"
            },
            FaceRegion.RIGHT_CHEEK: {
                SkinCondition.DARK: "右臉頰發黑：可能與肺部功能相關",
                SkinCondition.RED: "右臉頰發紅：可能與肺熱或呼吸系統發炎相關",
                SkinCondition.PALE: "右臉頰發白：可能與肺氣虛弱相關",
                SkinCondition.YELLOW: "右臉頰發黃：可能與肺部功能異常相關",
                SkinCondition.CYAN: "右臉頰發青：可能與肺部缺氧、呼吸功能不良相關"
            },
            FaceRegion.NOSE: {
                SkinCondition.DARK: "鼻尖發黑：可能與脾胃功能相關",
                SkinCondition.RED: "鼻尖發紅：可能與胃熱或消化系統問題相關",
                SkinCondition.PALE: "鼻尖發白：可能與脾胃氣血不足相關",
                SkinCondition.YELLOW: "鼻尖發黃：可能與脾胃濕熱相關",
                SkinCondition.CYAN: "鼻尖發青：可能與脾胃虚寒相關"
            },
            FaceRegion.PHILTRUM: {
                SkinCondition.DARK: "人中發黑：可能與生殖系統、荷爾蒙失調相關",
                SkinCondition.RED: "人中發紅：可能與生殖系統發炎或熱症相關",
                SkinCondition.PALE: "人中發白：可能與生殖系統功能低下相關",
                SkinCondition.YELLOW: "人中發黃：可能與生殖系統濕熱相關",
                SkinCondition.CYAN: "人中發青：可能與生殖系統虚寒相關"
            },
            FaceRegion.CHIN: {
                SkinCondition.DARK: "下巴發黑：可能與腎臟或泌尿系統功能相關",
                SkinCondition.RED: "下巴發紅：可能與內分泌失調或腎臟問題相關",
                SkinCondition.PALE: "下巴發白：可能與腎氣不足相關",
                SkinCondition.YELLOW: "下巴發黃：可能與腎臟代謝功能異常相關",
                SkinCondition.CYAN: "下巴發青：可能與腎陽虚弱相關"
            },
            FaceRegion.RIGHT_NOSE_WING: {
                SkinCondition.DARK: "右鼻翼發黑：可能與右肺功能相關",
                SkinCondition.RED: "右鼻翼發紅：可能與右肺熱或發炎相關",
                SkinCondition.PALE: "右鼻翼發白：可能與右肺氣虚相關",
                SkinCondition.YELLOW: "右鼻翼發黃：可能與右肺功能異常相關",
                SkinCondition.CYAN: "右鼻翼發青：可能與右肺缺氧相關"
            },
            FaceRegion.LEFT_NOSE_WING: {
                SkinCondition.DARK: "左鼻翼發黑：可能與左肺功能相關",
                SkinCondition.RED: "左鼻翼發紅：可能與左肺熱或發炎相關",
                SkinCondition.PALE: "左鼻翼發白：可能與左肺氣虚相關",
                SkinCondition.YELLOW: "左鼻翼發黃：可能與左肺功能異常相關",
                SkinCondition.CYAN: "左鼻翼發青：可能與左肺缺氧相關"
            },
            FaceRegion.NOSE_TIP: {
                SkinCondition.DARK: "鼻頭(脾)發黑：可能與脾臟功能失調相關",
                SkinCondition.RED: "鼻頭(脾)發紅：可能與脾臟熱症相關",
                SkinCondition.PALE: "鼻頭(脾)發白：可能與脾虚相關",
                SkinCondition.YELLOW: "鼻頭(脾)發黃：可能與脾胃濕熱相關",
                SkinCondition.CYAN: "鼻頭(脾)發青：可能與脾胃虚寒相關"
            },
            FaceRegion.NOSE_ROOT: {
                SkinCondition.DARK: "鼻根(肺)發黑：可能與肺部功能障礙相關",
                SkinCondition.RED: "鼻根(肺)發紅：可能與肺熱或上呼吸道發炎相關",
                SkinCondition.PALE: "鼻根(肺)發白：可能與肺氣不足相關",
                SkinCondition.YELLOW: "鼻根(肺)發黃：可能與肺部功能異常相關",
                SkinCondition.CYAN: "鼻根(肺)發青：可能與肺部缺氧、呼吸功能障礙相關"
            }
        }

        return diagnosis_map.get(region, {}).get(condition, "")

    def draw_face_regions(self, image):
        """在圖像上繪製面部區域標註"""
        if not self.face_regions or self.current_face_rect is None:
            return image

        annotated_image = image.copy()

        # 定義顏色對應
        condition_colors = {
            SkinCondition.NORMAL: (0, 255, 0),
            SkinCondition.DARK: (0, 0, 139),
            SkinCondition.RED: (0, 0, 255),
            SkinCondition.PALE: (255, 255, 255),
            SkinCondition.YELLOW: (0, 255, 255),
            SkinCondition.CYAN: (255, 255, 0)
        }

        # 為每個區域繪製框線和標籤
        for region, region_rect in self.face_regions.items():
            x, y, w, h = region_rect

            condition = self.diagnosis_results.get(region, SkinCondition.NORMAL)
            color = condition_colors.get(condition, (0, 255, 0))

            # 繪製矩形框
            thickness = 3 if condition != SkinCondition.NORMAL else 2
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)

            # 添加標籤
            label_text = region.value
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            cv2.rectangle(annotated_image,
                          (x, y - text_height - 5),
                          (x + text_width + 5, y),
                          color, -1)

            text_color = (0, 0, 0) if condition in [SkinCondition.PALE, SkinCondition.YELLOW, SkinCondition.CYAN] else (
            255, 255, 255)
            cv2.putText(annotated_image, label_text, (x + 2, y - 2),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return annotated_image

    def analyze_face_from_base64(self, base64_image: str, include_annotated_image: bool = True) -> Dict:
        """
        從base64圖像分析面部膚色

        Args:
            base64_image: base64編碼的圖像字符串
            include_annotated_image: 是否包含標註圖像

        Returns:
            包含分析結果的字典
        """
        try:
            # 清空之前的結果
            self.diagnosis_results.clear()
            self.face_regions.clear()
            self.current_face_rect = None

            # 將base64轉換為圖像
            image = self.base64_to_image(base64_image)

            # 檢測人臉
            faces = self.detect_faces(image)

            if len(faces) == 0:
                return {
                    "success": False,
                    "error": "未能檢測到面部",
                    "message": "請確保臉部完整且清晰可見，光線充足且均勻，避免過暗或逆光，正對鏡頭。"
                }

            # 獲取最大的人臉區域
            largest_face = self.get_largest_face(faces)
            self.current_face_rect = largest_face

            # 定義面部區域
            self.face_regions = self.define_face_regions(largest_face)

            # 分析每個區域
            region_analysis = {}
            for region, region_rect in self.face_regions.items():
                # 分析該區域的膚色
                mean_color = self.analyze_skin_color_for_region(image, region_rect)

                # 判斷膚色狀態
                condition = self.diagnose_skin_condition(mean_color)

                # 儲存診斷結果
                self.diagnosis_results[region] = condition

                # 準備區域分析數據
                b, g, r = mean_color
                region_analysis[region.value] = {
                    "condition": condition.value,
                    "rgb": {
                        "r": int(r),
                        "g": int(g),
                        "b": int(b)
                    },
                    "hex": f"#{int(r):02X}{int(g):02X}{int(b):02X}",
                    "diagnosis": self.get_organ_diagnosis(region,
                                                          condition) if condition != SkinCondition.NORMAL else ""
                }

            # 計算整體膚色RGB值
            overall_color = self.analyze_skin_color_for_region(image, largest_face)
            b, g, r = overall_color

            # 準備返回結果
            result = {
                "success": True,
                "overall_skin_color": {
                    "rgb": {
                        "r": int(r),
                        "g": int(g),
                        "b": int(b)
                    },
                    "hex": f"#{int(r):02X}{int(g):02X}{int(b):02X}"
                },
                "regions": region_analysis,
                "summary": self._generate_summary()
            }

            # 如果需要包含標註圖像
            if include_annotated_image:
                annotated_image = self.draw_face_regions(image)
                result["annotated_image"] = self.image_to_base64(annotated_image)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "圖像分析過程中發生錯誤"
            }

    def _generate_summary(self) -> Dict:
        """生成分析摘要"""
        normal_regions = []
        abnormal_regions = []

        for region, condition in self.diagnosis_results.items():
            if condition == SkinCondition.NORMAL:
                normal_regions.append(region.value)
            else:
                abnormal_regions.append({
                    "region": region.value,
                    "condition": condition.value,
                    "diagnosis": self.get_organ_diagnosis(region, condition)
                })

        return {
            "total_regions": len(self.diagnosis_results),
            "normal_regions_count": len(normal_regions),
            "abnormal_regions_count": len(abnormal_regions),
            "normal_regions": normal_regions,
            "abnormal_regions": abnormal_regions,
            "health_status": "正常" if len(abnormal_regions) == 0 else "需要關注"
        }


# 使用範例
def example_usage():
    """使用範例"""

    # 初始化分析器
    analyzer = FaceSkinAnalyzerAPI()

    # 假設您有一個base64編碼的圖像
    # base64_image_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."

    # 分析圖像
    # result = analyzer.analyze_face_from_base64(base64_image_data)

    # 處理結果
    # if result["success"]:
    #     print("分析成功！")
    #     print(f"整體膚色: {result['overall_skin_color']}")
    #     print(f"健康狀態: {result['summary']['health_status']}")
    #
    #     # 顯示異常區域
    #     for abnormal in result['summary']['abnormal_regions']:
    #         print(f"區域: {abnormal['region']}")
    #         print(f"狀態: {abnormal['condition']}")
    #         print(f"診斷: {abnormal['diagnosis']}")
    #         print("---")
    # else:
    #     print(f"分析失敗: {result['error']}")

    pass


class TCMFaceAnalysisIntegrator:
    """中醫問診系統整合類別"""

    def __init__(self):
        self.analyzer = FaceSkinAnalyzerAPI()

    def analyze_patient_face(self, base64_image: str, patient_id: str = None) -> Dict:
        """
        分析患者面部圖像並生成中醫診斷報告

        Args:
            base64_image: 患者面部圖像的base64編碼
            patient_id: 患者ID（可選）

        Returns:
            包含完整中醫面診結果的字典
        """
        try:
            # 進行面部分析
            analysis_result = self.analyzer.analyze_face_from_base64(base64_image, include_annotated_image=True)

            if not analysis_result["success"]:
                return analysis_result

            # 生成中醫診斷報告
            tcm_report = self._generate_tcm_report(analysis_result)

            # 整合結果
            integrated_result = {
                "patient_id": patient_id,
                "analysis_timestamp": self._get_timestamp(),
                "face_analysis": analysis_result,
                "tcm_diagnosis": tcm_report,
                "recommendations": self._generate_recommendations(analysis_result)
            }

            return integrated_result

        except Exception as e:
            return {
                "success": False,
                "error": f"面診分析過程出錯: {str(e)}",
                "patient_id": patient_id
            }

    def _generate_tcm_report(self, analysis_result: Dict) -> Dict:
        """生成中醫診斷報告"""

        # 分析五臟六腑對應關係
        organ_analysis = self._analyze_organ_conditions(analysis_result["regions"])

        # 分析體質類型
        constitution_type = self._determine_constitution_type(analysis_result["regions"])

        # 生成中醫病理分析
        pathology_analysis = self._generate_pathology_analysis(analysis_result["regions"])

        return {
            "overall_assessment": self._get_overall_tcm_assessment(analysis_result),
            "organ_analysis": organ_analysis,
            "constitution_type": constitution_type,
            "pathology_analysis": pathology_analysis,
            "severity_level": self._assess_severity_level(analysis_result)
        }

    def _analyze_organ_conditions(self, regions: Dict) -> Dict:
        """分析五臟六腑狀況"""
        organ_conditions = {
            "heart": {"status": "正常", "regions": [], "symptoms": []},
            "liver": {"status": "正常", "regions": [], "symptoms": []},
            "spleen": {"status": "正常", "regions": [], "symptoms": []},
            "lung": {"status": "正常", "regions": [], "symptoms": []},
            "kidney": {"status": "正常", "regions": [], "symptoms": []},
            "stomach": {"status": "正常", "regions": [], "symptoms": []},
            "gallbladder": {"status": "正常", "regions": [], "symptoms": []},
            "intestines": {"status": "正常", "regions": [], "symptoms": []}
        }

        # 根據面部區域分析對應器官
        for region_name, region_data in regions.items():
            condition = region_data["condition"]
            if condition != "正常":
                # 心臟相關
                if "額頭" in region_name and condition in ["發紅", "發黑"]:
                    organ_conditions["heart"]["status"] = "異常"
                    organ_conditions["heart"]["regions"].append(region_name)
                    if condition == "發紅":
                        organ_conditions["heart"]["symptoms"].append("心火上炎")
                    elif condition == "發黑":
                        organ_conditions["heart"]["symptoms"].append("心血不足")

                # 肝臟相關
                if "左臉頰" in region_name:
                    organ_conditions["liver"]["status"] = "異常"
                    organ_conditions["liver"]["regions"].append(region_name)
                    if condition == "發紅":
                        organ_conditions["liver"]["symptoms"].append("肝火上升")
                    elif condition == "發黃":
                        organ_conditions["liver"]["symptoms"].append("肝膽濕熱")
                    elif condition == "發黑":
                        organ_conditions["liver"]["symptoms"].append("肝血不足")

                # 肺臟相關
                if "右臉頰" in region_name or "鼻翼" in region_name or "鼻根" in region_name:
                    organ_conditions["lung"]["status"] = "異常"
                    organ_conditions["lung"]["regions"].append(region_name)
                    if condition == "發紅":
                        organ_conditions["lung"]["symptoms"].append("肺熱")
                    elif condition == "發白":
                        organ_conditions["lung"]["symptoms"].append("肺氣虛")
                    elif condition == "發青":
                        organ_conditions["lung"]["symptoms"].append("肺部缺氧")

                # 脾胃相關
                if "鼻尖" in region_name or "鼻頭" in region_name:
                    organ_conditions["spleen"]["status"] = "異常"
                    organ_conditions["spleen"]["regions"].append(region_name)
                    organ_conditions["stomach"]["status"] = "異常"
                    organ_conditions["stomach"]["regions"].append(region_name)
                    if condition == "發紅":
                        organ_conditions["stomach"]["symptoms"].append("胃熱")
                        organ_conditions["spleen"]["symptoms"].append("脾胃熱盛")
                    elif condition == "發黃":
                        organ_conditions["spleen"]["symptoms"].append("脾胃濕熱")
                    elif condition == "發白":
                        organ_conditions["spleen"]["symptoms"].append("脾氣虛")

                # 腎臟相關
                if "下巴" in region_name:
                    organ_conditions["kidney"]["status"] = "異常"
                    organ_conditions["kidney"]["regions"].append(region_name)
                    if condition == "發黑":
                        organ_conditions["kidney"]["symptoms"].append("腎虛")
                    elif condition == "發青":
                        organ_conditions["kidney"]["symptoms"].append("腎陽虛")
                    elif condition == "發白":
                        organ_conditions["kidney"]["symptoms"].append("腎氣不足")

        return organ_conditions

    def _determine_constitution_type(self, regions: Dict) -> Dict:
        """判斷體質類型"""
        constitution_scores = {
            "平和質": 0,
            "氣虛質": 0,
            "陽虛質": 0,
            "陰虛質": 0,
            "痰濕質": 0,
            "濕熱質": 0,
            "血瘀質": 0,
            "氣鬱質": 0,
            "特稟質": 0
        }

        abnormal_count = 0
        for region_name, region_data in regions.items():
            condition = region_data["condition"]
            if condition != "正常":
                abnormal_count += 1

                if condition == "發白":
                    constitution_scores["氣虛質"] += 2
                    constitution_scores["陽虛質"] += 1
                elif condition == "發紅":
                    constitution_scores["陰虛質"] += 2
                    constitution_scores["濕熱質"] += 1
                elif condition == "發黃":
                    constitution_scores["濕熱質"] += 2
                    constitution_scores["痰濕質"] += 1
                elif condition == "發黑":
                    constitution_scores["血瘀質"] += 2
                    constitution_scores["腎虛質"] = constitution_scores.get("腎虛質", 0) + 1
                elif condition == "發青":
                    constitution_scores["陽虛質"] += 2
                    constitution_scores["血瘀質"] += 1

        # 如果沒有異常，傾向平和質
        if abnormal_count == 0:
            constitution_scores["平和質"] = 10

        # 找出得分最高的體質類型
        primary_constitution = max(constitution_scores, key=constitution_scores.get)

        # 計算信心度
        max_score = constitution_scores[primary_constitution]
        total_score = sum(constitution_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0

        return {
            "primary_type": primary_constitution,
            "confidence": round(confidence, 2),
            "scores": constitution_scores,
            "description": self._get_constitution_description(primary_constitution)
        }

    def _get_constitution_description(self, constitution_type: str) -> str:
        """獲取體質描述"""
        descriptions = {
            "平和質": "陰陽氣血調和，以體態適中、面色紅潤、精力充沛等為主要特徵",
            "氣虛質": "元氣不足，以疲乏、氣短、自汗等氣虛表現為主要特徵",
            "陽虛質": "陽氣不足，以畏寒怕冷、手足不溫等虛寒表現為主要特徵",
            "陰虛質": "陰液虧少，以口燥咽乾、手足心熱等虛熱表現為主要特徵",
            "痰濕質": "痰濕凝聚，以形體肥胖、腹部肥滿、口黏苔膩等痰濕表現為主要特徵",
            "濕熱質": "濕熱內蘊，以面垢油膩、口苦、苔黃膩等濕熱表現為主要特徵",
            "血瘀質": "血行不暢，以膚色晦暗、舌質紫暗等血瘀表現為主要特徵",
            "氣鬱質": "氣機鬱滯，以神情抑鬱、憂慮脆弱等氣鬱表現為主要特徵",
            "特稟質": "先天失常，以生理缺陷、過敏反應等為主要特徵"
        }
        return descriptions.get(constitution_type, "體質類型描述暫無")

    def _generate_pathology_analysis(self, regions: Dict) -> Dict:
        """生成病理分析"""
        pathology_patterns = {
            "熱證": 0,
            "寒證": 0,
            "虛證": 0,
            "實證": 0,
            "濕證": 0,
            "燥證": 0,
            "氣滯": 0,
            "血瘀": 0
        }

        for region_name, region_data in regions.items():
            condition = region_data["condition"]
            if condition != "正常":
                if condition == "發紅":
                    pathology_patterns["熱證"] += 1
                    pathology_patterns["實證"] += 1
                elif condition == "發白":
                    pathology_patterns["寒證"] += 1
                    pathology_patterns["虛證"] += 1
                elif condition == "發黃":
                    pathology_patterns["濕證"] += 1
                elif condition == "發黑":
                    pathology_patterns["血瘀"] += 1
                    pathology_patterns["虛證"] += 1
                elif condition == "發青":
                    pathology_patterns["寒證"] += 1
                    pathology_patterns["氣滯"] += 1

        # 判斷主要病理模式
        main_pattern = max(pathology_patterns, key=pathology_patterns.get)

        return {
            "main_pattern": main_pattern,
            "pattern_scores": pathology_patterns,
            "syndrome_differentiation": self._get_syndrome_differentiation(pathology_patterns)
        }

    def _get_syndrome_differentiation(self, patterns: Dict) -> str:
        """獲取證候分析"""
        hot_cold_score = patterns["熱證"] - patterns["寒證"]
        deficiency_excess_score = patterns["實證"] - patterns["虛證"]

        if hot_cold_score > 2:
            heat_cold = "熱證偏重"
        elif hot_cold_score < -2:
            heat_cold = "寒證偏重"
        else:
            heat_cold = "寒熱較平"

        if deficiency_excess_score > 2:
            def_exc = "實證偏重"
        elif deficiency_excess_score < -2:
            def_exc = "虛證偏重"
        else:
            def_exc = "虛實較平"

        return f"{heat_cold}，{def_exc}"

    def _get_overall_tcm_assessment(self, analysis_result: Dict) -> str:
        """獲取整體中醫評估"""
        abnormal_count = analysis_result["summary"]["abnormal_regions_count"]
        total_regions = analysis_result["summary"]["total_regions"]

        if abnormal_count == 0:
            return "面色正常，氣血調和，體質良好"
        elif abnormal_count <= 2:
            return "面色大致正常，略有小恙，建議注意調養"
        elif abnormal_count <= 4:
            return "面色顯示一定程度的健康問題，建議積極調理"
        else:
            return "面色顯示較多健康隱患，建議及時就醫並加強調養"

    def _assess_severity_level(self, analysis_result: Dict) -> str:
        """評估嚴重程度"""
        abnormal_count = analysis_result["summary"]["abnormal_regions_count"]

        if abnormal_count == 0:
            return "正常"
        elif abnormal_count <= 2:
            return "輕度"
        elif abnormal_count <= 4:
            return "中度"
        else:
            return "重度"

    def _generate_recommendations(self, analysis_result: Dict) -> Dict:
        """生成建議"""
        recommendations = {
            "lifestyle": [],
            "diet": [],
            "exercise": [],
            "herbs": [],
            "acupoints": [],
            "follow_up": ""
        }

        abnormal_regions = analysis_result["summary"]["abnormal_regions"]

        for abnormal in abnormal_regions:
            region = abnormal["region"]
            condition = abnormal["condition"]

            # 生活方式建議
            if "額頭" in region and condition == "發紅":
                recommendations["lifestyle"].append("保持充足睡眠，避免熬夜")
                recommendations["lifestyle"].append("減少精神壓力，學習放鬆技巧")

            if condition == "發白":
                recommendations["lifestyle"].append("注意保暖，避免受寒")
                recommendations["lifestyle"].append("適當增加休息時間")

            # 飲食建議
            if condition == "發紅":
                recommendations["diet"].append("清淡飲食，多吃涼性食物如綠豆、冬瓜")
                recommendations["diet"].append("避免辛辣刺激性食物")

            if condition == "發黃":
                recommendations["diet"].append("減少油膩食物，多吃健脾利濕食物")
                recommendations["diet"].append("可適量食用薏米、山藥、茯苓等")

            if condition == "發白":
                recommendations["diet"].append("適量補充溫性食物，如生薑、桂圓")
                recommendations["diet"].append("增加蛋白質攝入，如雞肉、魚類")

            # 運動建議
            if condition in ["發白", "發青"]:
                recommendations["exercise"].append("適度運動，如太極拳、八段錦")
                recommendations["exercise"].append("避免劇烈運動，循序漸進")

            if condition == "發紅":
                recommendations["exercise"].append("選擇較為緩和的運動")
                recommendations["exercise"].append("運動後注意及時補水")

            # 中藥建議（僅供參考）
            if "脾" in region and condition == "發黃":
                recommendations["herbs"].append("參考使用健脾化濕類中藥（需醫師指導）")

            if "肺" in region and condition == "發白":
                recommendations["herbs"].append("參考使用補肺益氣類中藥（需醫師指導）")

            # 穴位建議
            if "額頭" in region:
                recommendations["acupoints"].append("可按摩印堂穴、神庭穴")

            if "臉頰" in region:
                recommendations["acupoints"].append("可按摩頰車穴、下關穴")

        # 去重
        for key in recommendations:
            if isinstance(recommendations[key], list):
                recommendations[key] = list(set(recommendations[key]))

        # 隨訪建議
        severity = self._assess_severity_level(analysis_result)
        if severity == "正常":
            recommendations["follow_up"] = "建議每半年進行一次面診檢查"
        elif severity == "輕度":
            recommendations["follow_up"] = "建議1-2個月後復診，觀察改善情況"
        elif severity == "中度":
            recommendations["follow_up"] = "建議2-4週後復診，並考慮專業中醫診療"
        else:
            recommendations["follow_up"] = "建議儘快就醫，進行專業中醫診療"

        return recommendations

    def _get_timestamp(self) -> str:
        """獲取時間戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 簡化的API接口類別
class SimpleFaceAnalysisAPI:
    """簡化的面部分析API接口"""

    def __init__(self):
        self.integrator = TCMFaceAnalysisIntegrator()

    def analyze(self, base64_image: str, patient_id: str = None,
                include_image: bool = False) -> Dict:
        """
        簡化的分析接口

        Args:
            base64_image: base64編碼的面部圖像
            patient_id: 患者ID
            include_image: 是否包含標註圖像

        Returns:
            分析結果字典
        """
        try:
            result = self.integrator.analyze_patient_face(base64_image, patient_id)

            if not include_image and "face_analysis" in result:
                # 移除圖像數據以節省傳輸
                result["face_analysis"].pop("annotated_image", None)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "patient_id": patient_id
            }

    def get_simple_result(self, base64_image: str) -> Dict:
        """
        獲取簡化結果（僅包含關鍵信息）

        Args:
            base64_image: base64編碼的面部圖像

        Returns:
            簡化的分析結果
        """
        full_result = self.analyze(base64_image, include_image=False)

        if not full_result.get("success", False):
            return full_result

        # 提取關鍵信息
        simplified = {
            "success": True,
            "health_status": full_result["face_analysis"]["summary"]["health_status"],
            "abnormal_regions_count": full_result["face_analysis"]["summary"]["abnormal_regions_count"],
            "constitution_type": full_result["tcm_diagnosis"]["constitution_type"]["primary_type"],
            "severity_level": full_result["tcm_diagnosis"]["severity_level"],
            "main_recommendations": {
                "lifestyle": full_result["recommendations"]["lifestyle"][:3],  # 只取前3條
                "diet": full_result["recommendations"]["diet"][:3],
                "follow_up": full_result["recommendations"]["follow_up"]
            },
            "timestamp": full_result["analysis_timestamp"]
        }

        return simplified



