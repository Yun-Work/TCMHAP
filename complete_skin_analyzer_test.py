import cv2
import numpy as np
from PIL import Image
import base64
import io
from enum import Enum
from typing import Dict, Tuple, List, Optional
import json


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


class FaceSkinAnalyzer:
    def __init__(self):
        self.face_cascade = None
        self.diagnosis_results = {}
        self.face_regions = {}
        self.current_face_rect = None
        self.init_face_detector()

    def init_face_detector(self):
        """初始化人臉檢測器"""
        try:
            # 嘗試載入OpenCV內建的人臉檢測器
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("無法載入人臉檢測模型")
            return True
        except Exception as e:
            print(f"人臉檢測器初始化失敗: {e}")
            return False

    def base64_to_image(self, base64_string):
        """將base64字符串轉換為OpenCV圖像"""
        try:
            # 移除base64前綴（如果存在）
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]

            # 解碼base64
            image_data = base64.b64decode(base64_string)

            # 轉換為PIL圖像
            image_pil = Image.open(io.BytesIO(image_data))

            # 確保圖像為RGB格式
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')

            # 轉換為numpy數組
            image_array = np.array(image_pil)

            # 轉換為BGR格式（OpenCV使用）
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            return image_bgr
        except Exception as e:
            raise Exception(f"base64轉換圖像失敗: {e}")

    def image_to_base64(self, image):
        """將OpenCV圖像轉換為base64字符串"""
        try:
            # 轉換為RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 轉換為PIL圖像
            image_pil = Image.fromarray(image_rgb)

            # 保存到字節流
            buffer = io.BytesIO()
            image_pil.save(buffer, format='PNG')
            buffer.seek(0)

            # 編碼為base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            raise Exception(f"圖像轉換base64失敗: {e}")

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
        """定義面部各個區域（改進版）"""
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
        cheek_y_start = y + h // 3 + h // 8  # 更精確的垂直位置
        cheek_height = h // 4  # 適中的高度
        cheek_width = w // 4  # 適中的寬度

        # 左臉頰 - 避開邊緣，確保在臉部內側
        regions[FaceRegion.LEFT_CHEEK] = (
            x + w // 8, cheek_y_start, cheek_width, cheek_height
        )

        # 右臉頰 - 確保對稱且在臉部內側
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
        philtrum_y = y + int(h * 0.7)  # 從臉部60%的位置開始，更接近真實人中位置
        regions[FaceRegion.PHILTRUM] = (
            x + w // 2 - philtrum_width // 2, philtrum_y,
            philtrum_width, philtrum_height
        )

        # 下巴
        chin_y_start = y + int(h * 0.9)  # 從臉部80%的位置開始，避開嘴唇
        chin_height = h - (chin_y_start - y)  # 剩餘的高度
        chin_width = w // 3  # 縮小寬度，集中在下巴中央
        chin_x_start = x + w // 2 - chin_width // 2  # 居中對齊

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
        """分析特定區域的膚色（改進版）"""
        x, y, w, h = region_rect

        # 確保區域在圖像範圍內
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)

        if w <= 0 or h <= 0:
            return (153, 134, 117)  # 回傳預設膚色值

        region = image[y:y + h, x:x + w]

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(region, (7, 7), 0)

        # 轉換到多個色彩空間進行分析
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        # 計算平均亮度
        avg_brightness = np.mean(hsv[:, :, 2])

        # 改進的膚色範圍檢測
        if avg_brightness < 50:  # 低光照
            lower_skin = np.array([0, 5, 20])
            upper_skin = np.array([40, 255, 255])
        else:  # 正常光照
            lower_skin = np.array([0, 10, 40])
            upper_skin = np.array([40, 255, 255])

        # 進行膚色檢測
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 改進的形態學處理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # 如果膚色檢測失敗，使用更寬鬆的條件
        if cv2.countNonZero(skin_mask) < (w * h * 0.1):  # 如果檢測到的膚色區域太小
            # 使用更寬鬆的HSV範圍
            lower_skin_loose = np.array([0, 8, 30])
            upper_skin_loose = np.array([50, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin_loose, upper_skin_loose)

            # 如果還是檢測不到，直接使用整個區域
            if cv2.countNonZero(skin_mask) == 0:
                skin_mask = np.ones((h, w), dtype=np.uint8) * 255

        # 提取膚色區域並計算平均顏色
        skin_region = cv2.bitwise_and(blurred, blurred, mask=skin_mask)
        mean_color = cv2.mean(skin_region, skin_mask)

        return mean_color[:3]  # 回傳BGR值

    def diagnose_skin_condition(self, mean_color):
        """根據RGB值判斷膚色狀態（增強版 - 包含發黃和發青）"""
        b, g, r = mean_color  # OpenCV使用BGR格式

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

        # 判斷膚色狀態（優先級順序很重要）
        if brightness < 70:  # 發黑檢測
            return SkinCondition.DARK

        elif brightness > 200 and min_color > 150 and saturation < 0.1:  # 發白檢測
            return SkinCondition.PALE

        elif red_ratio > 0.42 and r > 150:  # 原為0.45/150，調整為0.42/140
            return SkinCondition.RED

        elif (green_ratio > red_ratio and green_ratio > blue_ratio and
              green_ratio > 0.38 and g > 130):  # 發黃檢測
            # 黃色是紅+綠，但綠色成分更突出
            if red_ratio > 0.3:  # 確保有足夠的紅色成分形成黃色
                return SkinCondition.YELLOW

        elif (blue_ratio > red_ratio and blue_ratio > green_ratio and
              blue_ratio > 0.38 and b > 130):  # 發青檢測
            # 青色是綠+藍，但藍色成分更突出
            if green_ratio > 0.25:  # 確保有足夠的綠色成分形成青色
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
        """在圖像上繪製面部區域"""
        if not self.face_regions or self.current_face_rect is None:
            return image

        # 複製圖像以避免修改原圖
        annotated_image = image.copy()

        # 定義顏色對應
        condition_colors = {
            SkinCondition.NORMAL: (0, 255, 0),  # 綠色
            SkinCondition.DARK: (0, 0, 139),  # 深藍色
            SkinCondition.RED: (0, 0, 255),  # 紅色
            SkinCondition.PALE: (255, 255, 255),  # 白色
            SkinCondition.YELLOW: (0, 255, 255),  # 黃色
            SkinCondition.CYAN: (255, 255, 0)  # 青色
        }

        # 為每個區域繪製框線和標籤
        for region, region_rect in self.face_regions.items():
            x, y, w, h = region_rect

            # 獲取該區域的診斷結果
            condition = self.diagnosis_results.get(region, SkinCondition.NORMAL)
            color = condition_colors.get(condition, (0, 255, 0))

            # 繪製矩形框
            thickness = 3 if condition != SkinCondition.NORMAL else 2
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)

            # 添加標籤背景
            label_text = region.value
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1

            # 計算文字大小
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            # 繪製標籤背景
            cv2.rectangle(annotated_image,
                          (x, y - text_height - 5),
                          (x + text_width + 5, y),
                          color, -1)

            # 繪製標籤文字
            text_color = (0, 0, 0) if condition in [SkinCondition.PALE, SkinCondition.YELLOW, SkinCondition.CYAN] else (
            255, 255, 255)
            cv2.putText(annotated_image, label_text, (x + 2, y - 2),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return annotated_image

    def analyze_from_base64(self, base64_string):
        """從base64字符串分析圖像"""
        try:
            # 清空之前的結果
            self.diagnosis_results.clear()
            self.face_regions.clear()
            self.current_face_rect = None

            # 轉換base64為圖像
            image = self.base64_to_image(base64_string)

            # 檢測人臉
            faces = self.detect_faces(image)

            if len(faces) == 0:
                return {
                    "success": False,
                    "error": "未能檢測到面部。\n\n請確保：\n• 臉部完整且清晰可見\n• 光線充足且均勻\n• 避免過暗或逆光\n• 正對鏡頭\n\n調整後重新拍攝或選擇照片。",
                    "original_image": base64_string,
                    "annotated_image": None,
                    "overall_color": None,
                    "diagnosis": None
                }

            # 獲取最大的人臉區域
            largest_face = self.get_largest_face(faces)
            self.current_face_rect = largest_face

            # 定義面部區域
            self.face_regions = self.define_face_regions(largest_face)

            # 分析每個區域
            for region, region_rect in self.face_regions.items():
                # 分析該區域的膚色
                mean_color = self.analyze_skin_color_for_region(image, region_rect)

                # 判斷此膚色狀態
                condition = self.diagnose_skin_condition(mean_color)

                # 儲存診斷結果
                self.diagnosis_results[region] = condition

            # 計算整體膚色RGB值
            overall_color = self.analyze_skin_color_for_region(image, largest_face)

            # 生成帶有區域標註的圖像
            annotated_image = self.draw_face_regions(image)

            # 轉換結果圖像為base64
            annotated_base64 = self.image_to_base64(annotated_image)

            # 生成診斷文字
            diagnosis_text = self.get_diagnosis_text()

            return {
                "success": True,
                "error": None,
                "original_image": base64_string,
                "annotated_image": annotated_base64,
                "overall_color": {
                    "r": int(overall_color[2]),
                    "g": int(overall_color[1]),
                    "b": int(overall_color[0]),
                    "hex": f"#{int(overall_color[2]):02X}{int(overall_color[1]):02X}{int(overall_color[0]):02X}"
                },
                "diagnosis": diagnosis_text,
                "region_results": {region.value: condition.value for region, condition in
                                   self.diagnosis_results.items()}
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"分析過程中發生錯誤：{str(e)}",
                "original_image": base64_string,
                "annotated_image": None,
                "overall_color": None,
                "diagnosis": None
            }

    def get_diagnosis_text(self):
        """獲取診斷結果文字"""
        if not self.diagnosis_results:
            return ""

        result = "面色診斷結果：\n"
        has_abnormal = False

        for region, condition in self.diagnosis_results.items():
            if condition != SkinCondition.NORMAL:
                has_abnormal = True
                result += f"{region.value}：{condition.value}\n"

        if not has_abnormal:
            result += "所有區域膚色正常"
        else:
            result += "\n可能關聯的器官健康問題：\n"
            for region, condition in self.diagnosis_results.items():
                if condition != SkinCondition.NORMAL:
                    diagnosis = self.get_organ_diagnosis(region, condition)
                    if diagnosis:
                        result += diagnosis + "\n"

        return result


def analyze_face_from_base64(base64_string):
    """便捷函數：從base64字符串分析面部膚色"""
    analyzer = FaceSkinAnalyzer()
    return analyzer.analyze_from_base64(base64_string)


def analyze_face_from_file(file_path):
    """便捷函數：從文件路徑分析面部膚色"""
    try:
        # 讀取圖像文件
        with open(file_path, 'rb') as f:
            image_data = f.read()

        # 轉換為base64
        base64_string = base64.b64encode(image_data).decode('utf-8')

        # 添加適當的前綴
        if file_path.lower().endswith('.png'):
            base64_string = f"data:image/png;base64,{base64_string}"
        elif file_path.lower().endswith(('.jpg', '.jpeg')):
            base64_string = f"data:image/jpeg;base64,{base64_string}"
        else:
            base64_string = f"data:image/png;base64,{base64_string}"

        # 分析
        analyzer = FaceSkinAnalyzer()
        return analyzer.analyze_from_base64(base64_string)

    except Exception as e:
        return {
            "success": False,
            "error": f"讀取文件失敗：{str(e)}",
            "original_image": None,
            "annotated_image": None,
            "overall_color": None,
            "diagnosis": None
        }


def save_base64_image(base64_string, output_path):
    """將base64字符串保存為圖像文件"""
    try:
        # 移除base64前綴（如果存在）
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        # 解碼base64
        image_data = base64.b64decode(base64_string)

        # 保存到文件
        with open(output_path, 'wb') as f:
            f.write(image_data)

        return True
    except Exception as e:
        print(f"保存圖像失敗：{e}")
        return False


# 使用示例
if __name__ == "__main__":
    # 示例1：從文件分析
    def example_from_file():
        print("=== 從文件分析示例 ===")
        file_path = "example.jpg"  # 替換為您的圖像文件路徑

        result = analyze_face_from_file(file_path)

        if result["success"]:
            print("分析成功！")
            print(
                f"整體膚色 RGB: R={result['overall_color']['r']}, G={result['overall_color']['g']}, B={result['overall_color']['b']}")
            print(f"整體膚色 Hex: {result['overall_color']['hex']}")
            print("\n診斷結果：")
            print(result["diagnosis"])

            # 保存標註圖像
            if result["annotated_image"]:
                save_base64_image(result["annotated_image"], "annotated_result.png")
                print("\n標註圖像已保存為 'annotated_result.png'")
        else:
            print(f"分析失敗：{result['error']}")


    # 示例2：從base64字符串分析
    def example_from_base64():
        print("\n=== 從base64字符串分析示例 ===")

        # 這裡應該是您的base64字符串
        # base64_string = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."

        # 由於這是示例，我們從文件讀取來創建base64
        try:
            with open("example.jpg", 'rb') as f:  # 替換為您的圖像文件
                image_data = f.read()
                base64_string = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

            result = analyze_face_from_base64(base64_string)

            if result["success"]:
                print("分析成功！")
                print("區域分析結果：")
                for region, condition in result["region_results"].items():
                    print(f"  {region}: {condition}")
            else:
                print(f"分析失敗：{result['error']}")

        except FileNotFoundError:
            print("請提供有效的圖像文件路徑進行測試")


    # 示例3：批量處理
    def example_batch_processing():
        print("\n=== 批量處理示例 ===")

        image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]  # 替換為您的圖像文件列表
        results = []

        for file_path in image_files:
            try:
                result = analyze_face_from_file(file_path)
                results.append({
                    "file": file_path,
                    "result": result
                })
                print(f"處理完成：{file_path} - {'成功' if result['success'] else '失敗'}")
            except:
                print(f"處理失敗：{file_path}")

        # 輸出批量處理結果
        for item in results:
            if item["result"]["success"]:
                print(f"\n{item['file']} 分析結果：")
                print(item["result"]["diagnosis"])


    # 示例4：JSON輸出
    def example_json_output():
        print("\n=== JSON格式輸出示例 ===")

        try:
            result = analyze_face_from_file("example.jpg")  # 替換為您的圖像文件

            # 將結果轉換為JSON格式
            json_result = json.dumps(result, ensure_ascii=False, indent=2)
            print("JSON格式結果：")
            print(json_result)

            # 保存JSON結果到文件
            with open("analysis_result.json", "w", encoding="utf-8") as f:
                f.write(json_result)
            print("\nJSON結果已保存為 'analysis_result.json'")

        except FileNotFoundError:
            print("請提供有效的圖像文件路徑進行測試")



    