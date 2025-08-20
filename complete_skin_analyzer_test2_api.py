import os
import cv2
import numpy as np
from PIL import Image
import base64
import io
import tempfile
import dlib
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
    NOSE_TIP = "鼻頭(脾)"
    NOSE_ROOT = "鼻根(肺)"
    PHILTRUM = "人中"
    RIGHT_NOSE_WING = "右鼻翼"
    LEFT_NOSE_WING = "左鼻翼"


class SkinCondition(Enum):
    """膚色狀態定義"""
    NORMAL = "正常"
    DARK = "發黑"
    RED = "發紅"
    PALE = "發白"
    YELLOW = "發黃"
    CYAN = "發青"


class FacialLandmarkDetector:
    """基於dlib的面部特徵點檢測器"""

    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.predictor_path = predictor_path
        self.detector = None
        self.predictor = None
        self.init_detector()

    def init_detector(self):
        """初始化dlib檢測器"""
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)
            return True
        except Exception as e:
            print(f"dlib檢測器初始化失敗: {e}")
            return False

    def detect_faces_with_landmarks(self, image):
        """檢測人臉並返回特徵點"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        face_data = []
        for face in faces:
            landmarks = self.predictor(gray, face)
            face_data.append({
                'rect': face,
                'landmarks': landmarks
            })

        return face_data

    def detect_forehead_regions(self, landmarks):
        """檢測額頭區域並分成三部分"""
        eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]
        min_y = min([p[1] for p in eyebrow_points])
        left_x = landmarks.part(17).x
        right_x = landmarks.part(26).x
        forehead_height = int((right_x - left_x) * 0.4)
        segment_height = forehead_height // 3
        y_coords = [min_y - i * segment_height for i in range(4)]

        regions = {}
        for i in range(3):
            top_left = (left_x, y_coords[i + 1])
            bottom_right = (right_x, y_coords[i])
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]

            if i == 0:
                regions[FaceRegion.FOREHEAD_TOP] = (top_left[0], top_left[1], width, height)
            elif i == 1:
                regions[FaceRegion.FOREHEAD_MIDDLE] = (top_left[0], top_left[1], width, height)
            else:
                regions[FaceRegion.FOREHEAD_BOTTOM] = (top_left[0], top_left[1], width, height)

        return regions

    def detect_cheek_regions(self, landmarks):
        """檢測臉頰區域"""
        # 右臉頰 (從用戶視角看是右邊)
        right_x1, right_y1 = landmarks.part(40).x, landmarks.part(40).y
        right_x2, right_y2 = landmarks.part(4).x, landmarks.part(4).y
        right_top_left = (min(right_x1, right_x2), min(right_y1, right_y2))
        right_bottom_right = (max(right_x1, right_x2), max(right_y1, right_y2))
        right_width = right_bottom_right[0] - right_top_left[0]
        right_height = right_bottom_right[1] - right_top_left[1]

        # 左臉頰 (從用戶視角看是左邊)
        left_x1, left_y1 = landmarks.part(47).x, landmarks.part(47).y
        left_x2, left_y2 = landmarks.part(12).x, landmarks.part(12).y
        left_top_left = (min(left_x1, left_x2), min(left_y1, left_y2))
        left_bottom_right = (max(left_x1, left_x2), max(left_y1, left_y2))
        left_width = left_bottom_right[0] - left_top_left[0]
        left_height = left_bottom_right[1] - left_top_left[1]

        return {
            FaceRegion.RIGHT_CHEEK: (right_top_left[0], right_top_left[1], right_width, right_height),
            FaceRegion.LEFT_CHEEK: (left_top_left[0], left_top_left[1], left_width, left_height)
        }

    def detect_nose_regions(self, landmarks):
        """檢測鼻子相關區域"""
        regions = {}

        # 鼻頭(脾) - 使用特徵點30
        nose_tip_x = landmarks.part(30).x
        nose_tip_y = landmarks.part(30).y
        size = 20
        regions[FaceRegion.NOSE_TIP] = (
            nose_tip_x - size // 2,
            nose_tip_y - size // 2,
            size,
            size
        )

        # 鼻根(肺) - 使用特徵點27
        nose_root_x = landmarks.part(27).x
        nose_root_y = landmarks.part(27).y
        regions[FaceRegion.NOSE_ROOT] = (
            nose_root_x - size // 2,
            nose_root_y - size // 2,
            size,
            size
        )

        # 右鼻翼
        regions[FaceRegion.RIGHT_NOSE_WING] = (
            landmarks.part(32).x - size // 4,
            landmarks.part(32).y - size // 4,
            size // 2,
            size // 2
        )

        # 左鼻翼
        regions[FaceRegion.LEFT_NOSE_WING] = (
            landmarks.part(36).x - size // 4,
            landmarks.part(36).y - size // 4,
            size // 2,
            size // 2
        )

        return regions

    def detect_other_regions(self, landmarks):
        """檢測其他面部區域"""
        regions = {}

        # 人中區域 - 使用鼻子下方和嘴唇上方的區域
        philtrum_x = landmarks.part(33).x
        philtrum_y = landmarks.part(33).y
        mouth_top_y = landmarks.part(51).y

        philtrum_width = 20
        philtrum_height = mouth_top_y - philtrum_y

        regions[FaceRegion.PHILTRUM] = (
            philtrum_x - philtrum_width // 2,
            philtrum_y,
            philtrum_width,
            philtrum_height
        )

        # 下巴區域 - 使用特徵點8附近
        chin_x = landmarks.part(8).x
        chin_y = landmarks.part(8).y
        chin_size = 30

        regions[FaceRegion.CHIN] = (
            chin_x - chin_size // 2,
            chin_y - chin_size // 4,
            chin_size,
            chin_size // 2
        )

        return regions

    def get_all_face_regions(self, landmarks):
        """獲取所有面部區域"""
        regions = {}

        # 合併所有區域
        regions.update(self.detect_forehead_regions(landmarks))
        regions.update(self.detect_cheek_regions(landmarks))
        regions.update(self.detect_nose_regions(landmarks))
        regions.update(self.detect_other_regions(landmarks))

        return regions


class FaceSkinAnalyzer:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.landmark_detector = FacialLandmarkDetector(predictor_path)
        self.diagnosis_results = {}
        self.face_regions = {}
        self.current_face_rect = None

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

    def analyze_skin_color_for_region(self, image, region_rect):
        """分析特定區域的膚色"""
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
        if cv2.countNonZero(skin_mask) < (w * h * 0.1):
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
        """根據RGB值判斷膚色狀態"""
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
            }
        }

        return diagnosis_map.get(region, {}).get(condition, "")

    def draw_face_regions(self, image):
        """在圖像上繪製面部區域"""
        if not self.face_regions:
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

            # 使用dlib檢測人臉和特徵點
            face_data = self.landmark_detector.detect_faces_with_landmarks(image)

            if not face_data:
                return {
                    "success": False,
                    "error": "未能檢測到面部特徵點。\n\n請確保：\n• 臉部完整且清晰可見\n• 光線充足且均勻\n• 避免過暗或逆光\n• 正對鏡頭\n\n調整後重新拍攝或選擇照片。",
                    "original_image": base64_string,
                    "annotated_image": None,
                    "overall_color": None,
                    "diagnosis": None
                }

            # 使用第一個檢測到的人臉
            face_info = face_data[0]
            landmarks = face_info['landmarks']
            face_rect = face_info['rect']

            # 轉換dlib rect格式為(x, y, w, h)
            self.current_face_rect = (face_rect.left(), face_rect.top(),
                                      face_rect.width(), face_rect.height())

            # 使用dlib特徵點定義面部區域
            self.face_regions = self.landmark_detector.get_all_face_regions(landmarks)

            # 分析每個區域
            for region, region_rect in self.face_regions.items():
                # 分析該區域的膚色
                mean_color = self.analyze_skin_color_for_region(image, region_rect)

                # 判斷此膚色狀態
                condition = self.diagnose_skin_condition(mean_color)

                # 儲存診斷結果
                self.diagnosis_results[region] = condition

            # 計算整體膚色RGB值
            overall_color = self.analyze_skin_color_for_region(image, self.current_face_rect)

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


# API 相關函數
def main(request):
    """主要的API處理函數 - 處理Flask請求"""
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400

    try:
        image_bytes = request.files['image'].read()

        # 將圖像轉換為base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')

        # 根據文件類型添加適當的前綴
        filename = request.files['image'].filename.lower()
        if filename.endswith('.png'):
            base64_string = f"data:image/png;base64,{base64_string}"
        elif filename.endswith(('.jpg', '.jpeg')):
            base64_string = f"data:image/jpeg;base64,{base64_string}"
        else:
            base64_string = f"data:image/png;base64,{base64_string}"

        # 使用整合的分析器
        analyzer = FaceSkinAnalyzer()
        result = analyzer.analyze_from_base64(base64_string)

        return result

    except Exception as e:
        return {
            'success': False,
            'error': f'處理圖像時發生錯誤：{str(e)}'
        }, 500


def analyze_from_temp_file(image_bytes, predictor_path="shape_predictor_68_face_landmarks.dat"):
    """從圖像字節數據分析 - 使用臨時文件"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        # 轉換為base64進行分析
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        base64_string = f"data:image/jpeg;base64,{base64_string}"

        # 使用整合的分析器
        analyzer = FaceSkinAnalyzer(predictor_path)
        result = analyzer.analyze_from_base64(base64_string)

        return result

    finally:
        os.remove(temp_path)


# 便捷函數
def analyze_face_from_base64(base64_string, predictor_path="shape_predictor_68_face_landmarks.dat"):
    """便捷函數：從base64字符串分析面部膚色"""
    analyzer = FaceSkinAnalyzer(predictor_path)
    return analyzer.analyze_from_base64(base64_string)


def analyze_face_from_file(file_path, predictor_path="shape_predictor_68_face_landmarks.dat"):
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
        analyzer = FaceSkinAnalyzer(predictor_path)
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
    # 示例1：從文件分析（使用dlib精確檢測）
    def example_from_file():
        print("=== 從文件分析示例（使用dlib精確檢測） ===")
        file_path = "example.jpg"  # 替換為您的圖像文件路徑
        predictor_path = "shape_predictor_68_face_landmarks.dat"  # dlib模型路徑

        result = analyze_face_from_file(file_path, predictor_path)

        if result["success"]:
            print("分析成功！")
            print(
                f"整體膚色 RGB: R={result['overall_color']['r']}, G={result['overall_color']['g']}, B={result['overall_color']['b']}")
            print(f"整體膚色 Hex: {result['overall_color']['hex']}")
            print("\n診斷結果：")
            print(result["diagnosis"])

            # 保存標註圖像
            if result["annotated_image"]:
                save_base64_image(result["annotated_image"], "annotated_result_dlib.png")
                print("\n標註圖像已保存為 'annotated_result_dlib.png'")
        else:
            print(f"分析失敗：{result['error']}")


    # 示例2：從base64字符串分析
    def example_from_base64():
        print("\n=== 從base64字符串分析示例（使用dlib精確檢測） ===")
        predictor_path = "shape_predictor_68_face_landmarks.dat"

        try:
            with open("example.jpg", 'rb') as f:  # 替換為您的圖像文件
                image_data = f.read()
                base64_string = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

            result = analyze_face_from_base64(base64_string, predictor_path)

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
        print("\n=== 批量處理示例（使用dlib精確檢測） ===")
        predictor_path = "shape_predictor_68_face_landmarks.dat"

        image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]  # 替換為您的圖像文件列表
        results = []

        for file_path in image_files:
            try:
                result = analyze_face_from_file(file_path, predictor_path)
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
        print("\n=== JSON格式輸出示例（使用dlib精確檢測） ===")
        predictor_path = "shape_predictor_68_face_landmarks.dat"

        try:
            result = analyze_face_from_file("example.jpg", predictor_path)  # 替換為您的圖像文件

            # 將結果轉換為JSON格式
            json_result = json.dumps(result, ensure_ascii=False, indent=2)
            print("JSON格式結果：")
            print(json_result)

            # 保存JSON結果到文件
            with open("analysis_result_dlib.json", "w", encoding="utf-8") as f:
                f.write(json_result)
            print("\nJSON結果已保存為 'analysis_result_dlib.json'")

        except FileNotFoundError:
            print("請提供有效的圖像文件路徑進行測試")




