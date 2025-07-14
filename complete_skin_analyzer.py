import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
from enum import Enum
from typing import Dict, Tuple, List, Optional
import threading


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
            text_color = (0, 0, 0) if condition in [SkinCondition.PALE, SkinCondition.YELLOW, SkinCondition.CYAN] else (255, 255, 255)
            cv2.putText(annotated_image, label_text, (x + 2, y - 2),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return annotated_image

    def process_and_analyze_face(self, image):
        """完整的圖像處理和人臉分析流程"""
        self.diagnosis_results.clear()
        self.face_regions.clear()
        self.current_face_rect = None

        # 檢測人臉
        faces = self.detect_faces(image)

        if len(faces) == 0:
            return None, "未能檢測到面部。\n\n請確保：\n• 臉部完整且清晰可見\n• 光線充足且均勻\n• 避免過暗或逆光\n• 正對鏡頭\n\n調整後重新拍攝或選擇照片。", None

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

        return overall_color, None, annotated_image

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


class FaceSkinAnalysisGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("面部膚色分析系統")
        self.root.geometry("1400x800")  # 增加視窗寬度以容納新的顯示區域

        self.analyzer = FaceSkinAnalyzer()
        self.current_image = None
        self.show_regions = tk.BooleanVar(value=True)  # 控制是否顯示區域標註

        self.setup_ui()

    def setup_ui(self):
        """設定使用者介面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 按鈕框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # 選擇圖片按鈕
        self.btn_select = ttk.Button(button_frame, text="選擇照片", command=self.select_image)
        self.btn_select.pack(side=tk.LEFT, padx=(0, 10))

        # 攝影機按鈕
        self.btn_camera = ttk.Button(button_frame, text="拍攝照片", command=self.open_camera)
        self.btn_camera.pack(side=tk.LEFT, padx=(0, 10))

        # 區域顯示控制
        self.chk_regions = ttk.Checkbutton(button_frame, text="顯示面部區域標註",
                                           variable=self.show_regions,
                                           command=self.toggle_region_display)
        self.chk_regions.pack(side=tk.LEFT, padx=(10, 0))

        # 原始圖像顯示區域
        original_frame = ttk.LabelFrame(main_frame, text="原始圖像", padding="5")
        original_frame.grid(row=1, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))

        self.original_image_label = ttk.Label(original_frame, text="請選擇照片或拍攝照片",
                                              relief="solid", padding="20")
        self.original_image_label.pack(expand=True, fill="both")

        # 區域標註圖像顯示區域
        annotated_frame = ttk.LabelFrame(main_frame, text="面部區域分析", padding="5")
        annotated_frame.grid(row=1, column=1, padx=(5, 5), sticky=(tk.W, tk.E, tk.N, tk.S))

        self.annotated_image_label = ttk.Label(annotated_frame, text="區域分析結果將顯示在這裡",
                                               relief="solid", padding="20")
        self.annotated_image_label.pack(expand=True, fill="both")

        # 結果顯示區域
        result_frame = ttk.LabelFrame(main_frame, text="分析結果", padding="5")
        result_frame.grid(row=1, column=2, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))

        # RGB結果
        ttk.Label(result_frame, text="整體膚色 RGB 值：", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.rgb_text = tk.Text(result_frame, height=5, width=25, font=("Arial", 10))
        self.rgb_text.pack(fill="x", pady=(0, 10))

        # 診斷結果
        ttk.Label(result_frame, text="診斷結果：", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.diagnosis_text = tk.Text(result_frame, height=20, width=25, font=("Arial", 10))
        self.diagnosis_text.pack(fill="both", expand=True, pady=(0, 10))

        # 顏色圖例
        legend_frame = ttk.LabelFrame(result_frame, text="顏色圖例", padding="5")
        legend_frame.pack(fill="x", pady=(5, 0))

        legend_text = "綠色：正常\n紅色：發紅\n深藍：發黑\n白色：發白\n黃色：發黃\n青色：發青"
        ttk.Label(legend_frame, text=legend_text, font=("Arial", 9)).pack()

        # 配置格網權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def toggle_region_display(self):
        """切換區域顯示"""
        if hasattr(self, 'current_annotated_image') and self.current_annotated_image is not None:
            self.update_annotated_display()

    def select_image(self):
        """選擇圖片檔案"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            self.process_image_file(file_path)

    def process_image_file(self, file_path):
        """處理圖片檔案"""
        try:
            # 讀取圖像
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("錯誤", "無法讀取圖片檔案")
                return

            # 在背景執行緒中處理圖像
            threading.Thread(target=self.analyze_image, args=(image,), daemon=True).start()

        except Exception as e:
            messagebox.showerror("錯誤", f"處理圖片時出錯：{str(e)}")

    def analyze_image(self, image):
        """分析圖像"""
        try:
            # 清空之前的結果
            self.root.after(0, self.clear_results)

            # 儲存原始圖像
            self.current_image = image

            # 縮放圖像以適應顯示
            display_image = self.resize_image_for_display(image)

            # 分析人臉和膚色
            overall_color, error_message, annotated_image = self.analyzer.process_and_analyze_face(image)

            # 在主執行緒中更新UI
            if annotated_image is not None:
                annotated_display_image = self.resize_image_for_display(annotated_image)
                self.current_annotated_image = annotated_display_image
            else:
                self.current_annotated_image = None

            self.root.after(0, self.update_results, display_image, overall_color, error_message)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"分析圖片時出錯：{str(e)}"))

    def resize_image_for_display(self, image, max_size=(350, 350)):
        """調整圖像大小以適應顯示"""
        h, w = image.shape[:2]
        max_w, max_h = max_size

        # 計算縮放比例
        scale = min(max_w / w, max_h / h)

        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h))

        return image

    def clear_results(self):
        """清空結果顯示"""
        self.rgb_text.delete(1.0, tk.END)
        self.diagnosis_text.delete(1.0, tk.END)
        self.original_image_label.configure(image="", text="請選擇照片或拍攝照片")
        self.annotated_image_label.configure(image="", text="區域分析結果將顯示在這裡")

    def update_annotated_display(self):
        """更新標註圖像顯示"""
        if self.current_annotated_image is not None and self.show_regions.get():
            # 顯示帶標註的圖像
            image_rgb = cv2.cvtColor(self.current_annotated_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image_pil)

            self.annotated_image_label.configure(image=photo, text="")
            self.annotated_image_label.image = photo

        elif hasattr(self, 'current_original_display') and self.current_original_display is not None:
            # 顯示原始圖像
            image_rgb = cv2.cvtColor(self.current_original_display, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image_pil)

            self.annotated_image_label.configure(image=photo, text="")
            self.annotated_image_label.image = photo

        else:
            self.annotated_image_label.configure(image="", text="區域分析結果將顯示在這裡")

    def update_results(self, display_image, overall_color, error_message):
        """更新結果顯示"""
        # 儲存原始顯示圖像
        self.current_original_display = display_image

        # 顯示原始圖像
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image_pil)

        self.original_image_label.configure(image=photo, text="")
        self.original_image_label.image = photo

        # 更新標註圖像顯示
        self.update_annotated_display()

        if error_message:
            # 顯示錯誤訊息
            self.diagnosis_text.insert(tk.END, error_message)
        else:
            # 顯示RGB結果
            if overall_color:
                b, g, r = overall_color
                rgb_text = f"R: {int(r):03d}\nG: {int(g):03d}\nB: {int(b):03d}\n"
                hex_text = f"Hex: #{int(r):02X}{int(g):02X}{int(b):02X}"
                self.rgb_text.insert(tk.END, rgb_text + hex_text)

            # 顯示診斷結果
            diagnosis_result = self.analyzer.get_diagnosis_text()
            if diagnosis_result:
                self.diagnosis_text.insert(tk.END, diagnosis_result)

    def open_camera(self):
        """打開攝影機"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("錯誤", "無法打開攝影機")
                return

            # 建立攝影機視窗
            camera_window = tk.Toplevel(self.root)
            camera_window.title("攝影機")
            camera_window.geometry("640x480")

            camera_label = ttk.Label(camera_window)
            camera_label.pack(expand=True)

            capture_btn = ttk.Button(camera_window, text="拍照分析",
                                     command=lambda: self.capture_photo(cap, camera_window))
            capture_btn.pack(pady=10)

            # 顯示攝影機畫面
            self.show_camera_feed(cap, camera_label, camera_window)

        except Exception as e:
            messagebox.showerror("錯誤", f"打開攝影機時出錯：{str(e)}")

    def show_camera_feed(self, cap, label, window):
        """顯示攝影機畫面"""
        if not window.winfo_exists():
            cap.release()
            return

        ret, frame = cap.read()
        if ret:
            # 翻轉圖像（鏡像效果）
            frame = cv2.flip(frame, 1)

            # 轉換色彩空間並顯示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image_pil)

            label.configure(image=photo)
            label.image = photo

        # 繼續更新
        window.after(30, lambda: self.show_camera_feed(cap, label, window))

    def capture_photo(self, cap, camera_window):
        """拍照並分析"""
        ret, frame = cap.read()
        if ret:
            # 翻轉圖像
            frame = cv2.flip(frame, 1)

            # 關閉攝影機視窗
            cap.release()
            camera_window.destroy()

            # 分析拍攝的圖像
            threading.Thread(target=self.analyze_image, args=(frame,), daemon=True).start()

    def run(self):
        """執行應用程式"""
        self.root.mainloop()


def main():
    """主函數"""
    try:
        app = FaceSkinAnalysisGUI()
        app.run()
    except Exception as e:
        print(f"應用程式啟動失敗：{e}")


if __name__ == "__main__":
    main()