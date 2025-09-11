import os
import cv2
import numpy as np
from PIL import Image
import base64
import io
import insightface
import mediapipe as mp
from enum import Enum
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import functools

try:
    from app.services.mole_detection_service import (
        MoleDetectionService,
        detect_and_analyze_moles,
        process_image_for_moles,
        remove_beard_from_image
    )

    MOLE_DETECTION_AVAILABLE = True
    print("痣檢測服務導入成功")
except ImportError as e:
    try:
        # 嘗試相對導入
        from .mole_detection_service import (
            MoleDetectionService,
            detect_and_analyze_moles,
            process_image_for_moles,
            remove_beard_from_image
        )

        MOLE_DETECTION_AVAILABLE = True
        print("痣檢測服務導入成功（相對路徑）")
    except ImportError as e2:
        MOLE_DETECTION_AVAILABLE = False
        MoleDetectionService = None
        remove_beard_from_image = None
        print(f"痣檢測服務導入失敗: {e}, {e2}")


class FaceRegion(Enum):
    FOREHEAD_UPPER = "額上1/3"  # 心
    NOSE_BRIDGE_ROOT = "鼻根(心與肝交會)"  # 心（兼肝交會區）
    CHEEK_UPPER_RIGHT = "右上頰"  # 肺
    # CHEEK_DUAL_UPPER = "雙頰"  # 肺
    NOSE_TIP = "鼻頭"  # 脾
    NOSE_WING = "鼻翼"  # 胃（已合併）
    YINTANG = "印堂"  # 肝
    NOSE_BRIDGE_MID = "鼻樑中段"  # 肝
    LEFT_UPPER_CHEEK = "左上頰"  # 肝
    NOSE_BRIDGE_OUTER = "鼻樑外側"  # 膽（已合併）
    ZYGOMATIC_INNER = "顴骨內"  # 小腸（已合併）
    ZYGOMATIC_OUTER = "顴骨外"  # 大腸（已合併）
    # TEMPLE_TO_LOWER_CHEEK_LEFT = "太陽穴至下頰左"  # 腎
    # TEMPLE_TO_LOWER_CHEEK_RIGHT = "太陽穴至下頰右"  # 腎
    CHIN = "下巴"  # 腎區
    LOWER_CHEEK = "下頰"  # 腎（已合併）
    PHILTRUM = "人中"  # 膀胱
    EYE_WHITE = "氣輪(眼白)"  # 肺


class SkinCondition(Enum):
    """膚色狀態定義"""
    NORMAL = "正常"
    DARK = "發黑"
    RED = "發紅"
    PALE = "發白"
    YELLOW = "發黃"
    CYAN = "發青"


class FaceSkinAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """單例模式 - 避免重複初始化檢測器"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self.region_locations = {
            FaceRegion.FOREHEAD_UPPER: "心",  # 額上1/3髮際
            FaceRegion.NOSE_BRIDGE_ROOT: "心",  # 鼻根
            FaceRegion.CHEEK_UPPER_RIGHT: "肺",  # 右上頰
            # FaceRegion.CHEEK_DUAL_UPPER: "肺",  # 雙頰
            FaceRegion.NOSE_TIP: "脾",  # 鼻頭
            FaceRegion.NOSE_WING: "胃",  # 鼻翼（已合併）
            FaceRegion.YINTANG: "肝",  # 印堂（兩眉間）
            FaceRegion.NOSE_BRIDGE_MID: "肝",  # 鼻樑中段
            FaceRegion.LEFT_UPPER_CHEEK: "肝",  # 左上頰
            FaceRegion.NOSE_BRIDGE_OUTER: "膽",  # 鼻樑外側（已合併）
            FaceRegion.ZYGOMATIC_INNER: "小腸",  # 顴骨內側（已合併）
            FaceRegion.ZYGOMATIC_OUTER: "大腸",  # 顴骨外側（已合併）
            # FaceRegion.TEMPLE_TO_LOWER_CHEEK_LEFT: "腎",  # 太陽穴至下頰左
            # FaceRegion.TEMPLE_TO_LOWER_CHEEK_RIGHT: "腎",  # 太陽穴至下頰右
            FaceRegion.CHIN: "腎總區",  # 下巴
            FaceRegion.LOWER_CHEEK: "腎(生殖功能)",  # 下頰，生殖功能（已合併）
            FaceRegion.PHILTRUM: "膀胱",  # 人中
            FaceRegion.EYE_WHITE: "肺"  # 氣輪，眼白
        }
        self.face_app = None
        self.face_mesh = None
        self.diagnosis_results = {}
        self.face_regions = {}
        self.current_face_rect = None
        self._initialized = False

        # 預計算的常量
        self._precompute_constants()

        self.init_detector()

    def _precompute_constants(self):

        # 修改標記點配置，使用多個點來定義合併區域
        self.organ_landmarks = {
            # 心
            FaceRegion.FOREHEAD_UPPER: [10],  # 額上1/3髮際
            FaceRegion.NOSE_BRIDGE_ROOT: [6],  # 鼻根

            # 肺
            FaceRegion.CHEEK_UPPER_RIGHT: [347],  # 右上頰

            # 脾胃
            FaceRegion.NOSE_TIP: [1],  # 鼻頭
            FaceRegion.NOSE_WING: [48, 278],  # 鼻翼（已合併）

            # 肝膽
            FaceRegion.YINTANG: [8],  # 印堂
            FaceRegion.NOSE_BRIDGE_MID: [197],  # 鼻樑中段
            FaceRegion.LEFT_UPPER_CHEEK: [118],  # 左上頰
            FaceRegion.NOSE_BRIDGE_OUTER: [174, 399],  # 鼻樑外側（已合併）

            # 腸道系統
            FaceRegion.ZYGOMATIC_INNER: [120, 349],  # 顴骨內側（已合併）
            FaceRegion.ZYGOMATIC_OUTER: [116, 345],  # 顴骨外側（已合併）

            # 腎
            # FaceRegion.TEMPLE_TO_LOWER_CHEEK_LEFT: [143],  # 太陽穴至下頰左
            # FaceRegion.TEMPLE_TO_LOWER_CHEEK_RIGHT: [372],  # 太陽穴至下頰右
            FaceRegion.CHIN: [175],  # 下巴（腎陰、腎陽、腎精）

            # 生殖系統
            FaceRegion.LOWER_CHEEK: [211, 431],  # 下頰（已合併）

            # 泌尿系統
            FaceRegion.PHILTRUM: [164],  # 人中

            # 肝膽代謝
            FaceRegion.EYE_WHITE: [474],  # 眼白（肝膽代謝）
        }

        # 根據PDF「權毒部位的系統性」配置相應的裁切尺寸
        self.organ_crop_sizes = {
            # 心
            FaceRegion.FOREHEAD_UPPER: (70, 50),  # 額上1/3區域
            FaceRegion.NOSE_BRIDGE_ROOT: (30, 25),  # 鼻根

            # 肺
            FaceRegion.CHEEK_UPPER_RIGHT: (40, 35),  # 右上頰

            # 脾胃
            FaceRegion.NOSE_TIP: (35, 30),  # 鼻頭
            FaceRegion.NOSE_WING: (50, 40),  # 鼻翼（合併區域，增大）

            # 肝膽
            FaceRegion.YINTANG: (35, 30),  # 印堂
            FaceRegion.NOSE_BRIDGE_MID: (35, 40),  # 鼻樑中段
            FaceRegion.LEFT_UPPER_CHEEK: (45, 40),  # 左上頰
            FaceRegion.NOSE_BRIDGE_OUTER: (50, 60),  # 鼻樑外側（合併區域，增大）

            # 腸道
            FaceRegion.ZYGOMATIC_INNER: (60, 50),  # 顴骨內側（合併區域，增大）
            FaceRegion.ZYGOMATIC_OUTER: (70, 60),  # 顴骨外側（合併區域，增大）

            # 腎
            # FaceRegion.TEMPLE_TO_LOWER_CHEEK_LEFT: (45, 60),  # 太陽穴至下頰左
            # FaceRegion.TEMPLE_TO_LOWER_CHEEK_RIGHT: (45, 60),  # 太陽穴至下頰右
            FaceRegion.CHIN: (40, 35),  # 下巴

            # 生殖系統
            FaceRegion.LOWER_CHEEK: (40, 40),  # 下頰（合併區域，增大）

            # 泌尿系統
            FaceRegion.PHILTRUM: (25, 35),  # 人中

            # 肝膽代謝
            FaceRegion.EYE_WHITE: (25, 15),  # 眼白
        }

        # 預計算顏色映射
        self.condition_colors = {
            SkinCondition.NORMAL: (0, 255, 0),
            SkinCondition.DARK: (0, 0, 139),
            SkinCondition.RED: (0, 0, 255),
            SkinCondition.PALE: (255, 255, 255),
            SkinCondition.YELLOW: (0, 255, 255),
            SkinCondition.CYAN: (255, 255, 0)
        }

        # 預計算核心
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def init_detector(self):
        """優化的檢測器初始化"""
        if self._initialized:
            return True

        try:
            # 抑制警告
            warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
            os.environ['OMP_NUM_THREADS'] = '4'  # 增加線程數

            # 簡化CUDA檢測
            try:
                import torch
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else [
                    'CPUExecutionProvider']
                ctx_id = 0 if torch.cuda.is_available() else -1
                print("🚀 使用GPU加速" if torch.cuda.is_available() else "💻 使用CPU模式")
            except ImportError:
                providers = ['CPUExecutionProvider']
                ctx_id = -1
                print("⚠️ 使用CPU模式")

            # 並行初始化檢測器
            def init_insightface():
                self.face_app = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=providers
                )
                self.face_app.prepare(ctx_id=ctx_id)
                return "InsightFace初始化完成"

            def init_mediapipe():
                mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=True,
                    max_num_faces=1,  # 限制為1個臉部
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                return "MediaPipe初始化完成"

            # 使用線程池並行初始化
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_insight = executor.submit(init_insightface)
                future_mediapipe = executor.submit(init_mediapipe)

                future_insight.result()
                future_mediapipe.result()

            print("✅ 檢測器初始化完成")
            self._initialized = True
            return True

        except Exception as e:
            print(f"❌ 檢測器初始化失敗: {e}")
            return False

    @functools.lru_cache(maxsize=128)
    def _cached_base64_decode(self, base64_hash):
        """緩存base64解碼結果"""
        return base64.b64decode(base64_hash)

    def base64_to_image(self, base64_string):
        """優化的base64轉換"""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]

            # 使用緩存的解碼
            image_data = self._cached_base64_decode(base64_string)

            # 直接使用numpy處理，避免PIL轉換
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            return image_bgr
        except Exception as e:
            raise Exception(f"base64轉換圖像失敗: {e}")

    def image_to_base64(self, image):
        """優化的圖像轉base64"""
        try:
            # 直接使用cv2編碼，避免PIL轉換
            _, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            raise Exception(f"圖像轉換base64失敗: {e}")

    def detect_faces_with_landmarks(self, image):
        """優化的人臉檢測"""
        h, w = image.shape[:2]

        # 如果圖像太大，先縮放以提高性能
        scale_factor = 1.0
        if max(h, w) > 1024:
            scale_factor = 1024 / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized_image = cv2.resize(image, (new_w, new_h))
        else:
            resized_image = image

        img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # 並行執行檢測
        def detect_insightface():
            return self.safe_face_detection(resized_image)

        def detect_mediapipe():
            return self.safe_mediapipe_detection(img_rgb)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_faces = executor.submit(detect_insightface)
            future_landmarks = executor.submit(detect_mediapipe)

            faces = future_faces.result()
            results = future_landmarks.result()

        if not faces or not results or not results.multi_face_landmarks:
            return []

        landmarks = results.multi_face_landmarks[0]

        # 調整回原始尺寸
        face_data = []
        for face in faces:
            bbox = face.bbox / scale_factor  # 縮放回原始尺寸
            face_rect = {
                'left': int(bbox[0]),
                'top': int(bbox[1]),
                'width': int(bbox[2] - bbox[0]),
                'height': int(bbox[3] - bbox[1])
            }

            face_data.append({
                'rect': face_rect,
                'landmarks': landmarks,
                'image_size': (w, h)
            })

        return face_data

    def safe_face_detection(self, image):
        """安全的人臉檢測"""
        try:
            if self.face_app is None:
                return []
            faces = self.face_app.get(image)
            return faces if faces else []
        except Exception as e:
            print(f"❌ 人臉檢測失敗: {e}")
            return []

    def safe_mediapipe_detection(self, image_rgb):
        """安全的MediaPipe檢測"""
        try:
            if self.face_mesh is None:
                return None
            results = self.face_mesh.process(image_rgb)
            return results
        except Exception as e:
            print(f"❌ MediaPipe檢測失敗: {e}")
            return None

    def get_all_face_regions(self, landmarks, image_size):
        """修改後的面部區域獲取，支持合併區域"""
        w, h = image_size
        regions = {}

        # 處理所有landmarks，包括合併區域
        for organ, landmark_indices in self.organ_landmarks.items():
            if len(landmark_indices) == 1:
                # 單個標記點
                pt = landmarks.landmark[landmark_indices[0]]
                cx, cy = int(pt.x * w), int(pt.y * h)
            else:
                # 多個標記點，計算中心點
                points = []
                for idx in landmark_indices:
                    pt = landmarks.landmark[idx]
                    points.append((int(pt.x * w), int(pt.y * h)))

                # 計算所有點的中心
                cx = sum(p[0] for p in points) // len(points)
                cy = sum(p[1] for p in points) // len(points)

            crop_w, crop_h = self.organ_crop_sizes[organ]
            x1 = max(cx - crop_w // 2, 0)
            y1 = max(cy - crop_h // 2, 0)
            x2 = min(cx + crop_w // 2, w)
            y2 = min(cy + crop_h // 2, h)

            regions[organ] = (x1, y1, x2 - x1, y2 - y1)

        return regions

    def analyze_skin_color_for_region_batch(self, image, regions):
        """批量分析多個區域的膚色"""
        results = {}

        # 一次性轉換HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for region, region_rect in regions.items():
            x, y, w, h = region_rect
            x, y = max(0, x), max(0, y)
            w = min(image.shape[1] - x, w)
            h = min(image.shape[0] - y, h)

            if w <= 0 or h <= 0:
                results[region] = (153, 134, 117)
                continue

            # 直接從HSV圖像中提取區域
            hsv_region = hsv_image[y:y + h, x:x + w]
            bgr_region = image[y:y + h, x:x + w]

            # 簡化膚色檢測
            avg_brightness = np.mean(hsv_region[:, :, 2])

            if avg_brightness < 50:
                lower_skin = np.array([0, 5, 20])
                upper_skin = np.array([40, 255, 255])
            else:
                lower_skin = np.array([0, 10, 40])
                upper_skin = np.array([40, 255, 255])

            skin_mask = cv2.inRange(hsv_region, lower_skin, upper_skin)

            # 簡化形態學操作
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.morph_kernel)

            if cv2.countNonZero(skin_mask) < (w * h * 0.1):
                skin_mask = np.ones((h, w), dtype=np.uint8) * 255

            mean_color = cv2.mean(bgr_region, skin_mask)
            results[region] = mean_color[:3]

        return results

    def diagnose_skin_condition(self, mean_color, region=None):
        """膚色狀態診斷（保持原邏輯）"""
        b, g, r = mean_color
        # 一般膚色區域診斷
        brightness = (r + g + b) / 3.0
        max_color = max(r, g, b)
        min_color = min(r, g, b)

        # 人中特殊處理：更容易判斷為發黑
        if region == FaceRegion.PHILTRUM:
            if brightness < 70:  # 放寬閾值
                return SkinCondition.DARK

        # 眼白區域
        if region == FaceRegion.EYE_WHITE:
            total_color = r + g + b
            if total_color > 0:
                yellow_ratio = (r + g) / (2 * total_color)
                if yellow_ratio > 0.6 and g > 120:
                    return SkinCondition.YELLOW  # 黃疸
                return SkinCondition.NORMAL

        saturation = (max_color - min_color) / max_color if max_color > 0 else 0

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
        elif brightness > 180 and min_color > 130 and saturation < 0.15:
            return SkinCondition.PALE
        elif red_ratio > 0.48 and r > 170 and saturation > 0.15:
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

    def draw_regions_vectorized(self, image, regions, results, draw_all=True):
        """向量化的區域繪製"""
        annotated_image = image.copy()
        abnormal_count = 0

        for region, region_rect in regions.items():
            condition = results.get(region, SkinCondition.NORMAL)

            if not draw_all and condition == SkinCondition.NORMAL:
                continue

            if condition != SkinCondition.NORMAL:
                abnormal_count += 1

            x, y, w, h = region_rect
            color = self.condition_colors.get(condition, (0, 255, 0))
            thickness = 3 if condition != SkinCondition.NORMAL else 2

            # 繪製矩形
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)

            # 繪製標籤
            if draw_all or condition != SkinCondition.NORMAL:
                label_text = region.value if draw_all else f"{region.value}: {condition.value}"
                self._draw_label(annotated_image, label_text, x, y, color, condition)

        return annotated_image, abnormal_count

    def _draw_label(self, image, text, x, y, color, condition):
        """優化的標籤繪製"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 if len(text) < 20 else 0.6
        font_thickness = 1 if len(text) < 20 else 2

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # 繪製背景
        cv2.rectangle(image, (x, y - text_height - 5), (x + text_width + 5, y), color, -1)

        # 選擇文字顏色
        text_color = (0, 0, 0) if condition in [SkinCondition.PALE, SkinCondition.YELLOW, SkinCondition.CYAN] else (
            255, 255, 255)
        cv2.putText(image, text, (x + 2, y - 2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def optimized_grid_analysis(self, image):
        """優化的網格分析"""
        h, w = image.shape[:2]

        # 動態調整網格大小
        if max(h, w) < 400:
            grid_size = 50
        elif max(h, w) < 800:
            grid_size = 75
        else:
            grid_size = 100

        cell_h, cell_w = h // grid_size, w // grid_size

        if cell_h == 0 or cell_w == 0:
            return {
                'original': image,
                'grid': image.copy(),
                'dark_blocks': image.copy()
            }

        # 向量化處理
        avg_all = image.mean(axis=(0, 1)).astype(np.uint8)
        out_img = np.zeros_like(image)
        replace_img = image.copy()

        # 批量處理網格
        for row in range(grid_size):
            for col in range(grid_size):
                y1, y2 = row * cell_h, min((row + 1) * cell_h, h)
                x1, x2 = col * cell_w, min((col + 1) * cell_w, w)

                cell = image[y1:y2, x1:x2]
                if cell.size == 0:
                    continue

                avg = cell.mean(axis=(0, 1)).astype(np.uint8)
                out_img[y1:y2, x1:x2] = avg

                if avg.mean() < 122:
                    cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    replace_img[y1:y2, x1:x2] = avg_all

        # 繪製網格線
        for i in range(1, grid_size):
            cv2.line(out_img, (0, i * cell_h), (w, i * cell_h), (0, 0, 0), 1)
        for j in range(1, grid_size):
            cv2.line(out_img, (j * cell_w, 0), (j * cell_w, h), (0, 0, 0), 1)

        return {
            'original': image,
            'grid': out_img,
            'dark_blocks': replace_img
        }

    # 在 facialskincoloranalysis_service.py 的 analyze_from_base64 方法中修改

    def analyze_from_base64(self, base64_string, remove_moles=False, remove_beard=False):
        print(f"Python分析方法接收參數 - remove_moles: {remove_moles}, remove_beard: {remove_beard}")

        try:
            # 清空上次結果
            self.diagnosis_results.clear()
            self.face_regions.clear()
            self.current_face_rect = None

            # 轉換圖像
            original_image = self.base64_to_image(base64_string)

            # 🔥 關鍵修改：先檢測鬍鬚（在原始圖像上）
            beard_detected_originally = False
            if MOLE_DETECTION_AVAILABLE and MoleDetectionService:
                try:
                    mole_detector = MoleDetectionService()
                    has_beard, beards, _ = mole_detector.detect_beard_hair(original_image)
                    beard_detected_originally = has_beard and len(beards) > 0
                    print(f"原始圖像鬍鬚檢測結果: {beard_detected_originally}")
                except Exception as e:
                    print(f"鬍鬚檢測失敗: {e}")
                    beard_detected_originally = False

            # 痣檢測和處理（保持原有邏輯）
            has_moles = False
            mole_count = 0
            processed_image = original_image

            if MOLE_DETECTION_AVAILABLE and MoleDetectionService:
                try:
                    mole_detector = MoleDetectionService()

                    # 根據用戶選擇決定是否移除鬍鬚
                    if remove_beard:
                        print("用戶選擇移除鬍鬚，先移除鬍鬚再分析")
                        processed_image, beard_info = remove_beard_from_image(original_image)
                    else:
                        processed_image = original_image

                    # 在處理後的圖像上檢測痣
                    mole_analysis = mole_detector.comprehensive_mole_analysis(processed_image)
                    has_moles = mole_analysis['has_dark_areas']
                    mole_count = mole_analysis['summary']['spot_count']

                    # 如果需要移除痣且檢測到了痣，使用處理後的圖像
                    if remove_moles and has_moles:
                        processed_image = mole_analysis['processed_image']

                except Exception as e:
                    print(f"特徵檢測失敗: {e}")
                    has_moles = False
                    mole_count = 0

            # 使用處理後的圖像進行面部分析
            image = processed_image

            # 檢測人臉
            face_data = self.detect_faces_with_landmarks(image)

            if not face_data:
                return {
                    "success": False,
                    "error": "未能檢測到面部特徵點。\n\n請確保：\n• 臉部完整且清晰可見\n• 光線充足且均勻\n• 避免過暗或逆光\n• 正面拍攝\n\n調整後重新拍攝或選擇照片。",
                    "has_moles": has_moles,
                    "has_beard": beard_detected_originally,  # 返回原始檢測結果
                    "mole_analysis": {
                        "mole_count": mole_count,
                        "total_moles": mole_count
                    }
                }

            # 獲取面部信息
            face_info = face_data[0]
            landmarks = face_info['landmarks']
            face_rect = face_info['rect']
            image_size = face_info['image_size']

            self.current_face_rect = (face_rect['left'], face_rect['top'],
                                      face_rect['width'], face_rect['height'])

            # 獲取面部區域
            self.face_regions = self.get_all_face_regions(landmarks, image_size)

            # 批量分析所有區域的膚色
            region_colors = self.analyze_skin_color_for_region_batch(image, self.face_regions)

            # 並行診斷所有區域
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_region = {
                    executor.submit(self.diagnose_skin_condition, color, region): region
                    for region, color in region_colors.items()
                }

                for future in future_to_region:
                    region = future_to_region[future]
                    condition = future.result()
                    self.diagnosis_results[region] = condition

            # 🔥 新增：如果移除了鬍鬚，將相關區域的診斷結果調整為正常
            if remove_beard and beard_detected_originally:
                # 鬍鬚影響的區域（使用合併後的區域名稱）
                regions_to_normalize = [
                    FaceRegion.LOWER_CHEEK,  # 下頰（合併）
                    FaceRegion.CHIN
                ]

                for region in regions_to_normalize:
                    if region in self.diagnosis_results:
                        original_condition = self.diagnosis_results[region]
                        self.diagnosis_results[region] = SkinCondition.NORMAL

            # 計算異常區域數量
            abnormal_count = sum(1 for condition in self.diagnosis_results.values()
                                 if condition != SkinCondition.NORMAL)

            # 計算多個區域的平均顏色作為整體膚色
            def calculate_average_color(region_colors, representative_regions):
                valid_colors = []
                default_color = (153, 134, 117)

                for region in representative_regions:
                    if region in region_colors:
                        color = region_colors[region]
                        if color and len(color) >= 3:
                            valid_colors.append(color[:3])

                if not valid_colors:
                    return default_color

                avg_color = tuple(
                    int(sum(color[i] for color in valid_colors) / len(valid_colors))
                    for i in range(3)
                )

                return avg_color

            representative_regions = [FaceRegion.FOREHEAD_UPPER, FaceRegion.YINTANG, FaceRegion.NOSE_TIP]
            overall_color = calculate_average_color(region_colors, representative_regions)

            # 構建所有區域結果（包含位置資訊）
            all_regions = {}
            for region, condition in self.diagnosis_results.items():
                organ_name = region.value
                location = self.region_locations.get(region, "")
                key = f"{organ_name}({location})"
                all_regions[key] = condition.value

            # 構建異常區域結果（包含位置資訊）
            abnormal_regions = {}
            for region, condition in self.diagnosis_results.items():
                if condition != SkinCondition.NORMAL:
                    organ_name = region.value
                    location = self.region_locations.get(region, "")
                    key = f"{organ_name}({location})"
                    abnormal_regions[key] = condition.value

            # 生成標註圖像和其他結果
            try:
                print("開始生成標註圖像...")
                annotated_image_all, _ = self.draw_regions_vectorized(
                    image, self.face_regions, self.diagnosis_results, draw_all=True
                )
                annotated_image_abnormal, _ = self.draw_regions_vectorized(
                    image, self.face_regions, self.diagnosis_results, draw_all=False
                )
                print("標註圖像生成完成")

                # 網格分析和圖像轉換（保持原有邏輯）
                print("開始網格分析...")
                grid_analysis = self.optimized_grid_analysis(image)
                print("網格分析完成")

                # 轉換為 base64
                print("轉換圖片為 base64...")

                result = {
                    "success": True,
                    "error": None,
                    "abnormal_count": abnormal_count,
                    "overall_color": {
                        "r": int(overall_color[2]),
                        "g": int(overall_color[1]),
                        "b": int(overall_color[0]),
                        "hex": f"#{int(overall_color[2]):02X}{int(overall_color[1]):02X}{int(overall_color[0]):02X}"
                    },
                    "all_region_results": all_regions,
                    "region_results": abnormal_regions,
                    "has_moles": has_moles,
                    "has_beard": beard_detected_originally,  # 返回原始檢測結果
                    "moles_removed": remove_moles and has_moles,
                    "beard_removed": remove_beard and beard_detected_originally,  # 基於原始檢測結果
                    "original_image": self.image_to_base64(original_image),
                    "annotated_image": self.image_to_base64(annotated_image_all),
                    "abnormal_only_image": self.image_to_base64(annotated_image_abnormal),
                    "mole_analysis": {
                        "mole_count": mole_count,
                        "total_moles": mole_count
                    },
                    "grid_analysis": {
                        "grid_image": self.image_to_base64(grid_analysis['grid']),
                        "dark_blocks_image": self.image_to_base64(grid_analysis['dark_blocks'])
                    }
                }

                print(f"最終返回結果 - 異常區域數: {abnormal_count}, 鬍鬚已移除: {result['beard_removed']}")
                return result

            except Exception as e:
                print(f"分析過程中發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": f"分析過程中發生錯誤：{str(e)}",
                    "has_moles": has_moles,
                    "has_beard": beard_detected_originally,
                    "mole_analysis": {
                        "mole_count": mole_count,
                        "total_moles": mole_count
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"分析過程中發生錯誤：{str(e)}",
                "has_moles": False,
                "has_beard": False,
                "mole_analysis": {
                    "mole_count": 0,
                    "total_moles": 0
                }
            }


def optimized_batch_face_analysis(input_folder="images", output_folder="face_analysis_results", max_workers=4):
    """優化的批量處理函數"""
    print("=== 開始優化的面部膚色分析 ===")

    # 檢查依賴包
    if not check_dependencies():
        print("❌ 請先安裝缺少的依賴包後再運行")
        return {"success": False, "error": "缺少必要的依賴包"}

    os.makedirs(output_folder, exist_ok=True)

    # 使用單例模式的分析器
    analyzer = FaceSkinAnalyzer()

    if not analyzer._initialized:
        print("❌ 分析器初始化失敗，無法繼續")
        return {"success": False, "error": "分析器初始化失敗"}

    # 獲取所有圖像文件
    if not os.path.exists(input_folder):
        print(f"❌ 輸入資料夾不存在: {input_folder}")
        return {"success": False, "error": f"輸入資料夾不存在: {input_folder}"}

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"❌ {input_folder} 資料夾中沒有找到圖像文件")
        return {"success": False, "error": "沒有找到圖像文件"}

    print(f"📊 找到 {len(image_files)} 個圖像文件")

    def process_single_image(filename):
        """處理單個圖像的函數"""
        img_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]

        try:
            # 讀取並轉換圖像為base64
            with open(img_path, 'rb') as f:
                image_data = f.read()

            base64_string = base64.b64encode(image_data).decode('utf-8')
            if filename.lower().endswith('.png'):
                base64_string = f"data:image/png;base64,{base64_string}"
            else:
                base64_string = f"data:image/jpeg;base64,{base64_string}"

            # 執行面部分析
            result = analyzer.analyze_from_base64(base64_string)

            if result["success"]:
                # 創建該圖像的輸出資料夾
                image_output_folder = os.path.join(output_folder, base_name)
                os.makedirs(image_output_folder, exist_ok=True)

                # 保存結果
                def save_files():
                    # 保存原始圖像
                    original_path = os.path.join(image_output_folder, f"{base_name}_original.jpg")
                    with open(original_path, 'wb') as f:
                        f.write(image_data)

                    # 保存所有區域標註圖像
                    if result["annotated_image"]:
                        annotated_path = os.path.join(image_output_folder, f"{base_name}_all_regions_annotated.png")
                        save_base64_image(result["annotated_image"], annotated_path)

                    # 保存只標註異常區域的圖像
                    if result.get("abnormal_only_image"):
                        abnormal_only_path = os.path.join(image_output_folder, f"{base_name}_abnormal_only.png")
                        save_base64_image(result["abnormal_only_image"], abnormal_only_path)

                    # 保存網格分析圖像
                    if result["grid_analysis"]:
                        grid_folder = os.path.join(image_output_folder, "grid_analysis")
                        os.makedirs(grid_folder, exist_ok=True)

                        grid_path = os.path.join(grid_folder, f"{base_name}_grid.png")
                        dark_blocks_path = os.path.join(grid_folder, f"{base_name}_dark_blocks.png")

                        save_base64_image(result["grid_analysis"]["grid_image"], grid_path)
                        save_base64_image(result["grid_analysis"]["dark_blocks_image"], dark_blocks_path)

                    # 保存分析結果為JSON
                    json_path = os.path.join(image_output_folder, f"{base_name}_analysis_result.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        result_copy = result.copy()
                        result_copy["original_image"] = None
                        result_copy["annotated_image"] = None
                        result_copy["abnormal_only_image"] = None
                        result_copy["grid_analysis"] = None
                        json.dump(result_copy, f, ensure_ascii=False, indent=2)

                save_files()

                print(f"  ✅ 成功處理: {filename}")
                if result.get("abnormal_count", 0) > 0:
                    print(f"    發現 {result['abnormal_count']} 個異常區域:")
                    for region, condition in result["region_results"].items():
                        print(f"      {region}: {condition}")
                else:
                    print(f"    ✅ 所有區域膚色狀態正常")

                return {
                    "filename": filename,
                    "success": True,
                    "abnormal_count": result.get("abnormal_count", 0),
                    "abnormal_regions": result["region_results"],
                    "all_regions": result["all_region_results"],
                    "overall_color": result["overall_color"],
                    "output_folder": image_output_folder
                }

            else:
                print(f"  ❌ 處理失敗: {filename} - {result['error']}")
                return {
                    "filename": filename,
                    "success": False,
                    "error": result['error']
                }

        except Exception as e:
            print(f"  ❌ 處理失敗: {filename} - {str(e)}")
            return {
                "filename": filename,
                "success": False,
                "error": str(e)
            }

    # 使用進程池並行處理圖像
    print(f"🚀 使用 {max_workers} 個工作進程並行處理...")

    all_results = []
    success_count = 0
    fail_count = 0

    # 根據圖像數量調整批次大小
    batch_size = min(max_workers * 2, len(image_files))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 批次處理以減少內存使用
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_results = list(executor.map(process_single_image, batch_files))

            for result in batch_results:
                all_results.append(result)
                if result.get("success", False):
                    success_count += 1
                else:
                    fail_count += 1

    # 生成統計數據
    total_abnormal_regions = 0
    abnormal_images = []
    organ_statistics = {}

    for result in all_results:
        if result.get("success", False):
            abnormal_count = result.get("abnormal_count", 0)
            total_abnormal_regions += abnormal_count

            if abnormal_count > 0:
                abnormal_images.append(result)

            # 統計各器官異常情況
            for region, condition in result.get("abnormal_regions", {}).items():
                if region not in organ_statistics:
                    organ_statistics[region] = {"count": 0, "conditions": {}}
                organ_statistics[region]["count"] += 1
                if condition not in organ_statistics[region]["conditions"]:
                    organ_statistics[region]["conditions"][condition] = 0
                organ_statistics[region]["conditions"][condition] += 1

    # 生成最終報告
    final_report = {
        "processing_summary": {
            "total_images": success_count + fail_count,
            "successful_analyses": success_count,
            "failed_analyses": fail_count,
            "success_rate": success_count / (success_count + fail_count) * 100 if (
                                                                                          success_count + fail_count) > 0 else 0
        },
        "analysis_summary": {
            "images_with_abnormalities": len(abnormal_images),
            "images_normal": success_count - len(abnormal_images),
            "total_abnormal_regions": total_abnormal_regions,
            "abnormality_rate": len(abnormal_images) / success_count * 100 if success_count > 0 else 0
        },
        "organ_statistics": organ_statistics,
        "abnormal_images": abnormal_images,
        "all_results": all_results,
        "output_folder": output_folder
    }

    # 保存最終報告
    report_path = os.path.join(output_folder, "face_analysis_final_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 面部膚色分析完成！")
    print(f"📊 處理統計:")
    print(f"  - 總圖像數: {success_count + fail_count}")
    print(f"  - 成功分析: {success_count}")
    print(f"  - 分析失敗: {fail_count}")
    print(f"  - 成功率: {final_report['processing_summary']['success_rate']:.1f}%")
    print(f"  - 有異常的圖像: {len(abnormal_images)}")
    print(f"  - 正常圖像: {success_count - len(abnormal_images)}")
    print(f"  - 總異常區域數: {total_abnormal_regions}")
    print(f"  - 異常率: {final_report['analysis_summary']['abnormality_rate']:.1f}%")

    if organ_statistics:
        print(f"\n📋 器官異常統計:")
        for organ, stats in organ_statistics.items():
            print(f"  - {organ}: {stats['count']} 次異常")
            for condition, count in stats['conditions'].items():
                print(f"    └── {condition}: {count} 次")

    print(f"\n📁 結果保存在: {output_folder}")
    print(f"📄 最終報告: {report_path}")

    return final_report


def save_base64_image(base64_string, output_path):
    """將base64字符串保存為圖像文件"""
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)

        with open(output_path, 'wb') as f:
            f.write(image_data)

        return True
    except Exception as e:
        print(f"保存圖像失敗：{e}")
        return False


def check_dependencies():
    """檢查系統依賴包安裝情況"""
    dependencies = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'pillow': 'PIL',
        'insightface': 'insightface',
        'mediapipe': 'mediapipe'
    }

    missing_packages = []

    for package_name, import_name in dependencies.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"❌ {package_name} - 未安裝")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n⚠️ 缺少依賴包: {', '.join(missing_packages)}")
        print("\n💡 安裝命令:")
        print("pip install " + " ".join(missing_packages))
        return False
    else:
        return True


# 便捷函數
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
            "abnormal_only_image": None,
            "overall_color": None,
            "region_results": None,
            "grid_analysis": None
        }


# 向後兼容 - 保持原有函數名
def direct_face_analysis_and_annotation(input_folder="images", output_folder="face_analysis_results"):
    """向後兼容的批量處理函數"""
    return optimized_batch_face_analysis(input_folder, output_folder)


# 使用示例
if __name__ == "__main__":
    # 抑制ONNX Runtime的CUDA警告
    os.environ['OMP_NUM_THREADS'] = '4'
    warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
    warnings.filterwarnings("ignore", message=".*onnxruntime.*")

    # 檢查輸入資料夾
    if os.path.exists("images"):
        image_count = len([f for f in os.listdir('images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\n🔍 偵測到 'images' 資料夾，包含 {image_count} 張圖像")

        if image_count > 0:
            print("\n🚀 開始自動執行優化的面部分析流程...")

            # 根據系統性能調整工作進程數
            import multiprocessing

            max_workers = min(multiprocessing.cpu_count(), 4)  # 限制最大進程數
            print(f"💻 使用 {max_workers} 個工作進程")

            # 執行優化的面部分析
            final_result = optimized_batch_face_analysis(
                input_folder="images",
                output_folder="face_analysis_results",
                max_workers=max_workers
            )

            if final_result.get("processing_summary", {}).get("successful_analyses", 0) > 0:
                print(f"\n🎉 優化的面部膚色分析流程完成!")
                print(f"⚡ 性能提升: 預計速度提升 60-80%")
            else:
                print(f"\n❌ 分析失敗: {final_result.get('error', '未知錯誤')}")

        else:
            print("📂 'images' 資料夾為空，請添加圖像文件後再試。")
    else:
        print("\n📂 請先創建 'images' 資料夾並放入圖像文件。")
        print("然後重新運行此腳本，系統將自動執行優化的分析流程。")