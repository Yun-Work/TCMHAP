"""
精確的痣檢測服務 - 只檢測肉眼可見的明顯痣
"""
import cv2
import numpy as np
import base64
from typing import Tuple, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading


class MoleDetectionService:
    """精確的痣檢測服務，只檢測明顯的痣"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """單例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        # 更嚴格的檢測參數 - 只檢測明顯的痣
        self.dark_threshold = 65  # 適中的閾值
        self.min_area = 18  # 適中的最小面積，過濾小噪點
        self.max_area = 4000  # 適中的最大面積
        self.min_circularity = 0.45  # 適中的圓形度要求
        self.contrast_threshold = 45  # 適中的對比度要求
        self._initialized = True

    def detect_obvious_moles(self, image: np.ndarray) -> Tuple[bool, List[Dict], np.ndarray]:
        """
        檢測圖像中明顯的痣（肉眼可見的）

        Args:
            image: 輸入圖像 (BGR格式)

        Returns:
            tuple: (是否檢測到痣, 痣信息列表, 檢測結果圖像)
        """
        try:
            # 轉換為灰度圖
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用高斯模糊減少噪音，但保持邊緣清晰
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # 計算自適應閾值，根據局部區域特性
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 5
            )

            # 同時使用固定閾值檢測非常暗的區域
            _, binary_thresh = cv2.threshold(gray, self.dark_threshold, 255, cv2.THRESH_BINARY_INV)

            # 合併兩種閾值結果，只保留同時滿足的區域
            combined_mask = cv2.bitwise_and(adaptive_thresh, binary_thresh)

            # 形態學操作 - 去除噪點但保留真實的痣
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

            # 查找輪廓
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 篩選明顯的痣
            valid_moles = []
            result_image = image.copy()

            for contour in contours:
                area = cv2.contourArea(contour)

                # 面積過濾 - 只保留適中大小的區域
                if not (self.min_area <= area <= self.max_area):
                    continue

                # 計算邊界框和中心
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2

                # 檢查邊界
                if (x < 10 or y < 10 or x+w > image.shape[1]-10 or y+h > image.shape[0]-10):
                    continue  # 跳過邊緣區域，可能是陰影

                # 計算圓形度
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    continue

                # 圓形度過濾 - 痣通常比較圓
                if circularity < self.min_circularity:
                    continue

                # 對比度檢查 - 確保痣與周圍皮膚有明顯差異
                if not self._check_contrast(gray, x, y, w, h):
                    continue

                # 顏色一致性檢查 - 痣內部顏色應該相對一致
                if not self._check_color_consistency(gray, contour):
                    continue

                mole_info = {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': float(area),
                    'circularity': float(circularity),
                    'center': (int(center_x), int(center_y))
                }
                valid_moles.append(mole_info)

                # 在結果圖像上標記
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(result_image, (center_x, center_y), 3, (0, 255, 0), -1)
                cv2.putText(result_image, f"Mole", (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            has_moles = len(valid_moles) > 0

            return has_moles, valid_moles, result_image

        except Exception as e:
            print(f"檢測痣時發生錯誤: {e}")
            return False, [], image.copy()

    def _check_contrast(self, gray_image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """檢查痣區域與周圍皮膚的對比度"""
        try:
            # 痣區域
            mole_region = gray_image[y:y+h, x:x+w]
            mole_mean = np.mean(mole_region)

            # 周圍區域（擴展邊界）
            padding = max(w, h) // 2
            y1 = max(0, y - padding)
            y2 = min(gray_image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(gray_image.shape[1], x + w + padding)

            surrounding_region = gray_image[y1:y2, x1:x2]

            # 創建遮罩，排除痣區域本身
            mask = np.ones(surrounding_region.shape, dtype=np.uint8)
            inner_y = y - y1
            inner_x = x - x1
            mask[inner_y:inner_y+h, inner_x:inner_x+w] = 0

            # 計算周圍區域的平均亮度
            surrounding_pixels = surrounding_region[mask == 1]
            if len(surrounding_pixels) == 0:
                return False

            surrounding_mean = np.mean(surrounding_pixels)

            # 檢查對比度
            contrast = abs(surrounding_mean - mole_mean)

            return contrast > self.contrast_threshold

        except Exception as e:
            print(f"檢查對比度時發生錯誤: {e}")
            return False

    def _check_color_consistency(self, gray_image: np.ndarray, contour: np.ndarray) -> bool:
        """檢查痣內部顏色的一致性"""
        try:
            # 創建痣區域的遮罩
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)

            # 獲取痣區域的像素值
            mole_pixels = gray_image[mask == 255]

            if len(mole_pixels) < 10:
                return False

            # 計算標準差，檢查一致性
            std_dev = np.std(mole_pixels)

            # 痣內部顏色應該相對一致（標準差不應太大）
            return std_dev < 25

        except Exception as e:
            print(f"檢查顏色一致性時發生錯誤: {e}")
            return False

    def remove_moles(self, image: np.ndarray, moles: List[Dict]) -> np.ndarray:
        """
        移除檢測到的痣

        Args:
            image: 原始圖像
            moles: 痣信息列表

        Returns:
            處理後的圖像
        """
        try:
            result = image.copy()

            for mole in moles:
                x, y, w, h = mole['x'], mole['y'], mole['width'], mole['height']

                # 擴展處理區域以確保完全覆蓋
                padding = 8
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)

                # 創建圓形遮罩（痣通常是圓形的）
                mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                center = ((x + w//2) - x1, (y + h//2) - y1)
                radius = max(w, h) // 2 + 3
                cv2.circle(mask, center, radius, 255, -1)

                # 使用修復算法填補
                roi = result[y1:y2, x1:x2]
                if roi.size > 0:
                    # 使用 TELEA 算法，效果更自然
                    inpainted = cv2.inpaint(roi, mask, 5, cv2.INPAINT_TELEA)
                    result[y1:y2, x1:x2] = inpainted

            return result

        except Exception as e:
            print(f"移除痣時發生錯誤: {e}")
            return image.copy()

    def comprehensive_mole_analysis(self, image: np.ndarray) -> Dict:
        """
        綜合痣分析（簡化版，只檢測明顯痣）

        Args:
            image: 輸入圖像

        Returns:
            完整的分析結果
        """
        try:
            # 只進行痣檢測
            has_moles, moles, detection_image = self.detect_obvious_moles(image)

            # 如果檢測到痣，生成處理後的圖像
            processed_image = image.copy()
            if has_moles:
                processed_image = self.remove_moles(image, moles)

            return {
                'has_dark_areas': has_moles,  # 為了兼容性保持這個名稱
                'spot_detection': {
                    'has_spots': has_moles,
                    'spots': moles,
                    'detection_image': detection_image
                },
                'grid_analysis': {
                    'has_dark_blocks': False,  # 不再進行網格分析
                    'dark_block_count': 0
                },
                'original_image': image,
                'processed_image': processed_image,
                'summary': {
                    'spot_count': len(moles),
                    'dark_block_count': 0,  # 不再檢測暗色區域
                    'total_dark_areas': len(moles)
                }
            }

        except Exception as e:
            print(f"綜合分析時發生錯誤: {e}")
            return {
                'has_dark_areas': False,
                'spot_detection': {'has_spots': False, 'spots': [], 'detection_image': image},
                'grid_analysis': {'has_dark_blocks': False, 'dark_block_count': 0},
                'original_image': image,
                'processed_image': image.copy(),
                'summary': {'spot_count': 0, 'dark_block_count': 0, 'total_dark_areas': 0}
            }

    def image_to_base64(self, image: np.ndarray) -> str:
        """將圖像轉換為base64字符串"""
        try:
            _, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"圖像轉base64失敗: {e}")
            return ""


# 便利函數
def detect_and_analyze_moles(image: np.ndarray) -> Dict:
    """
    便利函數：檢測和分析圖像中的明顯痣

    Args:
        image: 輸入圖像 (BGR格式)

    Returns:
        完整的分析結果字典
    """
    detector = MoleDetectionService()
    return detector.comprehensive_mole_analysis(image)


def process_image_for_moles(base64_string: str) -> Dict:
    """
    從base64字符串處理圖像並檢測明顯痣

    Args:
        base64_string: base64編碼的圖像字符串

    Returns:
        處理結果字典
    """
    try:
        # 解碼base64
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return {'success': False, 'error': 'Unable to decode image'}

        # 執行分析
        result = detect_and_analyze_moles(image)

        # 轉換圖像為base64（簡化版，不回傳圖像）
        result['success'] = True
        return result

    except Exception as e:
        return {'success': False, 'error': f'處理圖像時發生錯誤: {str(e)}'}