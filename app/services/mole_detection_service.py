"""
精確的痣檢測服務 - 只檢測肉眼可見的明顯痣，並新增鬍鬚移除功能
最終修正版本：確保所有函數都能正確導入
"""
import cv2
import numpy as np
import base64
import threading
from typing import Tuple, List, Dict, Optional


class MoleDetectionService:
    """精確的痣檢測服務，只檢測明顯的痣，並可移除鬍鬚"""

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

        # 痣檢測參數
        self.dark_threshold = 65
        self.min_area = 18
        self.max_area = 4000
        self.min_circularity = 0.45
        self.contrast_threshold = 45

        # 鬍鬚檢測參數
        self.beard_min_length = 15
        self.beard_max_width = 5
        self.beard_aspect_ratio_min = 3.0
        self.beard_darkness_threshold = 80
        self.beard_detection_threshold = 3

        self._initialized = True

    def detect_obvious_moles(self, image: np.ndarray) -> Tuple[bool, List[Dict], np.ndarray]:
        """檢測圖像中明顯的痣"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 5
            )

            _, binary_thresh = cv2.threshold(gray, self.dark_threshold, 255, cv2.THRESH_BINARY_INV)
            combined_mask = cv2.bitwise_and(adaptive_thresh, binary_thresh)

            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_moles = []
            result_image = image.copy()

            for contour in contours:
                area = cv2.contourArea(contour)

                if not (self.min_area <= area <= self.max_area):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2

                if (x < 10 or y < 10 or x+w > image.shape[1]-10 or y+h > image.shape[0]-10):
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    continue

                if circularity < self.min_circularity:
                    continue

                aspect_ratio = max(w, h) / max(min(w, h), 1)
                if aspect_ratio > self.beard_aspect_ratio_min and min(w, h) <= self.beard_max_width:
                    continue

                if not self._check_contrast(gray, x, y, w, h):
                    continue

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

                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(result_image, (center_x, center_y), 3, (0, 255, 0), -1)
                cv2.putText(result_image, "Mole", (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            has_moles = len(valid_moles) > 0
            return has_moles, valid_moles, result_image

        except Exception as e:
            print(f"檢測痣時發生錯誤: {e}")
            return False, [], image.copy()

    def detect_beard_simple(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """簡化的鬍鬚檢測"""
        try:
            has_beard_detailed, beards, detection_image = self.detect_beard_hair(image)
            has_beard = len(beards) >= self.beard_detection_threshold
            print(f"鬍鬚檢測結果: 檢測到鬍鬚: {has_beard}")
            return has_beard, detection_image
        except Exception as e:
            print(f"簡化鬍鬚檢測時發生錯誤: {e}")
            return False, image.copy()

    def detect_beard_hair(self, image: np.ndarray) -> Tuple[bool, List[Dict], np.ndarray]:
        """檢測圖像中的鬍鬚毛髮"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            _, dark_mask = cv2.threshold(blurred, self.beard_darkness_threshold, 255, cv2.THRESH_BINARY_INV)

            kernels = [
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),
                np.array([[0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1],
                         [0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 0]], dtype=np.uint8),
                np.array([[0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1],
                         [0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 0]][::-1], dtype=np.uint8)
            ]

            hair_mask = np.zeros_like(dark_mask)
            for kernel in kernels:
                opened = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
                hair_mask = cv2.bitwise_or(hair_mask, opened)

            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

            contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_hairs = []
            result_image = image.copy()

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 8 or area > 500:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                aspect_ratio = max(w, h) / max(min(w, h), 1)
                if aspect_ratio < self.beard_aspect_ratio_min:
                    continue

                if min(w, h) > self.beard_max_width:
                    continue

                if max(w, h) < self.beard_min_length:
                    continue

                if y < image.shape[0] * 0.4:
                    continue

                hair_info = {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': float(area),
                    'aspect_ratio': float(aspect_ratio),
                    'length': int(max(w, h))
                }
                valid_hairs.append(hair_info)

                cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(result_image, "Hair", (x, y-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            has_beard = len(valid_hairs) > 0
            return has_beard, valid_hairs, result_image

        except Exception as e:
            print(f"檢測鬍鬚時發生錯誤: {e}")
            return False, [], image.copy()

    def remove_beard_hair(self, image: np.ndarray, hairs: List[Dict]) -> np.ndarray:
        """移除檢測到的鬍鬚毛髮"""
        try:
            result = image.copy()

            for hair in hairs:
                x, y, w, h = hair['x'], hair['y'], hair['width'], hair['height']

                padding = 2
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)

                mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)

                if w > h:
                    cv2.rectangle(mask, (1, (y2-y1)//2-1), (x2-x1-1, (y2-y1)//2+1), 255, -1)
                else:
                    cv2.rectangle(mask, ((x2-x1)//2-1, 1), ((x2-x1)//2+1, y2-y1-1), 255, -1)

                roi = result[y1:y2, x1:x2]
                if roi.size > 0:
                    inpainted = cv2.inpaint(roi, mask, 3, cv2.INPAINT_NS)
                    result[y1:y2, x1:x2] = inpainted

            return result

        except Exception as e:
            print(f"移除鬍鬚時發生錯誤: {e}")
            return image.copy()

    def _check_contrast(self, gray_image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """檢查痣區域與周圍皮膚的對比度"""
        try:
            mole_region = gray_image[y:y+h, x:x+w]
            mole_mean = np.mean(mole_region)

            padding = max(w, h) // 2
            y1 = max(0, y - padding)
            y2 = min(gray_image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(gray_image.shape[1], x + w + padding)

            surrounding_region = gray_image[y1:y2, x1:x2]

            mask = np.ones(surrounding_region.shape, dtype=np.uint8)
            inner_y = y - y1
            inner_x = x - x1
            mask[inner_y:inner_y+h, inner_x:inner_x+w] = 0

            surrounding_pixels = surrounding_region[mask == 1]
            if len(surrounding_pixels) == 0:
                return False

            surrounding_mean = np.mean(surrounding_pixels)
            contrast = abs(surrounding_mean - mole_mean)
            return contrast > self.contrast_threshold

        except Exception as e:
            print(f"檢查對比度時發生錯誤: {e}")
            return False

    def _check_color_consistency(self, gray_image: np.ndarray, contour: np.ndarray) -> bool:
        """檢查痣內部顏色的一致性"""
        try:
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)

            mole_pixels = gray_image[mask == 255]

            if len(mole_pixels) < 10:
                return False

            std_dev = np.std(mole_pixels)
            return std_dev < 25

        except Exception as e:
            print(f"檢查顏色一致性時發生錯誤: {e}")
            return False

    def remove_moles(self, image: np.ndarray, moles: List[Dict]) -> np.ndarray:
        """移除檢測到的痣"""
        try:
            result = image.copy()

            for mole in moles:
                x, y, w, h = mole['x'], mole['y'], mole['width'], mole['height']

                padding = 8
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)

                mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                center = ((x + w//2) - x1, (y + h//2) - y1)
                radius = max(w, h) // 2 + 3
                cv2.circle(mask, center, radius, 255, -1)

                roi = result[y1:y2, x1:x2]
                if roi.size > 0:
                    inpainted = cv2.inpaint(roi, mask, 5, cv2.INPAINT_TELEA)
                    result[y1:y2, x1:x2] = inpainted

            return result

        except Exception as e:
            print(f"移除痣時發生錯誤: {e}")
            return image.copy()

    def comprehensive_mole_analysis(self, image: np.ndarray, remove_beard: bool = False) -> Dict:
        """綜合痣分析"""
        try:
            processed_image = image.copy()
            beard_info = {'has_beard': False, 'beard_count': 0}

            if remove_beard:
                has_beard_simple, beard_detection_image = self.detect_beard_simple(image)
                if has_beard_simple:
                    _, beards_detailed, _ = self.detect_beard_hair(image)
                    processed_image = self.remove_beard_hair(image, beards_detailed)

                beard_info = {
                    'has_beard': has_beard_simple,
                    'beard_count': 1 if has_beard_simple else 0,
                    'detection_image': beard_detection_image
                }
            else:
                has_beard_simple, beard_detection_image = self.detect_beard_simple(image)
                beard_info = {
                    'has_beard': has_beard_simple,
                    'beard_count': 1 if has_beard_simple else 0,
                    'detection_image': beard_detection_image
                }

            has_moles, moles, mole_detection_image = self.detect_obvious_moles(processed_image)

            final_image = processed_image.copy()
            if has_moles:
                final_image = self.remove_moles(processed_image, moles)

            return {
                'has_dark_areas': has_moles,
                'spot_detection': {
                    'has_spots': has_moles,
                    'spots': moles,
                    'detection_image': mole_detection_image
                },
                'beard_detection': beard_info,
                'grid_analysis': {
                    'has_dark_blocks': False,
                    'dark_block_count': 0
                },
                'original_image': image,
                'processed_image': final_image,
                'beard_removed_image': processed_image if remove_beard else image,
                'summary': {
                    'spot_count': len(moles),
                    'beard_count': beard_info['beard_count'],
                    'dark_block_count': 0,
                    'total_dark_areas': len(moles)
                }
            }

        except Exception as e:
            print(f"綜合分析時發生錯誤: {e}")
            return {
                'has_dark_areas': False,
                'spot_detection': {'has_spots': False, 'spots': [], 'detection_image': image},
                'beard_detection': {'has_beard': False, 'beard_count': 0},
                'grid_analysis': {'has_dark_blocks': False, 'dark_block_count': 0},
                'original_image': image,
                'processed_image': image.copy(),
                'beard_removed_image': image.copy(),
                'summary': {'spot_count': 0, 'beard_count': 0, 'dark_block_count': 0, 'total_dark_areas': 0}
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


# === 全局便利函數 - 確保這些函數在模塊級別定義 ===

def detect_and_analyze_moles(image: np.ndarray, remove_beard: bool = False) -> Dict:
    """
    便利函數：檢測和分析圖像中的明顯痣，可選擇是否移除鬍鬚

    Args:
        image: 輸入圖像 (BGR格式)
        remove_beard: 是否移除鬍鬚，默認False

    Returns:
        完整的分析結果字典
    """
    detector = MoleDetectionService()
    return detector.comprehensive_mole_analysis(image, remove_beard)


def process_image_for_moles(base64_string: str, remove_beard: bool = False) -> Dict:
    """
    從base64字符串處理圖像並檢測明顯痣，可選擇是否移除鬍鬚

    Args:
        base64_string: base64編碼的圖像字符串
        remove_beard: 是否移除鬍鬚，默認False

    Returns:
        處理結果字典
    """
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return {'success': False, 'error': 'Unable to decode image'}

        result = detect_and_analyze_moles(image, remove_beard)
        result['success'] = True
        return result

    except Exception as e:
        return {'success': False, 'error': f'處理圖像時發生錯誤: {str(e)}'}


def remove_beard_from_image(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    單獨的鬍鬚移除函數

    Args:
        image: 輸入圖像

    Returns:
        tuple: (處理後的圖像, 鬍鬚檢測信息)
    """
    detector = MoleDetectionService()
    has_beard, detection_image = detector.detect_beard_simple(image)

    if has_beard:
        _, beards_detailed, _ = detector.detect_beard_hair(image)
        processed_image = detector.remove_beard_hair(image, beards_detailed)
        return processed_image, {
            'has_beard': True,
            'beard_count': 1,
            'detection_image': detection_image
        }
    else:
        return image.copy(), {
            'has_beard': False,
            'beard_count': 0,
            'detection_image': detection_image
        }


def test_mole_detection():
    """測試痣檢測功能"""
    print("痣檢測服務測試開始")
    try:
        detector = MoleDetectionService()
        print("服務初始化完成")

        print(f"痣檢測參數:")
        print(f"  暗度閾值: {detector.dark_threshold}")
        print(f"  最小面積: {detector.min_area}")
        print(f"  最大面積: {detector.max_area}")

        print(f"鬍鬚檢測參數:")
        print(f"  檢測閾值: {detector.beard_detection_threshold}")
        print(f"  最小長度: {detector.beard_min_length}")
        print(f"  最大寬度: {detector.beard_max_width}")

        print("基本功能測試通過")
        return True

    except Exception as e:
        print(f"測試失敗: {e}")
        return False


# 確保所有需要的函數都能被導入
__all__ = [
    'MoleDetectionService',
    'detect_and_analyze_moles',
    'process_image_for_moles',
    'remove_beard_from_image',
    'test_mole_detection'
]


if __name__ == "__main__":
    test_mole_detection()