#!/usr/bin/env python3
"""
Simple Photometry Calibration with Matplotlib
마우스로 구슬 중심과 하이라이트 영역을 선택하여 조명 방향을 자동 계산
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.widgets import Button
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from scipy import ndimage
from scipy.ndimage import label, find_objects, binary_fill_holes
from skimage.measure import find_contours
import traceback
from pathlib import Path
import cv2
import tifffile as tiff
import light_vec_calculator as lvcalc
import debug_light_vectors as debug_vis
import debug_image_extraction as debug_img


class SimpleCalibrationMatplotlib:
    """matplotlib을 사용한 단순한 캘리브레이션 클래스"""
    
    def __init__(self):
        self.images = []
        self.sphere_centers = []
        self.highlight_regions = []
        self.highlight_contours = []  # Auto 모드에서 감지된 blob contours
        self.sphere_diameter = 0
        self.pixel_resolution = 0
        self.light_directions = []
        self.light_matrix = None
        self.errors = 'single_sphere'
        
        # Rectification 관련
        self.map_x = None
        self.map_y = None
        self.rectified_images = []  # rectified 이미지 배열 저장
        
        # 선택 상태 변수들
        self.current_image = None
        self.current_image_path = ""
        self.temp_center = None
        self.temp_highlight = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.auto_mode = False  # Auto 모드 여부
        self.detected_blob = None  # Auto 모드에서 감지된 blob (contour 좌표)
        self.detected_blob_mask = None  # Auto 모드에서 감지된 blob의 mask
        self.last_mouse_pos: Optional[Tuple[float, float]] = None  # 마지막 커서 위치
        
        # matplotlib 관련
        self.fig = None
        self.ax = None
        self.sphere_circle = None
        self.highlight_rect = None
        self.blob_polygon = None  # Auto 모드에서 blob 표시용
        
        # 줌 관련
        self.zoom_factor = 1.7  # 줌인/줌아웃 배율
        self.min_zoom = 1  # 최소 줌 (전체 이미지보다 작게)
        self.max_zoom = 33.0  # 최대 줌
        self.current_xlim = None  # 현재 뷰 x 범위
        self.current_ylim = None  # 현재 뷰 y 범위
    
    def _debug_print_stack(self, label: str, limit: int = 8):
        """디버깅용 스택 출력"""
        print(f"{label}")
        try:
            traceback.print_stack(limit=limit)
        except Exception as e:
            print(f"[DEBUG] STACK TRACE FAILED: {e}")
    
    def _load_map_pair(self, map_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """map_dir에서 *map_x.tiff, *map_y.tiff 패턴으로 파일을 찾아 float32로 로드."""
        mx_matches = sorted(map_dir.glob('*map_x.tiff'))
        my_matches = sorted(map_dir.glob('*map_y.tiff'))
        
        if not mx_matches:
            raise FileNotFoundError(f"Missing map_x file in {map_dir}: no file matching '*map_x.tiff'")
        if not my_matches:
            raise FileNotFoundError(f"Missing map_y file in {map_dir}: no file matching '*map_y.tiff'")
        
        mx_path = mx_matches[0]
        my_path = my_matches[0]
        
        map_x = tiff.imread(str(mx_path)).astype(np.float32, copy=False)
        map_y = tiff.imread(str(my_path)).astype(np.float32, copy=False)
        if map_x.shape != map_y.shape:
            raise ValueError(f"map_x and map_y shape mismatch: {map_x.shape} vs {map_y.shape}")
        return map_x, map_y
    
    def _remap_image(self, src_img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, 
                     interpolation: int = cv2.INTER_LINEAR, 
                     border_mode: int = cv2.BORDER_CONSTANT, 
                     border_value: float = 0.0) -> np.ndarray:
        """cv2.remap을 사용해 src_img를 재배치. 채널 수/비트심도 보존."""
        dst = cv2.remap(src_img, map_x, map_y, interpolation=interpolation, 
                       borderMode=border_mode, borderValue=border_value)
        return dst
    
    def load_remap_maps(self, map_dir: str):
        """remap map 파일들을 로드"""
        map_dir_path = Path(map_dir)
        self.map_x, self.map_y = self._load_map_pair(map_dir_path)
        print(f"Remap maps loaded: size={self.map_x.shape[::-1]} (W,H)")
    
    def load_images(self, image_pattern: str, apply_rectification: bool = False) -> List[str]:
        """glob 패턴으로 이미지들을 로드하고 선택적으로 rectification 적용"""
        image_paths = sorted(glob.glob(image_pattern))
        
        if not image_paths:
            raise ValueError(f"Images not found: {image_pattern}")
        
        print(f"Loaded images: {len(image_paths)}")
        self.images = image_paths
        self.rectified_images = []
        
        # Rectification 적용
        if apply_rectification:
            if self.map_x is None or self.map_y is None:
                raise ValueError("Remap maps not loaded. Call load_remap_maps() first.")
            
            print("Applying rectification to images...")
            for i, image_path in enumerate(image_paths):
                # 이미지 로드
                pil_image = Image.open(image_path)
                img_array = np.array(pil_image)
                
                # Rectification 적용
                rectified = self._remap_image(img_array, self.map_x, self.map_y)
                self.rectified_images.append(rectified)
                print(f"  Rectified image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        return image_paths
    
    def on_click(self, event):
        """마우스 클릭 이벤트"""
        self._update_mouse_position(event)
        if event.inaxes != self.ax:
            return
        
        # 더블 클릭 처리
        if event.dblclick:
            print(f"[DEBUG] DOUBLE CLICK EVENT: Resetting view")
            self.reset_view()
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        if self.temp_center is None:
            # 구슬 중심 선택
            self.temp_center = (x, y)
            print(f"Sphere center selected: ({x}, {y})")
            self.update_display()
        else:
            # 하이라이트 영역 시작점
            self.drawing = True
            self.start_point = (x, y)
            print(f"Highlight region start: ({x}, {y})")
    
    def on_motion(self, event):
        """마우스 움직임 이벤트"""
        self._update_mouse_position(event)
        if event.inaxes != self.ax or not self.drawing:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        self.end_point = (x, y)
        self.update_display()
    
    def _update_mouse_position(self, event):
        """현재 커서 위치 저장"""
        if event is None or self.ax is None:
            return
        if getattr(event, 'inaxes', None) != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.last_mouse_pos = (float(event.xdata), float(event.ydata))
        # 디버깅용 로그는 과도하지 않도록 생략

    def zoom_view(self, zoom_in: bool, center: Optional[Tuple[float, float]] = None):
        """키보드 기반 줌인/줌아웃"""
        if self.current_image is None or self.ax is None:
            print("[DEBUG] ZOOM VIEW: current_image or ax is None, cannot zoom")
            return
        
        img_height, img_width = self.current_image.shape[:2]
        img_aspect = img_width / img_height
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_span = abs(xlim[1] - xlim[0]) or img_width
        y_span = abs(ylim[1] - ylim[0]) or img_height
        
        if center is None:
            if self.last_mouse_pos:
                center_x, center_y = self.last_mouse_pos
                print(f"[DEBUG] ZOOM VIEW: using last mouse position {self.last_mouse_pos}")
            else:
                center_x = (xlim[0] + xlim[1]) / 2
                center_y = (ylim[0] + ylim[1]) / 2
                print(f"[DEBUG] ZOOM VIEW: using view center ({center_x:.1f}, {center_y:.1f})")
        else:
            center_x, center_y = center
        
        zoom = self.zoom_factor if zoom_in else 1.0 / self.zoom_factor
        
        current_aspect = x_span / y_span if y_span != 0 else img_aspect
        if current_aspect > img_aspect:
            new_y_range = y_span / zoom
            new_x_range = new_y_range * img_aspect
        else:
            new_x_range = x_span / zoom
            new_y_range = new_x_range / img_aspect
        
        max_x_range = img_width / self.min_zoom
        max_y_range = img_height / self.min_zoom
        min_x_range = img_width / self.max_zoom
        min_y_range = img_height / self.max_zoom
        
        if new_x_range > max_x_range:
            new_x_range = max_x_range
            new_y_range = new_x_range / img_aspect
        elif new_x_range < min_x_range:
            new_x_range = min_x_range
            new_y_range = new_x_range / img_aspect
        
        if new_y_range > max_y_range:
            new_y_range = max_y_range
            new_x_range = new_y_range * img_aspect
        elif new_y_range < min_y_range:
            new_y_range = min_y_range
            new_x_range = new_y_range * img_aspect
        
        half_x = new_x_range / 2
        half_y = new_y_range / 2
        
        center_x = min(max(center_x, half_x), img_width - half_x)
        center_y = min(max(center_y, half_y), img_height - half_y)
        
        new_x_min = max(0, center_x - half_x)
        new_x_max = min(img_width, center_x + half_x)
        new_y_min = max(0, center_y - half_y)
        new_y_max = min(img_height, center_y + half_y)
        
        if xlim[1] >= xlim[0]:
            self.ax.set_xlim(new_x_min, new_x_max)
        else:
            self.ax.set_xlim(new_x_max, new_x_min)
        
        if ylim[1] >= ylim[0]:
            self.ax.set_ylim(new_y_min, new_y_max)
        else:
            self.ax.set_ylim(new_y_max, new_y_min)
        
        self.current_xlim = [new_x_min, new_x_max]
        self.current_ylim = [new_y_min, new_y_max]
        
        print(f"[DEBUG] ZOOM VIEW: {'IN' if zoom_in else 'OUT'} -> xlim [{new_x_min:.1f}, {new_x_max:.1f}], ylim [{new_y_min:.1f}, {new_y_max:.1f}]")
        
        if self.fig and self.fig.canvas:
            self.fig.canvas.draw_idle()
    
    def reset_view(self):
        """전체 이미지 뷰로 리셋"""
        print(f"[DEBUG] RESET VIEW: Called")
        if self.current_image is None:
            print(f"[DEBUG] RESET VIEW: current_image is None, returning")
            return
        
        img_height, img_width = self.current_image.shape[:2]
        self.ax.set_xlim(0, img_width)
        self.ax.set_ylim(img_height, 0)
        self.current_xlim = None
        self.current_ylim = None
        print(f"[DEBUG] RESET VIEW: View reset to full image ({img_width}x{img_height})")
        self.fig.canvas.draw()
    
    def detect_bright_blob(self, search_region: Tuple[Tuple[int, int], Tuple[int, int]]) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray]]:
        """
        선택된 영역 근처에서 밝은 blob 자동 추출 (타원형/자유곡면 blob)
        
        Parameters
        ----------
        search_region : Tuple[Tuple[int, int], Tuple[int, int]]
            검색 영역 ((x1, y1), (x2, y2))
        
        Returns
        -------
        Optional[Tuple[List[Tuple[int, int]], np.ndarray]]
            (blob contour 좌표 리스트, blob mask) 또는 None
        """
        if self.current_image is None:
            return None
        
        start, end = search_region
        x_min = min(start[0], end[0])
        x_max = max(start[0], end[0])
        y_min = min(start[1], end[1])
        y_max = max(start[1], end[1])
        
        # 이미지 경계 체크
        img_height, img_width = self.current_image.shape[:2]
        x_min = max(0, x_min)
        x_max = min(img_width, x_max)
        y_min = max(0, y_min)
        y_max = min(img_height, y_max)
        
        # 사용자가 선택한 박스 영역만 사용 (확장하지 않음)
        # 이미지 crop
        if len(self.current_image.shape) == 3:
            search_area = self.current_image[y_min:y_max, x_min:x_max]
            # Grayscale로 변환
            if search_area.shape[2] == 3:
                search_area_gray = np.mean(search_area, axis=2).astype(np.uint8)
            else:
                search_area_gray = search_area[:, :, 0]
        else:
            search_area_gray = self.current_image[y_min:y_max, x_min:x_max]
        
        # Threshold 계산 (상위 75% 밝기)
        threshold = np.percentile(search_area_gray, 75) 
        
        # Binary mask 생성
        binary_mask = search_area_gray > threshold
        
        # Morphological operations으로 노이즈 제거
        binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3, 3)))
        binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((5, 5)))
        
        # Connected components로 blob 찾기
        labeled_array, num_features = label(binary_mask)
        
        if num_features == 0:
            return None
        
        # 각 blob의 크기와 중심 거리 계산
        blob_properties = []
        for i in range(1, num_features + 1):
            blob_mask = labeled_array == i
            blob_size = np.sum(blob_mask)
            
            # Blob의 중심 계산
            y_coords, x_coords = np.where(blob_mask)
            blob_center_x = np.mean(x_coords) + x_min
            blob_center_y = np.mean(y_coords) + y_min
            
            # 원래 선택 영역 중심까지의 거리
            original_center_x = (x_min + x_max) / 2
            original_center_y = (y_min + y_max) / 2
            distance = np.sqrt((blob_center_x - original_center_x)**2 + (blob_center_y - original_center_y)**2)
            
            blob_properties.append({
                'index': i,
                'size': blob_size,
                'distance': distance,
                'mask': blob_mask
            })
        
        # 가장 큰 blob 중에서 원래 선택 영역에 가장 가까운 것 선택
        if not blob_properties:
            return None
        
        # 크기와 거리의 가중 평균으로 최적 blob 선택
        max_size = max(prop['size'] for prop in blob_properties)
        max_distance = max(prop['distance'] for prop in blob_properties)
        if max_distance == 0:
            max_distance = 1
        
        best_blob = None
        best_score = -1
        
        for prop in blob_properties:
            # 크기 점수 (0-1) + 거리 점수 (0-1, 가까울수록 높음)
            size_score = prop['size'] / max_size if max_size > 0 else 0
            distance_score = 1 - (prop['distance'] / max_distance) if max_distance > 0 else 1
            score = size_score * 0.6 + distance_score * 0.4
            
            if score > best_score:
                best_score = score
                best_blob = prop
        
        if best_blob is None:
            return None
        
        # Blob의 실제 mask 추출
        blob_mask_local = best_blob['mask']
        
        # Blob의 contour 추출 (실제 형상) - skimage.measure.find_contours 사용
        y_coords, x_coords = np.where(blob_mask_local)
        
        if len(y_coords) == 0:
            return None
        
        try:
            # find_contours를 사용하여 blob의 실제 외곽선 추출
            # find_contours는 (row, col) 순서로 좌표를 반환하므로 주의
            contours = find_contours(blob_mask_local.astype(float), level=0.5)
            
            if len(contours) == 0:
                # Contour를 찾지 못한 경우 bounding box 사용
                blob_x_min = int(np.min(x_coords)) + x_min
                blob_x_max = int(np.max(x_coords)) + x_min
                blob_y_min = int(np.min(y_coords)) + y_min
                blob_y_max = int(np.max(y_coords)) + y_min
                contour = [
                    (blob_x_min, blob_y_min),
                    (blob_x_max, blob_y_min),
                    (blob_x_max, blob_y_max),
                    (blob_x_min, blob_y_max)
                ]
                full_mask = np.zeros((img_height, img_width), dtype=bool)
                full_mask[blob_y_min:blob_y_max+1, blob_x_min:blob_x_max+1] = True
                return (contour, full_mask)
            
            # 가장 큰 contour 선택 (blob의 외곽선)
            largest_contour = max(contours, key=len)
            
            # find_contours는 (row, col) = (y, x) 순서로 반환
            # 우리는 (x, y) 순서로 변환하고 전체 이미지 좌표계로 변환
            contour = [(int(col) + x_min, int(row) + y_min) 
                      for row, col in largest_contour]
            
            # 전체 이미지 좌표계로 변환된 mask
            full_mask = np.zeros((img_height, img_width), dtype=bool)
            local_y_coords, local_x_coords = np.where(blob_mask_local)
            full_y_coords = local_y_coords + y_min
            full_x_coords = local_x_coords + x_min
            # 경계 체크
            valid = (full_y_coords < img_height) & (full_x_coords < img_width) & (full_y_coords >= 0) & (full_x_coords >= 0)
            full_mask[full_y_coords[valid], full_x_coords[valid]] = True
            
            return (contour, full_mask)
            
        except Exception as e:
            # find_contours 실패 시 bounding box 사용
            print(f"Warning: find_contours failed, using bounding box: {e}")
            blob_x_min = int(np.min(x_coords)) + x_min
            blob_x_max = int(np.max(x_coords)) + x_min
            blob_y_min = int(np.min(y_coords)) + y_min
            blob_y_max = int(np.max(y_coords)) + y_min
            contour = [
                (blob_x_min, blob_y_min),
                (blob_x_max, blob_y_min),
                (blob_x_max, blob_y_max),
                (blob_x_min, blob_y_max)
            ]
            full_mask = np.zeros((img_height, img_width), dtype=bool)
            full_mask[blob_y_min:blob_y_max+1, blob_x_min:blob_x_max+1] = True
            return (contour, full_mask)
    
    def on_release(self, event):
        """마우스 릴리즈 이벤트"""
        self._update_mouse_position(event)
        if event.inaxes != self.ax or not self.drawing:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        self.drawing = False
        self.end_point = (x, y)
        
        if self.auto_mode:
            # Auto 모드: blob 자동 추출
            search_region = (self.start_point, self.end_point)
            detected_result = self.detect_bright_blob(search_region)
            
            if detected_result:
                contour, blob_mask = detected_result
                self.detected_blob = contour
                self.detected_blob_mask = blob_mask
                
                # Bounding box 계산 (호환성을 위해)
                x_coords = [p[0] for p in contour]
                y_coords = [p[1] for p in contour]
                self.temp_highlight = ((min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)))
                
                print(f"Blob detected with {len(contour)} contour points")
            else:
                print("No blob detected. Please try selecting a different region.")
                self.temp_highlight = None
                self.detected_blob = None
                self.detected_blob_mask = None
        else:
            # Manual 모드: 사용자가 선택한 영역 사용
            self.temp_highlight = (self.start_point, self.end_point)
            self.detected_blob = None
            print(f"Highlight region completed: {self.start_point} -> {self.end_point}")
        
        self.update_display()
    
    def update_display(self, preserve_zoom=True):
        """화면 업데이트"""
        if self.current_image is None:
            return
        
        print(f"[DEBUG] UPDATE DISPLAY: start (preserve_zoom={preserve_zoom})")
        print(f"[DEBUG] UPDATE DISPLAY: image path={self.current_image_path}, image id={id(self.current_image)}, shape={self.current_image.shape if hasattr(self.current_image, 'shape') else 'N/A'}")
        
        # 현재 뷰 범위 저장 (줌 상태 유지)
        if preserve_zoom:
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            # 전체 이미지 범위 체크 (초기화 상태인지 확인)
            img_width = self.current_image.shape[1]
            img_height = self.current_image.shape[0]
            if (abs(current_xlim[0] - 0) < 1 and abs(current_xlim[1] - img_width) < 1 and
                abs(current_ylim[0] - 0) < 1 and abs(current_ylim[1] - img_height) < 1):
                # 전체 뷰 상태이면 저장된 범위 사용 (없으면 전체 뷰)
                if self.current_xlim is not None:
                    current_xlim = self.current_xlim
                    current_ylim = self.current_ylim
            else:
                # 줌 상태가 있으면 저장
                self.current_xlim = current_xlim
                self.current_ylim = current_ylim
        else:
            # 초기화 시 전체 이미지로 설정
            current_xlim = None
            current_ylim = None
            
        # 이미지 표시
        self.ax.clear()
        if len(self.current_image.shape) == 3:
            self.ax.imshow(self.current_image, aspect='equal')
        else:
            self.ax.imshow(self.current_image, cmap='gray', aspect='equal')
        self.ax.set_title(f"Image Selection - {os.path.basename(self.current_image_path)}")
        print(f"[DEBUG] UPDATE DISPLAY: imshow finished for {self.current_image_path}")
        
        # 구슬 중심 그리기
        if self.temp_center:
            center_circle = Circle(self.temp_center, 5, color='green', fill=True)
            self.ax.add_patch(center_circle)
            
            # 구슬 직경 원 (픽셀 단위로 변환)
            sphere_radius_px = (self.sphere_diameter / self.pixel_resolution) / 2
            sphere_circle = Circle(self.temp_center, sphere_radius_px, color='green', fill=False, linewidth=2)
            self.ax.add_patch(sphere_circle)
        
        # 하이라이트 영역 그리기
        if self.temp_highlight:
            if self.auto_mode and self.detected_blob:
                # Auto 모드: 감지된 blob의 실제 형상(contour)을 파란색으로 표시
                blob_polygon = Polygon(self.detected_blob, fill=False, 
                                      linewidth=2, linestyle='-', edgecolor='blue', alpha=0.8)
                self.ax.add_patch(blob_polygon)
            else:
                # Manual 모드: 사용자가 선택한 영역을 빨간색 사각형으로 표시
                start, end = self.temp_highlight
                width = end[0] - start[0]
                height = end[1] - start[1]
                highlight_rect = Rectangle(start, width, height, color='red', fill=False, linewidth=2)
                self.ax.add_patch(highlight_rect)
        
        # 현재 그리기 중인 하이라이트 영역
        if self.drawing and self.start_point and self.end_point:
            width = self.end_point[0] - self.start_point[0]
            height = self.end_point[1] - self.start_point[1]
            if self.auto_mode:
                # Auto 모드: 검색 영역을 노란색으로 표시
                temp_rect = Rectangle(self.start_point, width, height, color='yellow', fill=False, linewidth=2, alpha=0.7, linestyle=':')
            else:
                # Manual 모드: 선택 영역을 빨간색으로 표시
                temp_rect = Rectangle(self.start_point, width, height, color='red', fill=False, linewidth=2, alpha=0.7)
            self.ax.add_patch(temp_rect)
        
        # 뷰 범위 복원 (줌 상태 유지)
        if current_xlim is not None and current_ylim is not None:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            print(f"[DEBUG] UPDATE DISPLAY: restored zoom -> xlim={current_xlim}, ylim={current_ylim}")
        else:
            # 초기 상태: 전체 이미지
            self.ax.set_xlim(0, self.current_image.shape[1])
            self.ax.set_ylim(self.current_image.shape[0], 0)
            self.current_xlim = None
            self.current_ylim = None
            print(f"[DEBUG] UPDATE DISPLAY: set to full image view")
        
        # Aspect ratio 설정 (이미지 비율 유지)
        self.ax.set_aspect('equal')
        
        self.fig.canvas.draw()
        print(f"[DEBUG] UPDATE DISPLAY: canvas.draw() completed")
    
    def select_sphere_and_highlight(self, image_path: str, image_index: int, auto_mode: bool = False) -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]:
        """이미지에서 구슬 중심과 하이라이트 영역 선택"""
        print(f"\n[DEBUG] ===== IMAGE SELECTION START: Image {image_index + 1} =====")
        print(f"[DEBUG] File: {os.path.basename(image_path)}")
        print(f"[DEBUG] Full path: {image_path}")
        
        self.auto_mode = auto_mode
        
        # 이미지 로드 (rectified 이미지가 있으면 사용, 없으면 원본 사용)
        try:
            # rectified 이미지가 있으면 사용
            if len(self.rectified_images) > image_index:
                self.current_image = self.rectified_images[image_index]
                print(f"[DEBUG] Using rectified image for index {image_index}")
            else:
                # 원본 이미지 로드
                pil_image = Image.open(image_path)
                self.current_image = np.array(pil_image)
                print(f"[DEBUG] Using original image for index {image_index}")
        except Exception as e:
            raise ValueError(f"Cannot load image: {image_path}, error: {e}")
        
        self.current_image_path = image_path
        # 기본 커서 위치를 이미지 중앙으로 초기화
        self.last_mouse_pos = (
            self.current_image.shape[1] / 2,
            self.current_image.shape[0] / 2,
        )
        print(f"[DEBUG] IMAGE LOADED: path={self.current_image_path}, image id={id(self.current_image)}, shape={self.current_image.shape if hasattr(self.current_image, 'shape') else 'N/A'}")
        self.temp_center = None
        self.temp_highlight = None
        self.detected_blob = None
        self.drawing = False
        
        # 줌 상태 초기화 (새 이미지)
        self.current_xlim = None
        self.current_ylim = None
        
        # matplotlib 윈도우 설정
        print(f"[DEBUG] WINDOW: Creating new figure and axes")
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        print(f"[DEBUG] WINDOW: Figure created - figure number: {self.fig.number if hasattr(self.fig, 'number') else 'N/A'}")
        
        # 윈도우 닫힘 이벤트 연결 (디버깅용)
        def on_close(event):
            print(f"[DEBUG] ===== WINDOW CLOSE EVENT TRIGGERED =====")
            print(f"[DEBUG] WINDOW CLOSE: event: {event}")
            print(f"[DEBUG] WINDOW CLOSE: canvas: {event.canvas if hasattr(event, 'canvas') else 'N/A'}")
            print(f"[DEBUG] WINDOW CLOSE: figure number: {self.fig.number if hasattr(self.fig, 'number') else 'N/A'}")
            print(f"[DEBUG] WINDOW CLOSE: Current image path: {getattr(self, 'current_image_path', 'None')}")
            self._debug_print_stack("[DEBUG] WINDOW CLOSE STACK TRACE", limit=8)
            print(f"[DEBUG] ===== END WINDOW CLOSE EVENT =====\n")
        
        self.fig.canvas.mpl_connect('close_event', on_close)
        print(f"[DEBUG] WINDOW: close_event handler connected")
        
        # 기본 내비게이션 도구 완전히 비활성화
        if hasattr(self.fig.canvas.manager, 'toolbar') and self.fig.canvas.manager.toolbar:
            toolbar = self.fig.canvas.manager.toolbar
            toolbar.set_message('')
            # 기본 내비게이션 모드 설정 (None = no navigation)
            try:
                if hasattr(toolbar, 'set_mode'):
                    toolbar.set_mode('')
                # NavigationToolbar2의 기본 동작 비활성화
                if hasattr(toolbar, '_nav_stack'):
                    toolbar._nav_stack.clear()  # 내비게이션 스택 클리어
                print(f"[DEBUG] WINDOW: Toolbar message set and mode cleared")
            except Exception as e:
                print(f"[DEBUG] WINDOW: Toolbar mode setting error: {e}")
        
        # 기본 키보드/마우스 내비게이션 비활성화 (axes 레벨)
        self.ax.set_navigate(False)  # axes 레벨에서 기본 내비게이션 비활성화
        
        # figure 레벨에서도 내비게이션 비활성화
        if hasattr(self.fig, 'canvas') and hasattr(self.fig.canvas, 'manager'):
            manager = self.fig.canvas.manager
            if hasattr(manager, 'toolbar') and manager.toolbar:
                # 기본 줌/팬 기능 비활성화
                try:
                    if hasattr(manager.toolbar, '_active'):
                        manager.toolbar._active = ''
                except Exception as e:
                    print(f"[DEBUG] WINDOW: Error disabling toolbar active mode: {e}")
        
        print(f"[DEBUG] WINDOW: Navigation disabled at all levels")
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        print(f"[DEBUG] CONNECTING EVENTS: All event handlers connected")
        
        # 이미지 표시 (초기 상태로)
        print(f"[DEBUG] UPDATE DISPLAY CALL: initial render for {self.current_image_path}")
        self.update_display(preserve_zoom=False)
        
        if auto_mode:
            print("Usage (Auto Mode):")
            print("1. Left click to select sphere center (green circle)")
            print("2. Left drag to select search region near highlight (yellow dashed rectangle)")
            print("3. System will automatically detect bright blob (blue dashed rectangle)")
            print("4. Press 'Enter' to complete selection")
            print("5. Press 'r' to reselect")
            print("6. Press 'q' to quit")
            print("7. '+' to zoom in, '-' to zoom out (cursor position if available)")
            print("8. Double click: Reset to full image view")
        else:
            print("Usage (Manual Mode):")
            print("1. Left click to select sphere center (green circle)")
            print("2. Left drag to select highlight region (red rectangle)")
            print("3. Press 'Enter' to complete selection")
            print("4. Press 'r' to reselect")
            print("5. Press 'q' to quit")
            print("6. '+' to zoom in, '-' to zoom out (cursor position if available)")
            print("7. Double click: Reset to full image view")
        print("")
        print("IMPORTANT: Select highlight region on the sphere surface, not at the center!")
        print("TIP: Use mouse wheel to zoom in for precise selection on large images!")
        
        # 키보드 이벤트 연결
        def on_key(event):
            print(f"[DEBUG] KEYBOARD EVENT - key: {getattr(event, 'key', 'None')}")
            # 키보드 이벤트만 처리 (스크롤 이벤트 등 다른 이벤트 무시)
            # key 속성이 없으면 키보드 이벤트가 아님 (가장 확실한 체크)
            if not hasattr(event, 'key') or event.key is None:
                print(f"[DEBUG] KEYBOARD EVENT IGNORED: no key attribute")
                return
            
            # button 속성이 있으면 스크롤 이벤트이므로 무시 (안전 장치)
            if hasattr(event, 'button') and event.button:
                print(f"[DEBUG] KEYBOARD EVENT IGNORED: has button attribute (likely scroll event)")
                return
            
            if event.key == 'enter':
                print(f"[DEBUG] KEYBOARD EVENT: ENTER pressed")
                if self.temp_center and self.temp_highlight:
                    print("[DEBUG] KEYBOARD EVENT: Selection completed! Closing window...")
                    print("Selection completed!")
                    plt.close()
                    print(f"[DEBUG] KEYBOARD EVENT: plt.close() called, returning from select_sphere_and_highlight")
                else:
                    print("Please select both sphere center and highlight region.")
            elif event.key == 'r':
                print(f"[DEBUG] KEYBOARD EVENT: 'r' pressed - Reselecting...")
                print("Reselecting...")
                self.temp_center = None
                self.temp_highlight = None
                self.detected_blob = None
                self.drawing = False
                self.update_display()
            elif event.key == 'q':
                print(f"[DEBUG] KEYBOARD EVENT: 'q' pressed - Exiting...")
                print("Exiting...")
                plt.close()
                print(f"[DEBUG] KEYBOARD EVENT: plt.close() called")
            elif event.key in ('+', '=', 'plus', 'add', 'kp+'):
                print(f"[DEBUG] KEYBOARD EVENT: '+' detected - Zoom in")
                self.zoom_view(zoom_in=True)
            elif event.key in ('-', '_', 'minus', 'subtract', 'kp-'):
                print(f"[DEBUG] KEYBOARD EVENT: '-' detected - Zoom out")
                self.zoom_view(zoom_in=False)
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # 윈도우가 닫힐 때까지 대기
        print(f"[DEBUG] ===== CALLING plt.show(block=True) - Waiting for user interaction =====")
        print(f"[DEBUG] PLT.SHOW: Blocking call started, figure will wait for close event")
        print(f"[DEBUG] PLT.SHOW: Current figure state - is_closed: {not plt.fignum_exists(self.fig.number) if hasattr(self.fig, 'number') else 'unknown'}")
        plt.show(block=True)
        
        # plt.show() 반환 직후 즉시 디버그
        print(f"[DEBUG] ===== PLT.SHOW RETURNED - Window closed =====")
        self._debug_print_stack("[DEBUG] PLT.SHOW RETURN STACK TRACE", limit=10)
        
        print(f"[DEBUG] PLT.SHOW: Figure number after return: {self.fig.number if hasattr(self.fig, 'number') else 'N/A'}")
        try:
            fig_exists_after = plt.fignum_exists(self.fig.number) if hasattr(self.fig, 'number') else True
            print(f"[DEBUG] PLT.SHOW: Figure exists after return: {fig_exists_after}")
        except Exception as e:
            print(f"[DEBUG] PLT.SHOW: Error checking figure after return: {e}")
        print(f"[DEBUG] ===== plt.show() RETURNED - Window closed =====")

        # u (→, East), v (↓, South)
        print(f"[DEBUG] RETURNING FROM select_sphere_and_highlight:")
        print(f"[DEBUG]   sphere_center: {self.temp_center}") #(u, v)
        print(f"[DEBUG]   highlight_region: {self.temp_highlight}") #(start_point(u, v), end_point(u, v))
        print(f"[DEBUG] ===== IMAGE SELECTION END: Image {image_index + 1} =====\n")
        
        return self.temp_center, self.temp_highlight
    
        """Light directions를 XYZ 좌표계로 변환"""
        if len(self.light_directions) == 0:
            raise ValueError("Light directions not set")
        
        self.light_directions = lvcalc.convert_image_coordinate_to_XYZ_coordinate(self.light_directions)

        """Light matrix 구축"""
        if len(self.light_directions) == 0:
            raise ValueError("Light directions not set")
        
        self.light_matrix = self.light_directions.T
        print(f"Light matrix shape: {self.light_matrix.shape}")

    def save_json_result(self, filename: str = "calibration_result.json"):
        """JSON 형식으로 결과 저장 (light_vec_calculator 사용)"""
        if self.light_matrix is None:
            raise ValueError("Light matrix not built")
        
        light_dir_array = np.array(self.light_directions)
        return lvcalc.save_calibration_json(light_dir_array, self.light_matrix, filename, 
                                           errors=self.errors, version="0.0.0-1")


def main_single_sphere():
    """메인 함수"""
    print("=== Simple Photometry Calibration (Matplotlib) ===")
    
    # 이미지 패턴 입력
    image_pattern = input("Enter image pattern (e.g., L2/*.bmp): ").strip()
    if not image_pattern:
        image_pattern = "L2/*.bmp"
    
    # 구슬 직경 입력
    try:
        sphere_diameter = float(input("Enter sphere diameter (mm): "))
    except ValueError:
        print("Invalid input. Using default 10.0mm.")
        sphere_diameter = 3.0
    
    # 픽셀 해상도 입력
    try:
        pixel_resolution = float(input("Enter pixel resolution FROM CAMERA CALIBRATION OR STEREO CALIBRATION (mm/px): "))
    except ValueError:
        print("Invalid input. Using default 0.1mm/px.")
        pixel_resolution = 0.01
    
    # Highlight 영역 선택 모드 선택
    mode_input = input("Highlight 영역을 직접 그리시겠습니까? (Yes or auto)) ").strip().lower()
    auto_mode = (mode_input == 'auto' or mode_input == 'a')
    
    if auto_mode:
        print("Auto mode: Highlight 영역 근처의 밝은 blob을 자동으로 추출합니다.")
    else:
        print("Manual mode: Highlight 영역을 직접 그립니다.")
    
    # 캘리브레이션 객체 생성
    calib = SimpleCalibrationMatplotlib()
    calib.sphere_diameter = sphere_diameter
    calib.pixel_resolution = pixel_resolution
    
    # Remap 디렉토리 입력 (선택적)
    remap_dir = input("Enter remap directory path (or press Enter to skip rectification): ").strip()
    apply_rectification = False
    if remap_dir:
        try:
            calib.load_remap_maps(remap_dir)
            apply_rectification = True
            print("Rectification will be applied to all images.")
        except Exception as e:
            print(f"Warning: Failed to load remap maps: {e}")
            print("Continuing without rectification...")
            apply_rectification = False
    
    # 이미지 로드 및 rectification 적용
    try:
        image_paths = calib.load_images(image_pattern, apply_rectification=apply_rectification)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # step 1: 각 이미지에서 구슬 중심과 하이라이트 영역 선택 (highlight 영역 수집만)
    for i, image_path in enumerate(image_paths):
        print(f"\n[DEBUG] ===== MAIN LOOP: Processing image {i+1}/{len(image_paths)} =====")
        print(f"[DEBUG] Image path: {image_path}")
        try:
            center, highlight = calib.select_sphere_and_highlight(image_path, i, auto_mode=auto_mode)
            print(f"[DEBUG] MAIN LOOP: select_sphere_and_highlight returned - center: {center}, highlight: {highlight}")
            if center and highlight:
                calib.sphere_centers.append(center)
                calib.highlight_regions.append(highlight)
                # Auto 모드에서 감지된 contour 저장
                if auto_mode and calib.detected_blob:
                    calib.highlight_contours.append(calib.detected_blob)
                else:
                    calib.highlight_contours.append(None)
                print(f"Image {i+1} selection completed: center={center}, highlight={highlight}")
            else:
                print(f"[DEBUG] MAIN LOOP: Image {i+1} selection incomplete (center or highlight is None), skipping...")
                print(f"Image {i+1} selection incomplete, skipping...")
                
        except Exception as e:
            print(f"[DEBUG] MAIN LOOP: Exception occurred for image {i+1}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Error processing image {i+1}: {e}")
            continue
        print(f"[DEBUG] ===== MAIN LOOP: Finished processing image {i+1} =====\n")

    # step 2: highlight_position을 (number of lights, 1, 2) 형태로 변환
    if len(calib.sphere_centers) == 0 or len(calib.highlight_regions) == 0:
        print("Error: No sphere centers or highlight regions collected.")
        return
    
    num_lights = len(calib.highlight_regions)
    sphere_radius_px = (sphere_diameter / pixel_resolution) / 2.0
    
    # highlight_position 리스트 구성: 각 이미지의 highlight 중심을 구 중심 기준 상대 좌표로 변환
    highlight_position_list = []
    for center, highlight in zip(calib.sphere_centers, calib.highlight_regions):
        # 하이라이트 영역의 중심 계산
        start_point, end_point = highlight
        highlight_center = ((start_point[0] + end_point[0]) / 2, 
                           (start_point[1] + end_point[1]) / 2)
        
        # 구슬 중심에서 하이라이트 중심까지의 벡터 (픽셀 단위)
        # 영상 좌표계: u (→, East), v (↓, South)
        u = highlight_center[0] - center[0]  # x (horizontal) 차이 = u (→, East)
        v = highlight_center[1] - center[1]  # y (vertical) 차이 = v (↓, South)
        
        highlight_position_list.append([(u, v)])  # (number of spheres=1, (u,v)) 형태
    
    # (number of lights, number of spheres=1, 2) 형태로 변환
    # compute_light_vector_from_highlight_position은 list를 받지만, numpy array도 처리 가능
    highlight_position = highlight_position_list  # list[list[tuple[float, float]]] 형태
    print(f"Number of lights: {len(highlight_position)}")
    print(f"Sphere radius (px): {sphere_radius_px}")
    
    # step 3: compute light vectors from highlight positions
    light_dir = lvcalc.compute_light_vector_from_highlight_position(highlight_position, sphere_radius_px)  # (num_lights, 1, uvw)
    print(f"Light direction shape (before averaging): {light_dir.shape}")
    
    # step 4: average light vectors (single sphere이므로)
    light_dir_avg = lvcalc.average_light_vector(light_dir)  # (num_lights, uvw)
    print(f"Light direction shape (after averaging): {light_dir_avg.shape}")
    
    # step 5: convert uvw coordinate to XYZ coordinate
    light_dir_XYZ = lvcalc.convert_image_coordinate_to_XYZ_coordinate(light_dir_avg)  # (num_lights, XYZ)
    light_matrix_XYZ = light_dir_XYZ.T  # (3, num_lights)
    
    # 결과 저장
    calib.light_directions = light_dir_XYZ
    calib.light_matrix = light_matrix_XYZ
    calib.errors = 'single_sphere'
    
    print(f"Light directions (XYZ): {calib.light_directions}")
    print(f"Light matrix shape: {calib.light_matrix.shape}")
    
    # 결과 저장
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_path = './output_calibration_results/' + timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ps_calib_name = save_path + '/ps_calib_L2SplitOnly_XYZ.json'
    debug_vector_name = save_path + '/debug_vector.png'
    debug_extraction_name = save_path + '/debug_extraction.png'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    calib.save_json_result(ps_calib_name)
    
    print("\n=== Calibration Completed ===")
    print(f"Selected sphere centers: {calib.sphere_centers}")
    print(f"Selected highlight regions: {calib.highlight_regions}")

    # Debugging: Save light vectors in multiple viewpoints
    debug_vis.save_light_vector_views(calib.light_directions, output_prefix=debug_vector_name)
    
    # Debugging: Save extraction debug images
    if len(calib.images) > 0 and len(calib.sphere_centers) > 0:
        sphere_radius_px = (calib.sphere_diameter / calib.pixel_resolution) / 2.0
        sphere_diameter_px = calib.sphere_diameter / calib.pixel_resolution
        
        # rectified 이미지가 있으면 사용, 없으면 원본 이미지 경로 사용
        if len(calib.rectified_images) > 0:
            print("Using rectified images for debug_extraction.png")
            debug_img.save_extraction_debug_images(
                calib.rectified_images, 
                calib.sphere_centers, 
                calib.highlight_regions,
                sphere_diameter_px,
                debug_extraction_name,
                highlight_contours=calib.highlight_contours
            )
        else:
            print("Using original images for debug_extraction.png")
            debug_img.save_extraction_debug_images(
                calib.images, 
                calib.sphere_centers, 
                calib.highlight_regions,
                sphere_diameter_px,
                debug_extraction_name,
                highlight_contours=calib.highlight_contours
            )
    
    # Debugging: Save rectified images
    if len(calib.rectified_images) > 0:
        rectified_dir = save_path + '/rectified_images'
        if not os.path.exists(rectified_dir):
            os.makedirs(rectified_dir)
        
        print(f"\nSaving {len(calib.rectified_images)} rectified images to {rectified_dir}...")
        for i, rectified_img in enumerate(calib.rectified_images):
            image_name = os.path.basename(calib.images[i])
            image_stem = os.path.splitext(image_name)[0]
            rectified_path = os.path.join(rectified_dir, f"rectified_{image_stem}.bmp")
            
            # 이미지 형식 변환 (cv2.imwrite를 위해)
            img_to_save = rectified_img.copy()
            
            # float 형식이면 uint8로 변환
            if img_to_save.dtype != np.uint8:
                if img_to_save.dtype == np.float32 or img_to_save.dtype == np.float64:
                    # 0-1 범위면 0-255로 스케일링, 아니면 클리핑
                    if img_to_save.max() <= 1.0:
                        img_to_save = (img_to_save * 255).astype(np.uint8)
                    else:
                        img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
                else:
                    # 다른 형식은 uint8로 변환 시도
                    img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
            
            # RGB -> BGR 변환 (3채널인 경우)
            if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(rectified_path, img_to_save)
            print(f"  Saved: {rectified_path}")
        print("Rectified images saved successfully.")


if __name__ == "__main__":
    main_single_sphere()
