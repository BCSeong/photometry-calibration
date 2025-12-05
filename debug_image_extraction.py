"""
Debug visualization script for image extraction
이미지 추출 시각화를 위한 디버그 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from PIL import Image
from typing import List, Tuple, Optional
import os


def save_extraction_debug_images(image_paths: List[str], 
                                 sphere_centers: List[Tuple[int, int]],
                                 highlight_regions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                 sphere_diameter_px: float,
                                 output_filename: str = "debug_extraction.png",
                                 highlight_contours: Optional[List[Optional[List[Tuple[int, int]]]]] = None):
    """
    이미지 추출 결과를 디버그용으로 시각화하여 저장
    
    Parameters
    ----------
    image_paths : List[str]
        이미지 파일 경로 리스트
    sphere_centers : List[Tuple[int, int]]
        각 이미지의 sphere center 좌표 리스트 (x, y)
    highlight_regions : List[Tuple[Tuple[int, int], Tuple[int, int]]]
        각 이미지의 highlight 영역 리스트 ((x1, y1), (x2, y2))
    sphere_diameter_px : float
        Sphere diameter in pixels
    output_filename : str
        출력 PNG 파일명
    highlight_contours : Optional[List[Optional[List[Tuple[int, int]]]]]
        각 이미지의 highlight contour 좌표 리스트 (Auto 모드에서 감지된 blob contour)
    """
    if len(image_paths) != len(sphere_centers) or len(image_paths) != len(highlight_regions):
        raise ValueError("image_paths, sphere_centers, highlight_regions must have the same length")
    
    if highlight_contours is None:
        highlight_contours = [None] * len(image_paths)
    
    crop_size = int(sphere_diameter_px * 1.5)
    half_crop = crop_size // 2
    
    cropped_images = []
    
    for img_path, center, highlight, contour in zip(image_paths, sphere_centers, highlight_regions, highlight_contours):
        # 이미지 로드
        try:
            pil_image = Image.open(img_path)
            img = np.array(pil_image)
        except Exception as e:
            raise ValueError(f"Cannot load image: {img_path}, error: {e}")
        
        # 좌표 추출 (x, y)
        center_x, center_y = center
        highlight_start, highlight_end = highlight
        
        # Crop 영역 계산 (이미지 경계 체크)
        img_height, img_width = img.shape[:2] if len(img.shape) == 2 else img.shape[:2]
        
        x_min = max(0, center_x - half_crop)
        x_max = min(img_width, center_x + half_crop)
        y_min = max(0, center_y - half_crop)
        y_max = min(img_height, center_y + half_crop)
        
        # Crop
        if len(img.shape) == 2:  # Grayscale
            cropped = img[y_min:y_max, x_min:x_max]
        else:  # Color
            cropped = img[y_min:y_max, x_min:x_max, :]
        
        # Crop된 영역 내에서의 상대 좌표 계산
        crop_center_x = center_x - x_min
        crop_center_y = center_y - y_min
        
        # Highlight region을 crop 영역 기준으로 변환
        highlight_start_rel = (highlight_start[0] - x_min, highlight_start[1] - y_min)
        highlight_end_rel = (highlight_end[0] - x_min, highlight_end[1] - y_min)
        
        # Highlight center of mass 계산
        highlight_center_x = (highlight_start[0] + highlight_end[0]) / 2
        highlight_center_y = (highlight_start[1] + highlight_end[1]) / 2
        highlight_center_rel = (highlight_center_x - x_min, highlight_center_y - y_min)
        
        # matplotlib figure로 변환하여 그림 그리기
        fig, ax = plt.subplots(figsize=(crop_size/50, crop_size/50), dpi=50)
        
        if len(cropped.shape) == 2:  # Grayscale
            ax.imshow(cropped, cmap='gray')
        else:  # Color
            ax.imshow(cropped)
        
        # Sphere center dot
        ax.plot(crop_center_x, crop_center_y, 'go', markersize=10, label='Sphere Center')
        
        # Sphere diameter circle
        circle = Circle((crop_center_x, crop_center_y), sphere_diameter_px / 2, 
                       fill=False, color='green', linewidth=2)
        ax.add_patch(circle)
        
        # Highlight region 표시
        if contour is not None:
            # Auto 모드: 감지된 blob의 실제 contour (타원형/자유곡면) 표시
            # Contour를 crop 영역 기준 좌표로 변환
            contour_rel = [(x - x_min, y - y_min) for x, y in contour]
            blob_polygon = Polygon(contour_rel, fill=False, 
                                  linewidth=2, linestyle='-', edgecolor='blue', alpha=0.8, 
                                  label='Detected Blob')
            ax.add_patch(blob_polygon)
        else:
            # Manual 모드: 선택한 영역을 사각형으로 표시
            rect_width = highlight_end_rel[0] - highlight_start_rel[0]
            rect_height = highlight_end_rel[1] - highlight_start_rel[1]
            rect = Rectangle(highlight_start_rel, rect_width, rect_height,
                            fill=False, color='red', linewidth=2, label='Highlight Region')
            ax.add_patch(rect)
        
        # Highlight center of mass
        ax.plot(highlight_center_rel[0], highlight_center_rel[1], 'r+', 
               markersize=15, markeredgewidth=3, label='Highlight Center')
        
        ax.set_xlim(0, cropped.shape[1])
        ax.set_ylim(cropped.shape[0], 0)
        ax.axis('off')  # 축 제거
        
        # Figure를 numpy array로 변환
        fig.canvas.draw()
        # Renderer에서 buffer 가져오기
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        buf = buf.reshape((h, w, 3))
        cropped_images.append(buf)
        
        plt.close(fig)
    
    # 모든 이미지를 가로로 붙이기
    # 높이는 가장 높은 이미지에 맞추고, 작은 이미지는 중앙 정렬
    max_height = max(img.shape[0] for img in cropped_images)
    total_width = sum(img.shape[1] for img in cropped_images)
    
    combined_image = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
    
    x_offset = 0
    for img in cropped_images:
        h, w = img.shape[:2]
        y_offset = (max_height - h) // 2
        combined_image[y_offset:y_offset+h, x_offset:x_offset+w] = img
        x_offset += w
    
    # 최종 이미지 저장
    result_img = Image.fromarray(combined_image)
    result_img.save(output_filename, dpi=(150, 150))
    print(f"Saved debug extraction image: {output_filename}")
    
    return output_filename

