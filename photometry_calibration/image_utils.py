"""
Shared image utilities for photometry calibration.
Remap map 로드, rectification 적용, 이미지 경로 수집 등 공용 함수.
"""

import glob
import numpy as np
import cv2
import tifffile as tiff
from pathlib import Path
from typing import List, Tuple

# remap map 제외 키워드
_REMAP_KEYWORDS = ('map_x', 'map_y', 'remap')


def load_map_pair(map_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """map_dir에서 *map_x.tiff, *map_y.tiff를 찾아 float32로 로드.

    Parameters
    ----------
    map_dir : str or Path
        remap map 파일이 있는 디렉토리 경로

    Returns
    -------
    (map_x, map_y) : Tuple[np.ndarray, np.ndarray]
        float32 remap map 배열
    """
    map_dir_path = Path(map_dir)
    mx_matches = sorted(map_dir_path.glob('*map_x.tiff'))
    my_matches = sorted(map_dir_path.glob('*map_y.tiff'))

    if not mx_matches:
        raise FileNotFoundError(f"Missing map_x file in {map_dir}: no file matching '*map_x.tiff'")
    if not my_matches:
        raise FileNotFoundError(f"Missing map_y file in {map_dir}: no file matching '*map_y.tiff'")

    map_x = tiff.imread(str(mx_matches[0])).astype(np.float32, copy=False)
    map_y = tiff.imread(str(my_matches[0])).astype(np.float32, copy=False)
    if map_x.shape != map_y.shape:
        raise ValueError(f"map_x and map_y shape mismatch: {map_x.shape} vs {map_y.shape}")

    print(f"Remap maps loaded: size={map_x.shape[::-1]} (W,H)")
    return map_x, map_y


def remap_image(
    src_img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float = 0.0,
) -> np.ndarray:
    """cv2.remap으로 이미지를 rectification. 채널 수/비트심도 보존."""
    return cv2.remap(src_img, map_x, map_y,
                     interpolation=interpolation,
                     borderMode=border_mode,
                     borderValue=border_value)


def apply_rectification(
    images: List[np.ndarray],
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> List[np.ndarray]:
    """이미지 리스트 전체에 rectification 적용."""
    return [remap_image(img, map_x, map_y) for img in images]


def collect_image_paths(
    image_pattern: str,
    extensions: Tuple[str, ...] = ('.bmp', '.png', '.tiff', '.tif', '.jpg', '.jpeg'),
    exclude_keywords: Tuple[str, ...] = _REMAP_KEYWORDS,
) -> List[str]:
    """이미지 경로를 디렉토리 또는 glob 패턴으로 수집.

    remap map 파일(*map_x*, *map_y*, *remap*) 은 자동으로 제외됩니다.

    Parameters
    ----------
    image_pattern : str
        디렉토리 경로 또는 glob 패턴 (예: 'L2', 'L2/*.bmp')
    extensions : tuple
        수집할 확장자 목록
    exclude_keywords : tuple
        파일명(stem)에 포함되면 제외할 키워드

    Returns
    -------
    List[str] : 정렬된 이미지 파일 경로 리스트
    """
    path_obj = Path(image_pattern)
    if path_obj.is_dir():
        image_paths = []
        for ext in extensions:
            for p in path_obj.glob('*' + ext):
                if not any(kw in p.stem.lower() for kw in exclude_keywords):
                    image_paths.append(p)
        image_paths = sorted([str(p) for p in image_paths])
    else:
        image_paths = sorted(glob.glob(image_pattern))

    return image_paths


def load_images_grayscale(image_pattern: str) -> Tuple[List[str], List[np.ndarray]]:
    """이미지를 grayscale로 로드.

    Returns
    -------
    (image_paths, images) : 경로 리스트와 numpy 배열(uint8 grayscale) 리스트
    """
    image_paths = collect_image_paths(image_pattern)
    if not image_paths:
        raise ValueError(f"이미지를 찾을 수 없습니다: {image_pattern}")

    images = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {p}")
        images.append(img)

    print(f"로드된 이미지 수: {len(images)}, 크기: {images[0].shape[::-1]} (W x H)")
    return image_paths, images
