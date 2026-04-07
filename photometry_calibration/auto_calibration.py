"""
Auto Photometry Calibration
입력 이미지를 평균하여 sphere를 자동 검출하는 캘리브레이션 파이프라인.
"""

import os
import datetime
import numpy as np
import cv2
from typing import List, Tuple, Optional

from . import image_utils
from . import light_vec_calculator as lvcalc


def compute_average_image(images: List[np.ndarray]) -> np.ndarray:
    """여러 조명 이미지를 평균하여 그림자를 줄인 이미지를 생성.

    서로 다른 방향의 조명 이미지를 평균하면 그림자가 상쇄되어
    sphere 추출이 용이해집니다.
    """
    stack = np.stack(images, axis=0).astype(np.float64)
    avg = np.mean(stack, axis=0)
    return np.clip(avg, 0, 255).astype(np.uint8)


def _detect_blob_centers(
    avg_image: np.ndarray,
    sphere_radius_px: float,
    sphere_diameter_px: float,
) -> list:
    """Step 1: SimpleBlobDetector로 검정 중심 위치를 검출."""
    inverted = cv2.bitwise_not(avg_image)

    params = cv2.SimpleBlobDetector_Params()

    expected_area = np.pi * sphere_radius_px ** 2
    params.filterByArea = True
    params.minArea = expected_area * 0.2
    params.maxArea = expected_area * 3.0

    params.filterByCircularity = True
    params.minCircularity = 0.4

    params.filterByConvexity = True
    params.minConvexity = 0.4

    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    params.filterByColor = True
    params.blobColor = 255

    params.minThreshold = 30
    params.maxThreshold = 220
    params.thresholdStep = 10

    params.minDistBetweenBlobs = sphere_diameter_px * 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)
    return inverted, keypoints


def _find_outer_radius_radial(
    avg_image: np.ndarray,
    cx: float, cy: float,
    sphere_radius_px: float,
    num_angles: int = 360,
) -> Tuple[np.ndarray, float]:
    """Step 2: 중심에서 방사형 gradient profile로 외경 edge point를 탐색."""
    h, w = avg_image.shape[:2]
    img_float = avg_image.astype(np.float64)

    r_min = sphere_radius_px * 0.9
    r_max = sphere_radius_px * 1.1
    num_samples = max(int(r_max - r_min), 50)
    radii = np.linspace(r_min, r_max, num_samples)

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    edge_points = []

    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        profile = np.zeros(len(radii))
        for j, r in enumerate(radii):
            x = cx + r * cos_a
            y = cy + r * sin_a
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                profile[j] = img_float[iy, ix]

        grad_abs = np.abs(np.diff(profile))
        if len(grad_abs) == 0:
            continue

        local_peak = np.argmax(grad_abs)
        peak_r = radii[local_peak]

        ex = cx + peak_r * cos_a
        ey = cy + peak_r * sin_a
        edge_points.append((ex, ey))

    edge_points = np.array(edge_points)
    distances = np.sqrt((edge_points[:, 0] - cx)**2 + (edge_points[:, 1] - cy)**2)
    median_radius = float(np.median(distances))

    return edge_points, median_radius


def _fit_ellipse_contour(edge_points: np.ndarray) -> Tuple[tuple, np.ndarray]:
    """Edge point 집합에 타원을 fit하고 contour를 생성."""
    pts_for_fit = edge_points.reshape(-1, 1, 2).astype(np.float32)
    ellipse = cv2.fitEllipse(pts_for_fit)

    contour = cv2.ellipse2Poly(
        center=(int(round(ellipse[0][0])), int(round(ellipse[0][1]))),
        axes=(int(round(ellipse[1][0] / 2)), int(round(ellipse[1][1] / 2))),
        angle=int(round(ellipse[2])),
        arcStart=0, arcEnd=360, delta=5,
    )
    contour = contour.reshape(-1, 1, 2).astype(np.int32)

    return ellipse, contour


def find_spheres(
    avg_image: np.ndarray,
    sphere_diameter_mm: float,
    pixel_resolution_mm_per_px: float,
    num_spheres_expected: int,
    debug_dir: str,
) -> Optional[List[dict]]:
    """하이브리드 방식으로 sphere의 외경을 검출.

    Step 1: SimpleBlobDetector로 검정 중심 위치 확보
    Step 2: 중심에서 방사형 gradient profile → 외경 edge point 탐색
    Step 3: edge point에 타원 fit → contour 생성

    Returns:
        검출된 sphere 정보 리스트 또는 개수 불일치 시 None
    """
    sphere_diameter_px = sphere_diameter_mm / pixel_resolution_mm_per_px
    sphere_radius_px = sphere_diameter_px / 2.0
    print(f"Sphere 직경: {sphere_diameter_px:.1f} px (반경: {sphere_radius_px:.1f} px)")

    _, keypoints = _detect_blob_centers(avg_image, sphere_radius_px, sphere_diameter_px)

    print(f"SimpleBlobDetector 검출 수: {len(keypoints)}")
    for i, kp in enumerate(keypoints):
        print(f"  blob [{i}] center=({kp.pt[0]:.1f}, {kp.pt[1]:.1f}), "
              f"blob_diameter={kp.size:.1f}px")

    if len(keypoints) != num_spheres_expected:
        blob_candidates = []
        for kp in keypoints:
            cx, cy = kp.pt
            r = kp.size / 2.0
            blob_candidates.append({
                'center': (cx, cy),
                'radius': r,
                'contour': cv2.ellipse2Poly(
                    (int(cx), int(cy)), (int(r), int(r)), 0, 0, 360, 5
                ).reshape(-1, 1, 2).astype(np.int32),
            })
        _save_debug_overlay_image(avg_image, blob_candidates, num_spheres_expected,
                          debug_dir, suffix='blob_step1')
        print(f"\n[실패] Step 1: 검출된 blob 수({len(keypoints)})가 "
              f"기대값({num_spheres_expected})과 다릅니다.")
        print("  - sphere diameter / pixel resolution 값 재확인")
        print("  - 이미지 품질(노출, 포커스) 확인")
        print(f"\nDebug 이미지가 '{debug_dir}'에 저장되었습니다.")
        return None

    candidates = []
    for i, kp in enumerate(keypoints):
        cx, cy = kp.pt
        print(f"\n  Sphere [{i}] 외경 탐색 (center=({cx:.1f}, {cy:.1f}))...")

        edge_points, median_radius = _find_outer_radius_radial(
            avg_image, cx, cy, sphere_radius_px
        )

        if len(edge_points) < 5:
            print(f"    [경고] edge point 부족 ({len(edge_points)}개), 건너뜀")
            continue

        ellipse, contour = _fit_ellipse_contour(edge_points)
        ecx, ecy = ellipse[0]
        major, minor = ellipse[1]
        angle = ellipse[2]
        avg_radius = (major + minor) / 4.0

        print(f"    외경 타원: center=({ecx:.1f}, {ecy:.1f}), "
              f"axes=({major:.1f}, {minor:.1f}), angle={angle:.1f}")
        print(f"    평균 반경: {avg_radius:.1f}px (기대: {sphere_radius_px:.1f}px)")

        candidates.append({
            'center': (ecx, ecy),
            'radius': avg_radius,
            'ellipse': ellipse,
            'contour': contour,
            'edge_points': edge_points,
        })

    _save_debug_overlay_image(avg_image, candidates, num_spheres_expected, debug_dir)

    if len(candidates) != num_spheres_expected:
        print(f"\n[실패] Step 2: 외경 fit 성공 수({len(candidates)})가 "
              f"기대값({num_spheres_expected})과 다릅니다.")
        print(f"\nDebug 이미지가 '{debug_dir}'에 저장되었습니다.")
        return None

    print(f"\n[성공] {num_spheres_expected}개 sphere 외경 검출 완료.")
    return candidates


def _save_debug_overlay_image(
    avg_image: np.ndarray,
    candidates: List[dict],
    num_expected: int,
    debug_dir: str,
    suffix: str = 'blob_detection',
):
    """평균 이미지 위에 검출된 타원 contour와 중심점을 표시하여 저장."""
    debug_img = cv2.cvtColor(avg_image, cv2.COLOR_GRAY2BGR)

    for i, c in enumerate(candidates):
        color = (0, 255, 0) if i < num_expected else (0, 255, 255)
        cx, cy = c['center']
        r = c['radius']

        if 'ellipse' in c:
            ellipse = c['ellipse']
            cv2.ellipse(debug_img, ellipse, color, 2)
        else:
            cv2.circle(debug_img, (int(cx), int(cy)), int(r), color, 2)

        cv2.circle(debug_img, (int(cx), int(cy)), 5, color, -1)
        cv2.putText(debug_img, f"#{i} r={r:.0f}", (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    info_text = f"Found: {len(candidates)} / Expected: {num_expected}"
    status = "OK" if len(candidates) == num_expected else "MISMATCH"
    info_color = (0, 255, 0) if status == "OK" else (0, 0, 255)
    cv2.putText(debug_img, f"{info_text} [{status}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)

    out_path = os.path.join(debug_dir, f'debug_{suffix}.png')
    cv2.imwrite(out_path, debug_img)
    print(f"Debug 이미지 저장: {out_path}")


def extract_highlight(
    image: np.ndarray,
    sphere_center: Tuple[float, float],
    sphere_radius_px: float,
) -> Optional[Tuple[Tuple[int, int], list]]:
    """Sphere 내부에서 가장 밝은 blob(highlight)을 추출.

    Returns:
        ((hcx, hcy), contour_points, highlight_pixels) 또는 None
    """
    h, w = image.shape[:2]
    cx, cy = sphere_center

    r = int(sphere_radius_px)
    x_min = max(0, int(cx) - r)
    x_max = min(w, int(cx) + r)
    y_min = max(0, int(cy) - r)
    y_max = min(h, int(cy) + r)

    crop = image[y_min:y_max, x_min:x_max].copy()
    crop_h, crop_w = crop.shape[:2]

    yy, xx = np.ogrid[:crop_h, :crop_w]
    dist = np.sqrt((xx - (cx - x_min))**2 + (yy - (cy - y_min))**2)
    sphere_mask = dist <= sphere_radius_px * 0.95

    masked = crop.copy()
    masked[~sphere_mask] = 0

    binary = ((masked == 255) & sphere_mask).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5:
        return None

    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None
    hcx = M['m10'] / M['m00'] + x_min
    hcy = M['m01'] / M['m00'] + y_min

    contour_points = [(int(pt[0][0]) + x_min, int(pt[0][1]) + y_min) for pt in largest]

    fill_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.drawContours(fill_mask, [largest], -1, 255, -1)
    hy, hx = np.where(fill_mask > 0)
    highlight_pixels = np.stack([hx + x_min, hy + y_min], axis=-1)

    return (hcx, hcy), contour_points, highlight_pixels


def run_auto_calibration(
    image_pattern: str,
    sphere_diameter: float,
    pixel_resolution: float,
    num_spheres_expected: int,
    remap_dir: str,
    save_base: str = './output_auto_calibration',
    highlight_method: str = 'centroid',
):
    """Auto calibration 파이프라인 실행.

    Parameters
    ----------
    image_pattern : str
        이미지 디렉토리 또는 glob 패턴
    sphere_diameter : float
        Sphere 직경 (mm)
    pixel_resolution : float
        픽셀 해상도 (mm/px)
    num_spheres_expected : int
        기대하는 sphere 개수
    remap_dir : str
        Remap map 디렉토리 경로
    save_base : str
        결과 저장 기본 디렉토리
    highlight_method : str
        'centroid' 또는 'ring'
    """
    print("=== Auto Photometry Calibration ===")
    print(f"Image path: {image_pattern}")
    print(f"Sphere diameter: {sphere_diameter} mm")
    print(f"Pixel resolution: {pixel_resolution} mm/px")
    print(f"Expected spheres: {num_spheres_expected}")
    print(f"Highlight method: {highlight_method}")

    # Remap map 로드
    map_x, map_y = image_utils.load_map_pair(remap_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_base, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")

    # Step 1: 이미지 로드
    image_paths, images = image_utils.load_images_grayscale(image_pattern)

    # Step 1.5: Rectification 적용
    print("\nApplying rectification to images...")
    images = image_utils.apply_rectification(images, map_x, map_y)
    print(f"Rectification applied to {len(images)} images.")

    # Step 2: 이미지 평균
    print("\n--- Step: 이미지 평균 ---")
    avg_image = compute_average_image(images)

    # Step 3: Sphere (blob) 검출
    print("\n--- Step: Sphere 검출 ---")
    spheres = find_spheres(
        avg_image,
        sphere_diameter_mm=sphere_diameter,
        pixel_resolution_mm_per_px=pixel_resolution,
        num_spheres_expected=num_spheres_expected,
        debug_dir=save_dir,
    )

    if spheres is None:
        print("\nCalibration 중단. Debug 이미지를 확인하고 파라미터를 조정하세요.")
        return None

    print("\n=== Sphere 검출 완료 ===")
    for i, s in enumerate(spheres):
        cx, cy = s['center']
        print(f"  Sphere {i+1}: center=({cx:.1f}, {cy:.1f}), radius={s['radius']:.1f}px")

    # Step 4: 각 이미지(조명) × 각 sphere에서 highlight 추출
    print("\n--- Step: Highlight 추출 ---")
    num_lights = len(images)
    num_spheres = len(spheres)
    sphere_diameter_px = sphere_diameter / pixel_resolution

    all_centers = []
    all_highlights = []
    all_contours = []
    all_pixels = []

    for light_idx, img in enumerate(images):
        for sphere_idx, sphere in enumerate(spheres):
            scx, scy = sphere['center']
            sr = sphere['radius']
            result = extract_highlight(img, (scx, scy), sr)

            if result is None:
                print(f"  [경고] L{light_idx+1} Sphere{sphere_idx+1}: highlight 미검출")
                all_centers.append((int(scx), int(scy)))
                all_highlights.append(((int(scx), int(scy)), (int(scx), int(scy))))
                all_contours.append(None)
                all_pixels.append(None)
                continue

            (hcx, hcy), contour_pts, highlight_pixels = result
            xs = [p[0] for p in contour_pts]
            ys = [p[1] for p in contour_pts]
            highlight_region = ((min(xs), min(ys)), (max(xs), max(ys)))

            all_centers.append((int(scx), int(scy)))
            all_highlights.append(highlight_region)
            all_contours.append(contour_pts)
            all_pixels.append(highlight_pixels)

            print(f"  L{light_idx+1} Sphere{sphere_idx+1}: "
                  f"highlight=({hcx:.1f}, {hcy:.1f})")

    # Step 5: highlight position → (u, v) 변환
    print(f"\n--- Step: Light vector 계산 (method={highlight_method}) ---")
    sphere_radius_px = sphere_diameter_px / 2.0

    highlight_position_list, all_highlight_centers = lvcalc.build_highlight_position_list(
        highlight_method, all_centers, all_highlights, all_pixels,
        num_lights, num_spheres)

    print(f"Number of lights: {num_lights}, number of spheres: {num_spheres}")
    print(f"Sphere radius (px): {sphere_radius_px}")

    # Step 6-11: KYCAL pipeline
    result = lvcalc.run_kycal_pipeline(
        highlight_position_list, sphere_radius_px, save_dir,
        images_for_debug=images,
        all_centers=all_centers,
        all_highlights=all_highlights,
        all_contours=all_contours,
        sphere_diameter_px=sphere_diameter_px,
        num_lights=num_lights,
        num_spheres=num_spheres,
        all_highlight_centers=all_highlight_centers,
    )

    return result
