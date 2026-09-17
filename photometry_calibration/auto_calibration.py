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
from . import camera_model


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


def _find_outer_radius_radial_legacy(
    avg_image: np.ndarray,
    cx: float, cy: float,
    sphere_radius_px: float,
    num_angles: int = 360,
) -> Tuple[np.ndarray, float]:
    """개선 전 edge 탐색. 비교용으로만 남겨둔다 (radius_estimator='legacy').

    최근접 샘플링 + 전진차분 + 탐색창 경계 피크 허용. 실측 기준 2.7~4.6 px
    낮게 나온다.
    """
    h, w = avg_image.shape[:2]
    img_float = avg_image.astype(np.float64)

    r_min = sphere_radius_px * 0.9
    r_max = sphere_radius_px * 1.1
    num_samples = max(int(r_max - r_min), 50)
    radii = np.linspace(r_min, r_max, num_samples)

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    edge_points = []

    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        profile = np.zeros(len(radii))
        for j, r in enumerate(radii):
            ix, iy = int(round(cx + r * cos_a)), int(round(cy + r * sin_a))
            if 0 <= ix < w and 0 <= iy < h:
                profile[j] = img_float[iy, ix]

        grad_abs = np.abs(np.diff(profile))
        if len(grad_abs) == 0:
            continue
        peak_r = radii[int(np.argmax(grad_abs))]
        edge_points.append((cx + peak_r * cos_a, cy + peak_r * sin_a))

    edge_points = np.array(edge_points)
    distances = np.hypot(edge_points[:, 0] - cx, edge_points[:, 1] - cy)
    return edge_points, float(np.median(distances))


def _find_outer_radius_radial(
    avg_image: np.ndarray,
    cx: float, cy: float,
    sphere_radius_px: float,
    num_angles: int = 360,
) -> Tuple[np.ndarray, float]:
    """Step 2: 중심에서 방사형 gradient profile로 외경 edge point를 탐색.

    반경은 r/R 을 통해 elevation 으로 2배 증폭되어 들어가고, 실루엣에서 역산한
    sphere 중심거리에도 쓰이므로 (합쳐서 약 0.6 deg/px) 서브픽셀 정확도가 필요하다.

    - 방사 프로파일을 이중선형 보간으로 샘플링 (최근접 샘플링은 격자에 스냅된다)
    - 중심차분으로 gradient 계산. 전진차분(np.diff)은 피크를 반 샘플 안쪽으로
      민다 (0.82 px 간격에서 -0.41 px).
    - gradient 피크를 포물선 보간으로 서브픽셀 정밀화
    """
    r_min = sphere_radius_px * 0.9
    r_max = sphere_radius_px * 1.1
    step = 0.25
    radii = np.arange(r_min, r_max, step, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    map_x = (cx + np.outer(radii, np.cos(angles))).astype(np.float32)
    map_y = (cy + np.outer(radii, np.sin(angles))).astype(np.float32)
    profile = cv2.remap(avg_image.astype(np.float32), map_x, map_y,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    grad = np.abs(np.gradient(profile, step, axis=0))
    k = np.argmax(grad, axis=0)

    # 포물선 서브픽셀 보간 (경계 샘플은 보정 없이 사용)
    ok = (k > 0) & (k < len(radii) - 1)
    idx = np.arange(len(angles))
    gm = grad[np.clip(k - 1, 0, None), idx]
    g0 = grad[k, idx]
    gp = grad[np.clip(k + 1, None, len(radii) - 1), idx]
    denom = gm - 2.0 * g0 + gp
    shift = np.where(np.abs(denom) > 1e-9, 0.5 * (gm - gp) / np.where(denom == 0, 1, denom), 0.0)
    shift = np.clip(np.where(ok, shift, 0.0), -1.0, 1.0)
    peak_r = radii[k] + shift * step

    # 탐색창 경계에 붙은 피크는 실제 limb 가 아니다. 그림자 쪽처럼 limb 대비가
    # 약한 방위각에서는 창 끝의 약한 gradient 가 이기며, 이것이 반경을 크게
    # 끌어내린다 (h=0 에서 360 중 82 개가 그랬다).
    valid = (peak_r > r_min + 1.0) & (peak_r < r_max - 1.0)
    if valid.sum() >= 16:
        angles, peak_r = angles[valid], peak_r[valid]

    edge_points = np.column_stack([cx + peak_r * np.cos(angles),
                                   cy + peak_r * np.sin(angles)])
    return edge_points, float(np.median(peak_r))


def _fit_circle_robust(edge_points: np.ndarray, cx: float, cy: float,
                       n_iter: int = 6, sigma: float = 2.5):
    """Edge point 에 원을 robust fit 하여 중심과 반경을 구한다.

    중심이 (dx, dy) 만큼 어긋나 있으면 반경이 r(theta) = R + dx*cos + dy*sin 로
    나타나므로, 이 형태를 이상치 제거와 함께 최소제곱으로 푼다.

    cv2.fitEllipse 는 대수적(conic) fit 이라 edge point 가 흩어지면 작은 쪽으로
    편향된다. 그림자에 오염된 방위각이 있으면 특히 심하다.
    """
    dx = edge_points[:, 0] - cx
    dy = edge_points[:, 1] - cy
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    C, S = np.cos(th), np.sin(th)

    # MAD 로 초기 이상치를 걸러낸 뒤 최소제곱에 들어간다. 최소제곱으로 시작하면
    # 이상치가 20% 만 있어도 표준편차가 부풀어 제거 문턱이 무력해진다.
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) * 1.4826
    keep = np.abs(r - med) < max(4.0 * mad, 1.5) if mad > 1e-9 else np.ones(r.size, bool)
    if keep.sum() < 16:
        keep = np.ones(r.size, bool)

    sol = np.array([med, 0.0, 0.0])
    res = r - med
    for _ in range(n_iter):
        A = np.column_stack([np.ones(keep.sum()), C[keep], S[keep]])
        sol, *_ = np.linalg.lstsq(A, r[keep], rcond=None)
        res = r - (sol[0] + sol[1] * C + sol[2] * S)
        s = np.std(res[keep])
        new = np.abs(res) < sigma * s if s > 1e-9 else keep
        if new.sum() < 16 or (new == keep).all():
            break
        keep = new
    return cx + sol[1], cy + sol[2], float(sol[0]), int(keep.sum()), float(np.std(res[keep]))


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
    radius_estimator: str = 'robust',
) -> Optional[List[dict]]:
    """하이브리드 방식으로 sphere의 외경을 검출.

    Step 1: SimpleBlobDetector로 검정 중심 위치 확보
    Step 2: 중심에서 방사형 gradient profile → 외경 edge point 탐색
    Step 3: edge point에 타원 fit → contour 생성

    radius_estimator:
        'robust' (기본) 서브픽셀 edge 탐색 + robust 원 fit
        'legacy'        개선 전 동작 (최근접 샘플링, 전진차분, fitEllipse)

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

        finder = (_find_outer_radius_radial_legacy if radius_estimator == 'legacy'
                  else _find_outer_radius_radial)
        edge_points, median_radius = finder(avg_image, cx, cy, sphere_radius_px)

        if len(edge_points) < 5:
            print(f"    [경고] edge point 부족 ({len(edge_points)}개), 건너뜀")
            continue

        ellipse, contour = _fit_ellipse_contour(edge_points)
        if radius_estimator == 'legacy':
            (ecx, ecy), (major, minor), _ = ellipse
            avg_radius = (major + minor) / 4.0
            print(f"    외경 타원(legacy): center=({ecx:.2f}, {ecy:.2f}), "
                  f"반경={avg_radius:.2f}px (기대: {sphere_radius_px:.2f}px)")
        else:
            ecx, ecy, avg_radius, n_kept, resid = _fit_circle_robust(edge_points, cx, cy)
            print(f"    외경 원: center=({ecx:.2f}, {ecy:.2f}), 반경={avg_radius:.2f}px "
                  f"(기대: {sphere_radius_px:.2f}px)")
            print(f"    edge point {n_kept}/{len(edge_points)} 사용, 잔차 {resid:.2f}px")

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
    highlight_method: str = 'ring',
    auto_radius: bool = True,
    exact_projection: bool = True,
    radius_estimator: str = 'robust',
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
        'ring' (기본) 또는 'centroid'
    radius_estimator : str
        'robust' (기본) 또는 'legacy'. legacy 는 개선 전 edge 탐색/타원 fit 으로,
        실측 기준 반경이 2.7~4.6 px 낮게 나온다. 비교용.
    auto_radius : bool
        True (기본) 면 검출된 sphere 실루엣 반경을 사용. False 면
        sphere_diameter / pixel_resolution 으로 계산한 nominal 반경을 사용.
        초점면이 sphere 적도면에 있지 않으면 겉보기 반경이 nominal 과 달라지며,
        그 차이가 반사 법칙의 2배 증폭을 거쳐 elevation 오차가 된다.
    exact_projection : bool
        True (기본) 면 선형 배율 가정(구면 전체 동일 배율, V=(0,0,1)) 대신
        정확한 원근 투영 모델을 사용. remap_dir 의 카메라 캘리브레이션에서
        초점거리와 주점을 읽어야 하므로, 없으면 자동으로 꺼진다.
    """
    print("=== Auto Photometry Calibration ===")
    print(f"Image path: {image_pattern}")
    print(f"Sphere diameter: {sphere_diameter} mm")
    print(f"Pixel resolution: {pixel_resolution} mm/px")
    print(f"Expected spheres: {num_spheres_expected}")
    print(f"Highlight method: {highlight_method}")
    print(f"Auto radius: {auto_radius}")
    print(f"Exact projection: {exact_projection}")
    print(f"Radius estimator: {radius_estimator}")

    # Remap map 로드
    map_x, map_y = image_utils.load_map_pair(remap_dir)

    camera = None
    if exact_projection:
        camera = camera_model.load_camera_model(remap_dir)
        if camera is None:
            print("[경고] remap_dir 에서 카메라 모델(CameraFocalLength/CameraCenterX/Y)을 "
                  "찾지 못했습니다. 선형 배율 가정으로 진행합니다.")
        else:
            print(f"Camera model: {camera}")

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
        radius_estimator=radius_estimator,
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
    nominal_radius_px = sphere_diameter_px / 2.0
    sphere_centers = [s['center'] for s in spheres]

    if auto_radius:
        sphere_radius_px = [s['radius'] for s in spheres]
        print(f"Sphere radius (px): auto-measured "
              f"{[f'{r:.2f}' for r in sphere_radius_px]} "
              f"(nominal {nominal_radius_px:.2f})")
    else:
        sphere_radius_px = nominal_radius_px
        print(f"Sphere radius (px): {sphere_radius_px} (nominal)")

    highlight_position_list, all_highlight_centers = lvcalc.build_highlight_position_list(
        highlight_method, all_centers, all_highlights, all_pixels,
        num_lights, num_spheres)

    print(f"Number of lights: {num_lights}, number of spheres: {num_spheres}")

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
        camera=camera,
        sphere_centers=sphere_centers if camera else None,
        sphere_radius_mm=sphere_diameter / 2.0 if camera else None,
    )

    return result
