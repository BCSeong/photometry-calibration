# Photometry Calibration

Sphere 기반 조명 방향 캘리브레이션 도구.
Orthographic camera 가정 하에 calibration sphere의 highlight 위치로부터 **조명에서 sphere 표면으로 향하는 단위 벡터**를 산출합니다.

## 요구사항

- **Python**: 3.9 이상
- 의존성: `pip install -r requirements.txt`

## 프로젝트 구조

```
photometry-calibration/
├── main.py                          # 진입점 (auto/manual 서브커맨드)
├── photometry_calibration/          # 패키지
│   ├── auto_calibration.py          # Auto 모드: sphere 자동 검출 + highlight 추출
│   ├── manual_calibration.py        # Manual 모드: matplotlib GUI로 수동 선택
│   ├── light_vec_calculator.py      # 핵심 연산 (light vector, 좌표 변환)
│   ├── image_utils.py               # 이미지 I/O, remap/rectification
│   ├── debug_image_extraction.py    # debug 시각화 (extraction)
│   └── debug_light_vectors.py       # debug 시각화 (3D vectors)
├── standalone_tools/                # 독립 유틸리티
│   ├── remap_apply.py               # 배치 remap 적용 도구
│   └── test_photometry_verification.py
├── requirements.txt
└── README.md
```

## 빠른 시작 (Auto 모드)

Auto 모드는 이미지 평균으로 sphere를 자동 검출하고, 각 조명 이미지에서 highlight를 자동 추출합니다.

### Interactive 모드 (권장)

```bash
python main.py auto --interactive
```

프롬프트에 따라 값을 입력합니다:
1. 이미지 디렉토리 (예: `example_4mm`)
2. Sphere 직경 (mm, 예: `4.0`)
3. 픽셀 해상도 (mm/px, 예: `0.01`)
4. Sphere 개수 (예: `7`)
5. Highlight method (`centroid` 또는 `ring`)
6. Remap 디렉토리 경로 (rectification map 위치)
7. 저장 디렉토리

### CLI 모드

```bash
python main.py auto \
    --image_pattern example_4mm \
    --sphere_diameter 4.0 \
    --pixel_resolution 0.01 \
    --num_spheres 7 \
    --remap_dir ./remap \
    --highlight_method centroid \
    --save_dir ./output_auto_calibration
```

### Manual 모드

matplotlib GUI로 sphere 중심과 highlight 영역을 수동 선택합니다.

```bash
python main.py manual --interactive
```

## 파라미터

| 파라미터 | 설명 | 예시 |
|----------|------|------|
| `image_pattern` | 이미지 디렉토리 또는 glob 패턴 | `example_4mm`, `L2/*.bmp` |
| `sphere_diameter` | Calibration sphere 직경 (mm) | `4.0` |
| `pixel_resolution` | 카메라 캘리브레이션에서 얻은 픽셀 해상도 (mm/px) | `0.01` |
| `num_spheres` | 이미지 내 sphere 개수 | `7` |
| `remap_dir` | Rectification remap map 디렉토리 (`*map_x.tiff`, `*map_y.tiff`) | `./remap` |
| `highlight_method` | Highlight 위치 계산 방법: `centroid` 또는 `ring` | `centroid` |

### Highlight Method

- **centroid** (기본값): highlight 픽셀의 무게중심(mean). 원형 highlight에 적합.
- **ring**: 극좌표 분해 — median(radial distance) + circular mean(angle). 호(arc) 형태 highlight에서 실제 radial distance를 보존.

## 캘리브레이션 파이프라인

```
이미지 로드 → Rectification → 이미지 평균 → Sphere 검출 → Highlight 추출
    → Highlight (u,v) offset 계산 → 반사 법칙으로 Light Vector 계산 (uvw)
    → uvw→XYZ 좌표 변환 → XYZ_backward 변환 → 구면 좌표 변환 → JSON 저장
```

### 핵심: Light Vector 계산 (반사 법칙)

Orthographic camera 가정에서, sphere 표면의 highlight는 반사 법칙에 의해 형성됩니다.

1. Highlight 위치 `(u, v)`에서 sphere 표면 법선 계산:
   ```
   N = (u/R, v/R, w/R),  w = sqrt(R^2 - u^2 - v^2)
   ```
2. Viewing vector `V = (0, 0, 1)` (카메라 방향)
3. 반사 법칙으로 조명 벡터 계산:
   ```
   L = V - 2(N·V)N
   ```

**산출물 `L`은 조명에서 sphere 표면으로 향하는 단위 벡터**입니다.

## 좌표계 규약

### 1. 영상 좌표계 (UVW) — 내부 연산용

```
        u (+)
        ───→  (East, image width 방향)
        |
    v (+) ↓   (South, image height 방향)

    w = u × v  (right-hand rule, 카메라에서 물체로 향함)
```

- `u`: 오른쪽 (→, East, image width 증가 방향)
- `v`: 아래쪽 (↓, South, image height 증가 방향)
- `w`: `u × v` (카메라에서 물체 방향, out of image plane)

### 2. XYZ 좌표계 — 출력 좌표계 (forward)

UVW에서 XYZ로의 변환 행렬:

```
[X]   [0   1   0] [u]
[Y] = [1   0   0] [v]
[Z]   [0   0  -1] [w]
```

- `X = v` : image height 방향 (아래쪽이 +)
- `Y = u` : image width 방향 (오른쪽이 +)
- `Z = -w` : `X × Y`로 결정 (물체에서 카메라 방향)

### 3. XYZ_backward 좌표계 — 역방향 출력

```
X_backward = -X
Y_backward = -Y
Z_backward =  Z  (유지)
```

XY 평면에서 방향을 반전합니다.

### 4. 구면 좌표계

XYZ 벡터를 elevation/azimuth로 변환:

- **elevation**: 수평면(XY)에서 Z축으로 올라가는 각도 (`arcsin(Z/r)`, -90~90도)
- **azimuth**: XY 평면에서의 각도 (`atan2(Y, X)`, 0~360도)

```
r = sqrt(X^2 + Y^2 + Z^2)
elevation = arcsin(Z / r)
azimuth   = atan2(Y, X)        (0~360도 변환)
```

## 출력 파일

결과는 `{save_dir}/YYYYMMDD_HHMMSS/` 디렉토리에 저장됩니다:

| 파일 | 설명 |
|------|------|
| `ps_calib_L2SplitOnly_XYZ.json` | 캘리브레이션 결과 (JSON) |
| `debug_vector.png_*.png` | 조명 벡터 3D 시각화 (front/top/side/perspective) |
| `debug_extraction.png` | Sphere/highlight 추출 결과 시각화 |
| `debug_blob_detection.png` | Sphere 검출 결과 (auto 모드) |

### JSON 구조

```json
{
  "forward": {
    "light_dir": [[X, Y, Z], ...],
    "light_dir_spherical_coord": [
      {"elevation_deg": 45.0, "azimuth_deg": 120.0}, ...
    ]
  },
  "backward": {
    "light_dir": [[-X, -Y, Z], ...],
    "light_dir_spherical_coord": [...]
  },
  "errors": [[ex, ey, ez], ...],
  "version": "0.0.0-1"
}
```

- `forward.light_dir`: **조명에서 sphere로 향하는 단위 벡터** (XYZ 좌표계, shape: num_lights x 3)
- `forward.light_dir_spherical_coord`: 동일 벡터의 구면 좌표 표현
- `backward`: XY 반전된 좌표계의 벡터
- `errors`: 복수 sphere 간 RMSE 오차 (0.1 미만이면 정상)
