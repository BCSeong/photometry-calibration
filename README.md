## 요구사항

- **Python**: 3.7 이상 (권장: 3.9)

## 개요
마우스로 구슬 중심과 하이라이트 영역을 선택하면 조명 방향을 자동으로 계산합니다.

## 캘리브레이션 파이프라인 (KYCAL Pipeline)

캘리브레이션은 다음 단계로 진행됩니다:

0. **이미지 로드**: 이미지 파일들을 로드하고 선택적으로 rectification 적용
1. **구슬 중심 및 하이라이트 영역 선택**: 각 이미지에서 구슬 중심과 하이라이트 영역을 수동 또는 자동으로 선택
2. **하이라이트 위치 변환**: 선택된 하이라이트 영역을 (number of lights, number of spheres, 2) 형태로 변환
3. **조명 벡터 계산**: 하이라이트 위치로부터 각 구슬에 대한 조명 벡터 계산 (uvw 좌표계)
4. **조명 벡터 평균 및 오차 계산**: 여러 구슬에 대해 조명 벡터의 평균과 오차 계산 (오차가 0.1 이상이면 bad 출력)
5. **UVW → XYZ 좌표 변환**: 영상 좌표계(uvw)를 XYZ 좌표계로 변환
6. **XYZ → XYZ_backward 변환**: XYZ 좌표를 XYZ_backward 좌표로 변환 (XY에 -1 곱셈)
7. **구면 좌표 변환**: 각 조명의 XYZ 및 XYZ_backward를 구면 좌표(elevation, azimuth)로 변환
8. **결과 저장**: JSON 파일 및 디버그 이미지 저장

## 파라미터
- `sphere_diameter`: 구슬의 직경 (mm)
- `pixel_resolution`: 이미지의 pixel resolution (mm/px)


### 설치
```bash
pip install -r requirements.txt
```

### 테스트

#### 1. light_vec_calculator.py 테스트
```bash
python light_vec_calculator.py
```
main() 실행하여 single sphere 와 multi sphere 에 대한 예제를 확인하세요

#### 2. simple_calibration_matplotlib.py 테스트
```bash
python simple_calibration_matplotlib.py
```
실제 칼리브레이션 파이프라인.

**테스트 입력값:**
- 입력폴더: `example_4mm/*.bmp`
- 구슬 직경: `4` (mm)
- 픽셀 해상도: `0.01` (mm/px)
- High

**사용 단계:**
1. 이미지 패턴 입력 (예: `example_4mm/*.bmp`)
2. 구슬 직경 입력 (mm, 예: `4`)
3. 픽셀 해상도 입력 (mm/px, 예: `0.01`)
4. Rectification 적용 여부 선택 (remap 디렉토리 경로 입력 또는 Enter로 건너뛰기)
5. Highlight 영역 선택 모드 선택 (Manual/Auto)
6. 각 이미지에서:
   - **Manual 모드**: 
     - 마우스 왼쪽 클릭으로 구슬 중심 선택 (녹색 원)
     - 마우스 왼쪽 드래그로 하이라이트 영역 선택 (빨간 사각형)
   - **Auto 모드**:
     - 마우스 왼쪽 클릭으로 구슬 중심 선택 (녹색 원)
     - 마우스 왼쪽 드래그로 하이라이트 검색 영역 선택 (노란 점선 사각형)
     - 시스템이 자동으로 밝은 blob 감지 (파란 다각형)
   - Enter 키로 선택 완료
   - `r` 키로 재선택
   - `+/-` 키로 이미지 확대/축소 (커서 위치 기준)
   - 더블 클릭으로 전체 이미지 뷰로 리셋
7. 자동으로 조명 방향 계산 및 결과 저장


## 출력 파일

캘리브레이션 결과는 `output_calibration_results/YYYYMMDD_HHMMSS/` 디렉토리에 저장됩니다:

- `ps_calib_L2SplitOnly_XYZ.json`: 캘리브레이션 결과 (JSON 형식)
  - `light_dir`: 조명 벡터 배열 (XYZ 좌표계)
  - `light_matrix`: 조명 행렬 (photometric stereo 계산용)
  - `errors`: 각 조명에 대한 오차
  - `light_dir_spherical_coord`: 각 조명의 구면 좌표 (elevation_deg, azimuth_deg_180, azimuth_deg_360)
  - `backward`: XYZ_backward 좌표계의 조명 벡터 및 구면 좌표
  - `version`: 버전 정보
- `debug_vector.png_*.png`: 조명 벡터 3D 시각화 (여러 시점)
  - `_front.png`: 정면 뷰
  - `_top.png`: 상단 뷰
  - `_side.png`: 측면 뷰
  - `_perspective.png`: 원근 뷰
- `debug_extraction.png`: 구슬 중심 및 하이라이트 영역 추출 결과 시각화
- `rectified_images/`: Rectification이 적용된 경우, rectified 이미지들 저장


## 좌표계 규약

### 영상 좌표계 (UVW)
- `u`: 오른쪽(→, East)
- `v`: 아래쪽(↓, South)
- `w`: u × v (카메라에서 물체로 향함, right-hand rule)

### XYZ 좌표계 (ICI library convention)
- `X`: 아래쪽(↓, South) ← v 매핑
- `Y`: 오른쪽(→, East) ← u 매핑
- `Z`: X × Y (외적으로 결정)

### XYZ_backward 좌표계
- `X_backward`: -X
- `Y_backward`: -Y
- `Z_backward`: Z (유지)

### 구면 좌표계
- `elevation_deg`: 수평면에서 수직으로 올라가는 각도 (-90° ~ 90°)
- `azimuth_deg_180`: XY 평면에서의 각도 (-180° ~ 180°, X축 기준)
- `azimuth_deg_360`: XY 평면에서의 각도 (0° ~ 360°, X축 기준, 기본 사용)

## 개발 상태
- [x] 기본 프로젝트 구조
- [x] 파라미터 설정 클래스
- [x] 이미지 로더 (glob)
- [x] Rectification 지원 (remap maps)
- [x] Light matrix 구축
- [x] UVW → XYZ 좌표 변환
- [x] XYZ → XYZ_backward 좌표 변환
- [x] 구면 좌표 변환
- [x] JSON 출력 (LightCalibrationResult 형식)
- [x] 3D 시각화 (여러 시점)
- [x] 하이라이트 영역 자동 추출 (Auto 모드)
- [x] 오차 계산 및 검증
- [x] 사용 예시
- [ ] 구슬 경계 자동 검출 (개발 예정 없음)
