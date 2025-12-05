## 요구사항

- **Python**: 3.7 이상 (권장: 3.9)

## 개요
마우스로 구슬 중심과 하이라이트 영역을 선택하면 조명 방향을 자동으로 계산합니다.


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
- 입력폴더: `example_3mm_0.01px/*.bmp`
- 구슬 직경: `3` (mm)
- 픽셀 해상도: `0.01` (mm/px)

**사용 단계:**
1. 이미지 패턴 입력 (예: `example_3mm_0.01px/*.bmp`)
2. 구슬 직경 입력 (mm, 예: `3`)
3. 픽셀 해상도 입력 (mm/px, 예: `0.01`)
4. Highlight 영역 선택 모드 선택 (Yes/Auto)
5. 각 이미지에서:
   - 마우스 왼쪽 클릭으로 구슬 중심 선택 (녹색 원)
   - 마우스 왼쪽 드래그로 하이라이트 영역 선택 (빨간 사각형)
   - Enter 키로 선택 완료
   - `+/-` 키로 이미지 확대/축소
6. 자동으로 조명 방향 계산 및 JSON 결과 저장


## 출력 파일
- `calibration_result.json`: 캘리브레이션 결과 (JSON 형식)
- `light_positions.png`: 조명 위치 3D 시각화


## 개발 상태
- [x] 기본 프로젝트 구조
- [x] 파라미터 설정 클래스
- [x] 이미지 로더 (glob)
- [x] Light matrix 구축
- [x] JSON 출력
- [x] 3D 시각화
- [x] 사용 예시
- [ ] 이미지 입력 시 remap 하여 입력하는 파이프라인
- [ ] XYZ 좌표계 규약 확인
- [ ] 하이라이트 영역 자동 추출 (개발 X)
- [ ] 구슬 경계 자동 검출 (개발 X)
