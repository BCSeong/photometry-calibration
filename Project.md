# Photometry Calibration – Project

## 개요
구슬 하이라이트 위치로부터 조명 방향(light vector)을 계산하는 캘리브레이션 파이프라인(KYCAL) 문서 및 검토.

---

## KYCAL 파이프라인 (주석 기준)

| 단계 | 설명 |
|------|------|
| **0** | 이미지 로드 |
| **1** | 각 이미지에서 구슬 중심과 하이라이트 영역 선택 (highlight 영역 수집만) |
| **2** | highlight_position을 **(number of lights, number of spheres, 2)** 형태로 변환 |
| **3** | compute light vectors from highlight positions → 구슬 하나마다 분할조명 개수만큼의 light vector 계산 |
| **4** | average light vectors → 여러 구슬에 대해 (3)을 진행한 뒤 light vector 평균 및 오차 계산 (오차 0.1 이상이면 bad) |
| **5** | convert uvw coordinate to XYZ coordinate |
| **6** | convert XYZ to XYZ_backward |
| **7** | convert each light's XYZ and XYZ_backward to spherical coordinates |
| **8** | save json and debug results |

---

## Multiple sphere 파이프라인 검토

### 기대 데이터 형태 (light_vec_calculator 규약)

- **highlight_position**: `list[list[tuple[float, float]]]`  
  - 외부 리스트: 조명(light) 인덱스  
  - 내부 리스트: 해당 조명에서의 구(sphere)별 (u, v)  
  - 형태: **(num_lights, num_spheres, 2)**

- **compute_light_vector_from_highlight_position**  
  - 입력: `(num_lights, num_spheres, 2)`  
  - 출력: `(num_lights, num_spheres, 3)` (uvw)

- **compute_error**  
  - 입력: `(num_lights, num_spheres, 3)`  
  - axis=1(구 차원)으로 평균 후 RMSE  
  - 출력: `(num_lights, 3)`

- **average_light_vector**  
  - 입력: `(num_lights, num_spheres, 3)`  
  - axis=1로 평균  
  - 출력: `(num_lights, 3)`

### 기존 구현 (single sphere 가정)

- step 1: 이미지마다 **1개** (center, highlight)만 수집 → `sphere_centers`, `highlight_regions` 길이 = 이미지 개수.
- step 2:  
  - `highlight_position_list.append([(u, v)])`  
  - 한 조명당 리스트 1개, 그 안에 (u,v) 1개 → **(num_lights, 1, 2)** 만 지원.
- step 3~4: `(num_lights, 1, uvw)` 로 계산·평균·오차까지는 호출 규약에 맞음.
- step 5~8: `(num_lights, 3)` 기준으로 이후 변환·저장 정상.

### Multiple sphere 시 정상 동작을 위해 필요한 것

1. **step 2 형태**  
   - 주석: **(number of lights, number of spheres, 2)**.  
   - 현재는 **(number of lights, 1, 2)** 만 구성됨.  
   - **num_spheres > 1** 이면, 수집된 (center, highlight)를 **조명별로 num_spheres개씩 묶어서**  
     `(num_lights, num_spheres, 2)` 로 만들어야 함.

2. **이미지 순서 가정**  
   - 한 이미지 = 한 (조명, 구) 쌍이라고 가정하면:  
     이미지 순서 = [L1_S1, L1_S2, …, L1_Sn, L2_S1, …, L2_Sn, …]  
     즉 **총 이미지 수 = num_lights × num_spheres**.  
   - 수집된 `sphere_centers` / `highlight_regions`를 **num_spheres개 단위로 잘라**  
     조명 인덱스별로 묶으면 **(num_lights, num_spheres, 2)** 구성 가능.

3. **입력 추가**  
   - **num_spheres** (기본 1)를 입력받고,  
   - `len(sphere_centers) == num_lights * num_spheres` 인지 검사하면  
     multiple sphere 파이프라인이 주석/규약대로 동작할 수 있음.

### 검토 요약

| 항목 | 상태 | 비고 |
|------|------|------|
| step 0~1 | OK | 이미지 로드, (center, highlight) 수집 |
| step 2 형태 (single) | OK | (num_lights, 1, 2) |
| step 2 형태 (multiple) | 수정 필요 | num_spheres 입력 + (num_lights, num_spheres, 2) 리셰이프 |
| step 3~4 | OK | (num_lights, num_spheres, 3) → 평균/오차 연산 규약 일치 |
| step 5~8 | OK | (num_lights, 3) 기준 변환·저장 |

---

## 파라미터

- **sphere_diameter**: 구 직경 (mm)
- **pixel_resolution**: mm/px
- **num_spheres**: **이미지 1장당 구(sphere) 개수** (1 = single sphere, 2 이상 = multiple sphere).
- **num_lights** = 이미지 개수 (이미지 1장 = 조명 1개).

즉, **이미지 수 = num_lights**이고, 각 이미지에서 구를 **num_spheres**개 선택한다.  
선택 루틴은 single-sphere 로직을 **이미지당 num_spheres번** 반복한다 (같은 이미지를 num_spheres번 보여 주며, 매번 구 1개의 중심·하이라이트 선택).  
수집된 (center, highlight) 쌍은 **num_lights × num_spheres**개이며, 조명(이미지)별로 num_spheres개씩 묶어 (num_lights, num_spheres, 2)로 사용한다.
