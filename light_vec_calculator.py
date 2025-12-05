from typing import Any
import numpy as np
import json
import debug_light_vectors as debug_vis
import os
import datetime
# deprecated: use compute_light_vector_from_highlight_position instead
def compute_light_vector_from_angles(offset_px, radius_px, angle_deg):
    """    
    여러 각도의 하이라이트 오프셋으로부터 조명 벡터 행렬 생성.
    Orthographic camera 가정.

    ================================
    좌표계 규약 (Coordinate Convention)
    -------------------------------
    - 영상 좌표계:
        u : 오른쪽(→, East)
        v : 아래쪽(↓, South)        
        w : u (cross) v (right-hand rule) (카메라에서 물체로 향함 (out of image plane))

    - 각도 정의:
        angle_deg = 0   → 동쪽 (u+)
        angle_deg = 90  → 남쪽 (v+)
        angle_deg = 180 → 서쪽 (u−)
        angle_deg = 270 → 북쪽 (v-)

      회전 방향은 시계방향(clockwise)

    - 출력 벡터 정의:

        light_dir[i] = [Lu, Lv, Lw] (단위 벡터)
        light_matrix = light_dir.T   (3×N)
    ================================

    Parameters
    ----------
    offset_px : float
        구 중심으로부터 하이라이트까지의 거리 (px)
    radius_px : float
        구의 반경 (px)
    angle_deg : list[float]
        하이라이트 위치 각도 리스트 (deg)

    Returns
    -------
    light_dir : ndarray of shape (N,3)
        각 조명의 단위 조명 벡터
    light_matrix : ndarray of shape (3,N)
        photometric stereo 계산용 조명 행렬
    """

    light_dir = []

    # Viewing vector
    V = np.array([0.0, 0.0, -1.0]) # image to observer

    for angle in angle_deg:
        theta = np.deg2rad(angle)

        # v: 아래쪽(↓, South), u: 오른쪽(→, East)
        u = offset_px * np.cos(theta) 
        v = offset_px * np.sin(theta)  
        w = -1 * np.sqrt(radius_px**2 - offset_px**2) # sphere surface to camera

        # 표면 법선 (단위 벡터)
        N = np.array([u / radius_px, v / radius_px, w / radius_px])
        N /= np.linalg.norm(N)

        # 반사 법칙으로 조명 벡터 계산
        L = V - 2 * np.dot(N, V) * N # light -> sphere surface 
        L /= np.linalg.norm(L)

        light_dir.append(L)

    light_dir = np.array(light_dir)
    light_matrix = light_dir.T  # (3×N)

    return light_dir, light_matrix

# step 1
def compute_light_vector_from_highlight_position(highlight_position: list[list[tuple[float, float]]], radius_px: float):
    """
    여러 각도의 하이라이트 오프셋으로부터 조명 벡터 행렬 생성.
    Orthographic camera 가정.

    ================================
    좌표계 규약 (Coordinate Convention)
    -------------------------------
    - 영상 좌표계:
        u : 오른쪽(→, East)
        v : 아래쪽(↓, South)        
        w : u (cross) v (right-hand rule) (카메라에서 물체로 향함 (out of image plane))

      회전 방향은 시계방향(clockwise)

    ================================

    Input:
    ---------
    highlight_position: list[list[tuple[float, float]]]
        list of highlight positions for each sphere, shape (number of lights, number of spheres, (u,v))
    radius_px: float
        radius of the sphere in pixels

    Output:
    ---------
    light_dir: ndarray of shape (number of lights, number of spheres, uvw)
        light vector [Lu, Lv, Lw] (uvw coordinate)
        
    Parameters
    ----------
    offset_px : float
        구 중심으로부터 하이라이트까지의 거리 (px)
    radius_px : float
        구의 반경 (px)
    highlight_position : list[tuple[float, float]]
        하이라이트 위치 리스트 (u, v)


    """

    # 
    highlight_position_numLight_numSphere_uv = highlight_position.copy()
    
    # Viewing vector
    V = np.array([0.0, 0.0, 1.0]) # image to observer
    
    light_dir = []
    for highlight_position_numSphere_uv in highlight_position_numLight_numSphere_uv:
        light_dir_numSphere = []
        for uv in highlight_position_numSphere_uv:
            u, v = uv

            w = np.sqrt(radius_px**2 - u**2 - v**2)

            # 표면 법선 (단위 벡터)
            N = np.array([u / radius_px, v / radius_px, w / radius_px])
            N /= np.linalg.norm(N)

            # 반사 법칙으로 조명 벡터 계산
            L = V - 2 * np.dot(N, V) * N # light -> sphere surface 
            L /= np.linalg.norm(L)

            light_dir_numSphere.append(L)
        light_dir.append(np.array(light_dir_numSphere))
    light_dir = np.array(light_dir) # (number of lights, number of spheres, uvw)

    return light_dir
# step 2
# TODO : 좌표계 규약 최종 확인 예정
def convert_image_coordinate_to_XYZ_coordinate(light_dir):
    """
    image coordinate to XYZ coordinate conversion using rotation matrix
    
    Parameters
    ----------
    light_dir : ndarray of shape (N, 3)
        조명 벡터 [Lu, Lv, Lw] (image coordinate)
    
    Returns
    -------
    light_dir_XYZ : ndarray of shape (N, 3)
        조명 벡터 [LX, LY, LZ] (XYZ coordinate)
    
    좌표계 변환 (회전 행렬 사용):
    - 영상 좌표계:
        u : 오른쪽(→, East)
        v : 아래쪽(↓, South)        
        w : u (cross) v (카메라에서 물체로 향함 (out of image plane))
    - XYZ 좌표계 (ICI library convention):
        X : 위쪽(↑, North) ← v 매핑
        Y : 왼쪽(←, West) ← u 매핑
        Z : X (cross) Y (외적으로 결정)
    
    회전 행렬:
        [X]   [0  -1  0] [u]
        [Y] = [-1  0  0] [v]
        [Z]   [0   0 -1] [w]
    """
    if light_dir.ndim != 2 or light_dir.shape[-1] != 3:
        raise ValueError(f"light_dir must be shape (N, 3), got {light_dir.shape}")
    
    # 회전 행렬 정의
    # X ← v (North), Y ← u (West), Z는 외적으로 결정
    rotation_matrix = np.array([
        [0,  -1,  0],  # X = -v
        [-1,  0,  0],  # Y = -u
        [0,   0, -1]   # Z = -w (X × Y에 의해 결정됨)
    ])
    
    # 각 벡터를 회전 행렬로 변환
    light_dir_XYZ = []
    for vec in light_dir:
        vec_XYZ = rotation_matrix @ vec
        light_dir_XYZ.append(vec_XYZ)
    
    return np.array(light_dir_XYZ)

# step 3
def compute_error(light_dir_stack):
    '''
    Compares the light vectors between multiple spheres to verify whether the results are consistent.
    Parameters
    ----------
    light_dir_stack : list[ndarray of shape (N, 3)] # multiple spheres, shape (number of spheres, number of lights, 3)
        stacked light vectors
    Returns
    -------
    error : ndarray of shape (N, 3)
        error between light vectors
    '''
    # calculate RMSE error between light vectors and mean light vectors along the axis of number of spheres
    light_dir_stack = np.array(light_dir_stack)
    light_dir_stack_mean = np.mean(light_dir_stack, axis=1, keepdims=True)
    rmse_error = np.sqrt(np.mean((light_dir_stack - light_dir_stack_mean)**2, axis=1))

    return rmse_error # (N, 3), N= number of lights, 3= xyz dimensions

# step 4
def compute_good_bad(error):
    '''
    Compute good and bad lights based on the error.
    Parameters
    ----------
    error : ndarray of shape (N, 3)
        error between light vectors
    Returns : boolean (True if good, False if bad)
    '''
    if np.max(error) < 0.1:
        return True
    else:
        # print which light and dimension is the out of range
        print("!![Error] Get bad light vectors:")
        print(f"\tnumber of lights: 0 - {error.shape[0]-1}, number of dimensions(x,y,z): 0 -{error.shape[1]-1}")
        for i in range(error.shape[0]):
            for j in range(error.shape[1]):
                if error[i, j] > 0.1:
                    print(f"\t\tLight {i}, Dimension {j} is out of range, max err: {error[i, j]:.2f}")
                    
        return False

# step 5
def average_light_vector(light_dir_list):
    '''
    Average light vectors
    Parameters
    ----------
    light_dir_list : list[ndarray of shape (N, 3)]
        light vectors
    Returns
    '''
    return np.mean(light_dir_list, axis=1)

# step 6
def save_calibration_json(light_dir, light_matrix, output_filename="ps_calib.json", errors=None, version="0.0.0-1"):
    """
    캘리브레이션 결과를 ps_calib.json 형식으로 저장
    
    Parameters
    ----------
    light_dir : ndarray of shape (N,3)
        각 조명의 단위 조명 벡터
    light_matrix : ndarray of shape (3,N)
        photometric stereo 계산용 조명 행렬
    output_filename : str
        출력 파일명
    errors : ndarray of shape (N,3), str, or None
        각 조명에 대한 오차 
        - None: placeholder로 [0,0,0] 생성
        - 'ideal': 'ideal' 문자열로 저장 (ideal 조건)
        - ndarray/list: 실제 오차 값 저장
    version : str
        버전 문자열
    """
    # NumPy 배열을 리스트로 변환
    light_dir_list = light_dir.tolist()
    light_matrix_list = light_matrix.tolist()
    
    # errors 처리
    if errors is None:
        # placeholder: 각 조명마다 [0.0, 0.0, 0.0]
        num_lights = light_dir.shape[0]
        errors_value = [[0.0, 0.0, 0.0] for _ in range(num_lights)]
    elif isinstance(errors, str) and errors == 'ideal':
        # ideal 조건: 문자열로 저장
        errors_value = 'ideal'
    else:
        # 실제 오차 값 (ndarray 또는 list)
        if isinstance(errors, np.ndarray):
            errors_value = errors.tolist()
        else:
            errors_value = errors
    
    # JSON 구조 생성
    result = {
        "light_dir": light_dir_list,
        "light_matrix": light_matrix_list,
        "errors": errors_value,
        "version": version
    }
    
    # JSON 파일로 저장
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nCalibration result saved to: {output_filename}")
    return output_filename

# function for pseudo light vectors and matrices for L1-ring, L2-ring, L3-ring
def creadte_dummy_light_vector_and_matrix():
    return np.array([0.0, 0.0, 0.0]).reshape(1, 3), np.array([[0.0, 0.0, 0.0]]).T # (1, 3), (3, 1)

# function for pseudo light vectors and matrices including L1-ring, L2-ring, L3-ring
def stack_light_vector_and_matrix(light_dir_list, light_matrix_list):
    return np.concatenate(light_dir_list, axis=0), np.concatenate(light_matrix_list, axis=1)


# test function for single sphere, number of light = 4
def test_split_light_single_spheres():
    print(f"\n\n ================================ test_split_light_single_spheres ================================ \n\n")

    output_path = "output_single_sphere_ideal/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    calibration_filename_XYZ = output_path + "/ps_calib_L2SplitOnly_XYZ.json"
    debug_vector_filename = output_path + "/debug_vector.png"
    calibration_filename_L2Split_3LayerRing_XYZ = output_path + "/ps_calib_L2Split_3LayerRing_XYZ.json"


    # step 1: get highlight positions
    # manually set highlight positions for L2-split
    # They could be retrieved from the image, but for now, we will set them manually
    radius_px_L2 = 150 # radius of the sphere in pixels
    highlight_position_L2 = np.array([
                            [-44.5, -44.5],
                            [44.5, 44.5],
                            [-44.5, 44.5],
                            [44.5, -44.5]]).reshape(4, 1, 2) #(u, v)

    # step 2: compute light vectors from highlight positions
    light_dir_L2 = compute_light_vector_from_highlight_position(highlight_position_L2, radius_px_L2) # (# of light, # of spheres, uv)
    
    # step 3: calculate error btw multiple spheres, for single sphere skip this step
    errors = 'single_sphere'

    # step 4: average light vectors, for single sphere skip this step
    light_dir_L2_avg = average_light_vector(light_dir_L2) # (# of light, # of spheres, uvw) -> # (# of light, uv)
   
    # step 5: convert uvw coordinate to XYZ coordinate, based on ICI library convention
    light_dir_L2_XYZ = convert_image_coordinate_to_XYZ_coordinate(light_dir_L2_avg) # (N, uvw)
    light_matrix_L2_XYZ = light_dir_L2_XYZ.T # (3, N)

    # step 6: save calibration results
    save_calibration_json(light_dir_L2_XYZ, light_matrix_L2_XYZ,
        errors = errors, output_filename=calibration_filename_XYZ)
    # Debugging: Save light vectors in multiple viewpoints
    debug_vis.save_light_vector_views(light_dir_L2_XYZ, output_prefix=debug_vector_filename)
    

    # =============================================================================
    # Create dummy light vectors and matrices for L1-ring, L2-ring, L3-ring
    # So that they do not affect the result of the photometry calculation
    # =============================================================================
    light_dir_L1_ring, light_matrix_L1_ring = creadte_dummy_light_vector_and_matrix()
    light_dir_L2_ring, light_matrix_L2_ring = creadte_dummy_light_vector_and_matrix()
    light_dir_L3_ring, light_matrix_L3_ring = creadte_dummy_light_vector_and_matrix()

    # =============================================================================
    # Calibration results for L2-split + 3Layer Ring
    # =============================================================================
    
    # L2W1 - L1 - L2W2 - L2 - L2W3 - L3 - L2W4 - L1
    light_dir_list = [light_dir_L2_XYZ[[0]], light_dir_L1_ring, 
                        light_dir_L2_XYZ[[1]], light_dir_L2_ring, 
                        light_dir_L2_XYZ[[2]], light_dir_L3_ring,
                        light_dir_L2_XYZ[[3]], light_dir_L1_ring]
    
    # L2W1 - L1 - L2W2 - L2 - L2W3 - L3 - L2W4 - L1
    light_matrix_list = [light_matrix_L2_XYZ[:,[0]], light_matrix_L1_ring, 
                        light_matrix_L2_XYZ[:,[1]], light_matrix_L2_ring, 
                        light_matrix_L2_XYZ[:,[2]], light_matrix_L3_ring,
                        light_matrix_L2_XYZ[:,[3]], light_matrix_L1_ring]

    light_dir_seq_19_single_session, light_matrix_seq_19_single_session = stack_light_vector_and_matrix(light_dir_list, light_matrix_list)
    
    save_calibration_json(light_dir_seq_19_single_session, light_matrix_seq_19_single_session,
        errors = 'single_sphere', output_filename=calibration_filename_L2Split_3LayerRing_XYZ)     
    print(f"\n\n ================================ test_split_light_single_spheres_end ================================ \n\n")

# test function for multiple spheres, number of light = 4
def test_split_light_multiple_spheres():
    print(f"\n\n ================================ test_split_light_multiple_spheres ================================ \n\n")
    # =============================================================================
    # Create light vectors and matrices for ideal condition
    # =============================================================================
    output_path = "output_multi_sphere_ideal/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    calibration_filename_XYZ = output_path + "/ps_calib_L2SplitOnly_XYZ.json"
    debug_vector_filename = output_path + "/debug_vector.png"

    # step 1: get highlight positions
    # manually set highlight positions for L2-split
    # They could be retrieved from the image, but for now, we will set them manually
    radius_px_L2 = 150 # radius of the sphere in pixels
    pseudo_highlight_shift = 10
    highlight_position_L2_0 = np.array([[-44.5, -44.5],
                                         [44.5, 44.5],
                                         [-44.5, 44.5],
                                         [44.5, -44.5]]) # coordinates(u, v), shape (number of lights, 2)
    pos_h, pos_w = highlight_position_L2_0.shape[0], highlight_position_L2_0.shape[1]
    highlight_position_L2_1 = highlight_position_L2_0 + np.random.randn(pos_h, pos_w) * pseudo_highlight_shift
    highlight_position_L2_2 = highlight_position_L2_0 + np.random.randn(pos_h, pos_w) * pseudo_highlight_shift
    highlight_position_L2_3 = highlight_position_L2_0 + np.random.randn(pos_h, pos_w) * pseudo_highlight_shift
    highlight_position_L2_4 = highlight_position_L2_0 + np.random.randn(pos_h, pos_w) * pseudo_highlight_shift


    highlight_position_list = np.array([highlight_position_L2_0, highlight_position_L2_1, highlight_position_L2_2, highlight_position_L2_3, highlight_position_L2_4])
    highlight_position_list = np.transpose(highlight_position_list, (1, 0, 2)) # (number of lights, number of spheres, uv)

    # step 2: compute light vectors from highlight positions
    light_dir = compute_light_vector_from_highlight_position(highlight_position_list, radius_px_L2) # (number of lights, uvw)

    # step 3: calculate error btw multiple spheres, for multiple spheres skip this step
    error = compute_error(light_dir) # (number of lights, number of spheres, XYZ)
    good_bad = compute_good_bad(error)
    print(f"Good bad: {good_bad}, mean error: {np.mean(error):.2f}, max error: {np.max(error):.2f}")
    # assert good_bad, "Error is too large, calibration failed"

    # step 4: average light vectors, for multiple spheres skip this step
    light_dir_avg = average_light_vector(light_dir) # (N, 3)

    # step 5: convert uvw coordinate to XYZ coordinate, based on ICI library convention
    light_dir_XYZ = convert_image_coordinate_to_XYZ_coordinate(light_dir_avg) # (number of lights, XYZ)               
    light_matrix_XYZ = light_dir_XYZ.T # (3, N)    

    # step 6: save calibration results
    save_calibration_json(light_dir_XYZ, light_matrix_XYZ,
        errors = error, output_filename=calibration_filename_XYZ)

    # Debugging: Save light vectors in multiple viewpoints
    debug_vis.save_light_vector_views(light_dir_XYZ, output_prefix=debug_vector_filename)
    print(f"\n\n ================================ test_split_light_multiple_spheres_end ================================ \n\n")

if __name__ == "__main__":
    test_split_light_single_spheres()
    test_split_light_multiple_spheres()
    