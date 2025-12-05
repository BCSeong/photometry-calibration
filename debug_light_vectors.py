"""
Debug visualization script for light vectors
조명 벡터 시각화를 위한 디버그 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _setup_3d_axes(ax):
    """3D axes 기본 설정"""
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)


def _draw_vectors(ax, light_dir):
    """조명 벡터 그리기 - 각 점에 label 추가하고 (0,0,0)으로 향하는 벡터 그리기"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(light_dir)))
    
    # 각 조명 벡터 위치에 점 그리기
    for i, ray in enumerate(light_dir):
        color = colors[i % len(colors)]
        x, y, z = ray[0], ray[1], ray[2]
        
        # 구면 좌표계로 변환 (elevation과 azimuth 계산)
        r = np.sqrt(x**2 + y**2 + z**2)  # 거리
        if r > 1e-10:  # 0이 아닌 경우만
            # Azimuth: XY 평면에서의 각도 (0~360도, X축 기준)
            azimuth = np.arctan2(y, x)  # -π ~ π
            
            # Elevation: 수평면에서 수직으로 올라가는 각도 (0~90도)
            # z/r = sin(elevation), elevation = arcsin(z/r)
            elevation = np.arcsin(z / r)  # -π/2 ~ π/2
            
            # 1. 점 그리기
            ax.scatter(x, y, z, color=color, s=100)
            
            # 2. Label과 각도 정보를 텍스트로 표시 (Azimuth를 0~360도 범위로 변환)
            azimuth_deg = (np.degrees(azimuth) + 360) % 360
            elevation_deg = np.degrees(elevation)
            
            # 라벨과 각도 정보를 하나의 텍스트박스로 병합
            ax.text(x * 1.15, y * 1.15, z * 1.15, 
                   f'L{i+1}\nAz:{azimuth_deg:.1f}°\nEl:{elevation_deg:.1f}°',
                   fontsize=10, color=color, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor=color, linewidth=0.5))
            
            # 3. 각 점에서 (0,0,0)으로 향하는 벡터 그리기
            # 벡터 방향: (0,0,0) - ray = -ray
            ax.quiver(x, y, z, 
                     -x, -y, -z,
                     color=color, arrow_length_ratio=0.2, linewidth=2)
            
            # 4. Azimuth 각도 표시 (XY 평면에 투영한 원호)
            # XY 평면에서 원점에서 벡터의 XY projection까지의 선
            if np.sqrt(x**2 + y**2) > 1e-10:  # XY 평면에 투영이 있을 때만
                # 원점에서 XY projection까지의 선 (얇은 점선)
                ax.plot([0, x], [0, y], [0, 0], 
                       color=color, linestyle='--', linewidth=0.8, alpha=0.6)
                
                # Azimuth 각도 원호 그리기 (XY 평면에서)
                # 작은 반경의 원호로 각도 표시
                arc_radius = r * 0.3  # 벡터 길이의 30%
                num_points = 20
                arc_angles = np.linspace(0, azimuth, num_points)
                arc_x = arc_radius * np.cos(arc_angles)
                arc_y = arc_radius * np.sin(arc_angles)
                ax.plot(arc_x, arc_y, [0] * num_points,
                       color=color, linestyle='--', linewidth=0.8, alpha=0.5)
            
            # 5. Elevation 각도 표시 (수직 평면에서)
            # XY projection에서 점까지의 수직선과 원점에서 XY projection까지의 선 사이의 각도
            xy_proj_len = np.sqrt(x**2 + y**2)
            if xy_proj_len > 1e-10:
                # 수직 평면에서 elevation 각도 원호 그리기
                # 수직 평면은 XY projection 방향을 포함하는 평면
                arc_radius_vert = r * 0.3  # 벡터 길이의 30%
                num_points_vert = 20
                elev_angles = np.linspace(0, elevation, num_points_vert)
                
                # 수직 평면에서의 좌표 (azimuth 방향으로 회전)
                cos_az = x / xy_proj_len
                sin_az = y / xy_proj_len
                
                # elevation 원호의 점들 (수직 평면에서)
                arc_x_vert = arc_radius_vert * np.cos(elev_angles) * cos_az
                arc_y_vert = arc_radius_vert * np.cos(elev_angles) * sin_az
                arc_z_vert = arc_radius_vert * np.sin(elev_angles)
                
                ax.plot(arc_x_vert, arc_y_vert, arc_z_vert,
                       color=color, linestyle='--', linewidth=0.8, alpha=0.5)


def draw_light_vector(light_dir, view_azim=None, view_elev=None, title="Light Vectors"):
    """
    Draw light vectors and return figure object
    
    Parameters
    ----------
    light_dir : ndarray of shape (N, 3)
        조명 벡터 배열
    view_azim : float, optional
        Azimuth angle for view (degrees)
    view_elev : float, optional
        Elevation angle for view (degrees)
    title : str
        Plot title
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    _draw_vectors(ax, light_dir)
    
    if view_azim is not None and view_elev is not None:
        ax.view_init(elev=view_elev, azim=view_azim)
    
    _setup_3d_axes(ax)
    ax.set_title(title)
    
    return fig


def save_light_vector_views(light_dir, output_prefix="light_vectors"):
    """
    여러 시점의 조명 벡터를 PNG로 저장
    
    Parameters
    ----------
    light_dir : ndarray of shape (N, 3)
        조명 벡터 배열
    output_prefix : str
        PNG 파일명 접두사
    
    Returns
    -------
    saved_files : list
        저장된 파일명 리스트
    """
    viewpoints = [
        ('front', 0, 0),
        ('top', 0, 90),
        ('side', 90, 0),
        ('perspective', None, None)
    ]
    
    saved_files = []
    
    for view_name, azim, elev in viewpoints:
        if view_name == 'current':
            fig = draw_light_vector(light_dir, title=f'Light Vectors - {view_name.capitalize()} View')
            filename = f"{output_prefix}_current.png"
        else:
            fig = draw_light_vector(light_dir, view_azim=azim, view_elev=elev, 
                                   title=f'Light Vectors - {view_name.capitalize()} View')
            filename = f"{output_prefix}_{view_name}.png"
        
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        saved_files.append(filename)
        print(f"Saved: {filename}")
        plt.close(fig)
    
    return saved_files

