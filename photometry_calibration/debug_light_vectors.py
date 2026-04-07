"""
Debug visualization script for light vectors
조명 벡터 시각화를 위한 디버그 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _setup_3d_axes(ax, flip_z_for_view=True):
    """3D axes 기본 설정. X=image height, Y=image width.
    flip_z_for_view=True 이면 Z축을 반전해 +Z = 시선(카메라) 방향으로 표시 (조명이 카메라 쪽이면 위쪽에 보임)."""
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
    
    ax.set_xlabel('X (image height)')
    ax.set_ylabel('Y (image width)')
    ax.set_zlabel('Z (view)' if flip_z_for_view else 'Z')
    ax.grid(True)


def _draw_vectors(ax, light_dir, light_dir_deg=None, flip_z_for_view=True):
    """조명 벡터 그리기: light_dir = light→surface 단위벡터. 원점=surface, 점=조명 위치(-light_dir), 화살표=light→surface.
    이미지 위쪽 highlight ↔ L은 X 음수 쪽. flip_z_for_view=True 이면 Z만 반전해 +Z=시선(카메라) 방향으로 그림.
    
    Photometry: 시선(카메라)=+Z, 표면=원점. 조명이 카메라 쪽(+Z)이면 light→surface는 -Z 방향이어야 함.
    현재 파이프라인은 Z=-w(uvw)라 저장값은 조명이 카메라 쪽일 때 Z>0으로 나올 수 있음 → 디버그에서 Z 반전으로 보정.
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(light_dir)))
    z_scale = -1.0 if flip_z_for_view else 1.0

    for i, ray in enumerate(light_dir):
        color = colors[i % len(colors)]
        x, y, z = ray[0], ray[1], ray[2]
        px, py, pz = -x, -y, -z  # light position
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-10:
            r = 1.0

        # 표시 좌표: Z만 반전하면 +Z = 시선 방향, 조명(카메라 쪽)이 위쪽에 옴
        px_d, py_d, pz_d = px, py, pz * z_scale
        x_d, y_d, z_d = x, y, z * z_scale

        if light_dir_deg is not None and i < len(light_dir_deg):
            d = light_dir_deg[i]
            azimuth_deg = d.get('azimuth_deg', 0.0)
            elevation_deg = d['elevation_deg']
            azimuth = np.radians(azimuth_deg)
            elevation = np.radians(elevation_deg)
        else:
            azimuth = np.arctan2(y, x)
            elevation = np.arcsin(z / r)
            azimuth_deg = (np.degrees(azimuth) + 360) % 360
            elevation_deg = np.degrees(elevation)

        # 1. 점: 조명 위치 (Z 반전 적용)
        ax.scatter(px_d, py_d, pz_d, color=color, s=100)

        # 2. 라벨
        ax.text(px_d * 1.15, py_d * 1.15, pz_d * 1.15,
               f'L{i+1}\nAz:{azimuth_deg:.1f}°\nEl:{elevation_deg:.1f}°',
               fontsize=10, color=color, alpha=0.9,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor=color, linewidth=0.5))

        # 3. 화살표: light→surface (표시 좌표에서 원점으로)
        ax.quiver(px_d, py_d, pz_d, x_d, y_d, z_d,
                 color=color, arrow_length_ratio=0.2, linewidth=2)

        # 4. Azimuth (XY 평면, Z=0)
        xy_proj = np.sqrt(px**2 + py**2)
        if xy_proj > 1e-10:
            ax.plot([0, px_d], [0, py_d], [0, 0], color=color, linestyle='--', linewidth=0.8, alpha=0.6)
            arc_radius = r * 0.3
            num_points = 20
            az_plot = np.arctan2(py, px)
            arc_angles = np.linspace(0, az_plot, num_points)
            arc_x = arc_radius * np.cos(arc_angles)
            arc_y = arc_radius * np.sin(arc_angles)
            ax.plot(arc_x, arc_y, [0] * num_points, color=color, linestyle='--', linewidth=0.8, alpha=0.5)

        # 5. Elevation 원호 (표시 Z 반전 적용)
        if xy_proj > 1e-10:
            arc_radius_vert = r * 0.3
            num_points_vert = 20
            elev_plot = np.arcsin(pz / r)
            elev_angles = np.linspace(0, elev_plot, num_points_vert)
            cos_az = px / xy_proj
            sin_az = py / xy_proj
            arc_x_vert = arc_radius_vert * np.cos(elev_angles) * cos_az
            arc_y_vert = arc_radius_vert * np.cos(elev_angles) * sin_az
            arc_z_vert = arc_radius_vert * np.sin(elev_angles) * z_scale
            ax.plot(arc_x_vert, arc_y_vert, arc_z_vert, color=color, linestyle='--', linewidth=0.8, alpha=0.5)


def draw_light_vector(light_dir, view_azim=None, view_elev=None, title="Light Vectors", light_dir_deg=None):
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
    light_dir_deg : list of dict, optional
        각 조명의 구면 좌표 정보
        [{'elevation_deg': float, 'azimuth_deg': float}, ...]
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    _draw_vectors(ax, light_dir, light_dir_deg=light_dir_deg)
    
    if view_azim is not None and view_elev is not None:
        ax.view_init(elev=view_elev, azim=view_azim)
    
    _setup_3d_axes(ax)
    ax.set_title(title)
    
    return fig


def save_light_vector_views(light_dir, output_prefix="light_vectors", light_dir_deg=None):
    """
    여러 시점의 조명 벡터를 PNG로 저장
    
    Parameters
    ----------
    light_dir : ndarray of shape (N, 3)
        조명 벡터 배열
    output_prefix : str
        PNG 파일명 접두사
    light_dir_deg : list of dict, optional
        각 조명의 구면 좌표 정보
        [{'elevation_deg': float, 'azimuth_deg': float}, ...]
    
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
            fig = draw_light_vector(light_dir, title=f'Light Vectors - {view_name.capitalize()} View', light_dir_deg=light_dir_deg)
            filename = f"{output_prefix}_current.png"
        else:
            fig = draw_light_vector(light_dir, view_azim=azim, view_elev=elev, 
                                   title=f'Light Vectors - {view_name.capitalize()} View', light_dir_deg=light_dir_deg)
            filename = f"{output_prefix}_{view_name}.png"
        
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        saved_files.append(filename)
        print(f"Saved: {filename}")
        plt.close(fig)
    
    return saved_files

