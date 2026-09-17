"""Audit the light-source triangulation against a known source distance.

The pipeline locates the source by intersecting the rays from all spheres,
which needs no extra input. This checks that result against a distance taken
from the optical drawing, so a degraded triangulation can be spotted.

Two things to watch:

  ray residual   how far the fitted source sits from each ray. Grows when
                 spheres are few, poorly detected, or nearly collinear. This
                 one needs no reference and is the primary health signal.

  vs distance    elevation from the triangulated source against the elevation
                 from the drawing distance. A few tenths of a degree is normal;
                 more than that means either the triangulation is degraded or
                 the drawing no longer matches the machine.

Usage
-----
    python standalone_tools/audit_source_triangulation.py \
        --image_pattern <dir> --sphere_diameter 4.0 \
        --pixel_resolution 0.010017874 --num_spheres 7 \
        --remap_dir <lut> --led_distance 81.22
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photometry_calibration import image_utils
from photometry_calibration.auto_calibration import (
    compute_average_image, find_spheres, extract_highlight)
from photometry_calibration.camera_model import (
    load_camera_model, compute_light_vector_exact, triangulate_source,
    source_direction_at, ray_residual, refer_to_fov_centre)
import photometry_calibration.light_vec_calculator as lvcalc


def elevation(L):
    return float(np.degrees(np.arcsin(np.clip(-L[2], -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image_pattern', required=True)
    ap.add_argument('--sphere_diameter', type=float, required=True)
    ap.add_argument('--pixel_resolution', type=float, required=True)
    ap.add_argument('--num_spheres', type=int, required=True)
    ap.add_argument('--remap_dir', required=True)
    ap.add_argument('--led_distance', type=float, required=True,
                    help='FoV centre to source distance from the optical drawing [mm]')
    ap.add_argument('--highlight_method', default='ring', choices=['ring', 'centroid'])
    a = ap.parse_args()

    cam = load_camera_model(a.remap_dir)
    if cam is None:
        sys.exit('카메라 모델을 찾지 못했습니다. --remap_dir 를 확인하세요.')
    print(f'Camera: {cam}')

    map_x, map_y = image_utils.load_map_pair(a.remap_dir)
    _, images = image_utils.load_images_grayscale(a.image_pattern)
    images = image_utils.apply_rectification(images, map_x, map_y)
    avg = compute_average_image(images)

    spheres = find_spheres(avg, a.sphere_diameter, a.pixel_resolution,
                           a.num_spheres, debug_dir=os.path.dirname(a.remap_dir))
    if spheres is None:
        sys.exit('sphere 검출 실패')

    centres = [s['center'] for s in spheres]
    radii = [s['radius'] for s in spheres]
    Rs = a.sphere_diameter / 2.0
    d_focal = cam.focal_px * a.pixel_resolution
    origin = np.array([0.0, 0.0, -d_focal])

    all_c, all_h, all_p = [], [], []
    for img in images:
        for (cx, cy), r in zip(centres, radii):
            res = extract_highlight(img, (cx, cy), r)
            if res is None:
                all_c.append((cx, cy)); all_h.append(((0, 0), (0, 0))); all_p.append(None)
                continue
            _, contour, pixels = res
            xs = [p[0] for p in contour]; ys = [p[1] for p in contour]
            all_c.append((cx, cy))
            all_h.append(((min(xs), min(ys)), (max(xs), max(ys))))
            all_p.append(pixels)
    uv, _ = lvcalc.build_highlight_position_list(
        a.highlight_method, all_c, all_h, all_p, len(images), len(spheres))

    print(f'\n초점면까지 {d_focal:.2f} mm,  도면상 광원 거리 {a.led_distance:.2f} mm')
    print(f"\n{'조명':>5}{'삼각측량 위치 (mm)':>30}{'거리':>9}{'잔차':>8}"
          f"{'삼각측량':>10}{'도면거리':>10}{'차이':>8}")
    rows = []
    for li in range(len(images)):
        pts, dirs = [], []
        for si in range(len(spheres)):
            L, P = compute_light_vector_exact(*uv[li][si], centres[si], radii[si],
                                              cam, Rs, return_point=True)
            if L is not None:
                pts.append(P); dirs.append(L)
        src = triangulate_source(pts, dirs)
        if src is None:
            print(f'{li:>5}  삼각측량 실패 (유효 sphere {len(pts)}개)')
            continue
        e_tri = elevation(source_direction_at(src, origin))
        e_ref = float(np.mean([elevation(refer_to_fov_centre(L, P, d_focal, a.led_distance))
                               for P, L in zip(pts, dirs)
                               if refer_to_fov_centre(L, P, d_focal, a.led_distance) is not None]))
        resid = ray_residual(src, pts, dirs)
        rows.append((e_tri, e_ref, resid))
        print(f'{li:>5}{str(np.round(src, 2)):>30}{np.linalg.norm(src - origin):>9.2f}'
              f'{resid:>8.2f}{e_tri:>10.3f}{e_ref:>10.3f}{e_tri - e_ref:>+8.3f}')

    if rows:
        arr = np.array(rows)
        print(f"\n{'평균':>5}{'':>30}{'':>9}{arr[:, 2].mean():>8.2f}"
              f"{arr[:, 0].mean():>10.3f}{arr[:, 1].mean():>10.3f}"
              f"{(arr[:, 0] - arr[:, 1]).mean():>+8.3f}")
        gap = float(np.abs(arr[:, 0] - arr[:, 1]).max())
        print(f'\n최대 차이 {gap:.3f} deg,  최대 잔차 {arr[:, 2].max():.2f} mm')
        if gap > 0.5:
            print('  [주의] 차이가 0.5 deg 를 넘습니다. 삼각측량이 열화되었거나 '
                  '도면 거리가 실제와 다를 수 있습니다.')
        else:
            print('  삼각측량이 도면 거리와 일치하는 범위입니다.')


if __name__ == '__main__':
    main()
