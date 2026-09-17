"""Rectified-camera model and the exact (perspective) sphere reflection.

The default light-vector path assumes that magnification is uniform over the
whole sphere and that the view direction is (0, 0, 1) for every pixel.  Both
are only true for an orthographic camera.  A real reflecting point sits
``Rs*cos(theta)`` closer to the optics than the sphere centre, so it is
magnified more, and the ray through its pixel is tilted by ``atan(r/f)``.

The two effects partly cancel; the net elevation error is

    d(elevation) = (Rs/d) * (r/R)  ~  r / f      [rad]

which is +0.45 deg at r = 92.5 px, f = 11710 px.  Using the exact model
removes it.  Everything here needs the focal length, which is what sets how
far the camera is from orthographic (orthographic = infinite f).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CameraModel:
    """Pinhole model of the rectified image."""

    focal_px: float
    center_x: float
    center_y: float
    source: str = ""

    def __str__(self):
        return (f"f={self.focal_px:.2f} px, "
                f"principal=({self.center_x:.2f}, {self.center_y:.2f})")


_CALIB_GLOBS = ("*CalibrationResult.json", "*calibration*.json", "*.json")


def load_camera_model(map_dir) -> Optional[CameraModel]:
    """Read the rectified camera model from the remap directory.

    Looks for ``LotaCalibrationResult.json`` (``stereo_calib`` block) next to
    the remap maps.  Returns None when no usable file is found, so callers can
    fall back to the orthographic path instead of failing.
    """
    if not map_dir:
        return None
    d = Path(map_dir)
    if not d.is_dir():
        return None

    seen = set()
    for pattern in _CALIB_GLOBS:
        for path in sorted(d.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            sc = data.get("stereo_calib")
            if not isinstance(sc, dict):
                continue
            f = sc.get("CameraFocalLength")
            cx = sc.get("CameraCenterX")
            cy = sc.get("CameraCenterY")
            if f is None or cx is None or cy is None:
                continue
            if not (float(f) > 0):
                continue
            return CameraModel(float(f), float(cx), float(cy), str(path))
    return None


def sphere_centre_distance(radius_px: float, sphere_radius_mm: float,
                           focal_px: float) -> float:
    """Distance to the sphere centre from its silhouette radius.

    The silhouette is the tangent circle, not a great circle::

        r_sil = f * Rs / sqrt(d^2 - Rs^2)   ->   d = Rs * sqrt((f/r_sil)^2 + 1)
    """
    return sphere_radius_mm * np.sqrt((focal_px / radius_px) ** 2 + 1.0)


def refer_to_fov_centre(
    L: np.ndarray,
    P: np.ndarray,
    focal_plane_distance_mm: float,
    led_distance_mm: float,
) -> Optional[np.ndarray]:
    """Move a light direction measured at P to the FoV centre on the focal plane.

    The reflection happens on the sphere surface, not at the point the CAD
    figure is drawn from.  With the equator on the focal plane the two differ
    by about 0.6 deg; with the sphere bottom on it, 1.8 deg.  The gap scales as
    Rs/D, so it is a property of sphere-based calibration rather than an error.

    Needs the LED-to-specimen distance, which is an optical hardware parameter
    with no other source in the pipeline. Prefer fixing the target and defining
    the expected value offline; see the Confluence page. Off by default.

    Parameters
    ----------
    L : light direction at P (source -> surface), unit
    P : reflection point in camera coordinates [mm]
    focal_plane_distance_mm : distance to the focal plane, f * pixel_resolution
    led_distance_mm : straight-line FoV centre to LED distance [mm]

    Returns
    -------
    light direction at the FoV centre, same convention, or None if the
    LED cannot be placed at the given distance.
    """
    O = np.array([0.0, 0.0, -float(focal_plane_distance_mm)])
    w = P - O
    # LED sits at P - t*L with t > 0; require |LED - O| = led_distance
    b = float(np.dot(w, L))
    disc = b * b - (float(np.dot(w, w)) - led_distance_mm ** 2)
    if disc < 0.0:
        return None
    t = b + np.sqrt(disc)
    if t <= 0.0:
        return None
    led = P - t * L
    d = led - O
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    return -d / n


def compute_light_vector_exact(
    u: float,
    v: float,
    sphere_center: tuple,
    radius_px: float,
    camera: CameraModel,
    sphere_radius_mm: float,
    return_point: bool = False,
):
    """Light vector for one highlight, without the linear-magnification assumption.

    Traces the actual camera ray through the highlight pixel, intersects it
    with the sphere in 3D, and reflects the true view direction about the true
    surface normal.  Off-axis spheres are handled correctly as well, because
    the ray tilt is taken from the absolute pixel position.

    Parameters
    ----------
    u, v : highlight offset from the sphere centre [px]
    sphere_center : (cx, cy) absolute sphere centre in the image [px]
    radius_px : sphere silhouette radius [px]
    camera : CameraModel of the rectified image
    sphere_radius_mm : physical sphere radius [mm]

    return_point : bool
        also return the 3D reflection point, which refer_to_fov_centre needs

    Returns
    -------
    L : unit light vector (u, v, w), same convention as the orthographic path
        (+w towards the camera, L points from the source to the surface), or
        None when the ray misses the sphere. With return_point, (L, P).
    """
    f = camera.focal_px
    cx, cy = sphere_center
    Rs = float(sphere_radius_mm)

    d = sphere_centre_distance(radius_px, Rs, f)

    # sphere centre in 3D; +w is towards the camera, so the scene is at -w
    axis = np.array([cx - camera.center_x, cy - camera.center_y, -f])
    axis /= np.linalg.norm(axis)
    C = d * axis

    # ray through the highlight pixel
    ray = np.array([cx + u - camera.center_x, cy + v - camera.center_y, -f])
    ray /= np.linalg.norm(ray)

    # ray-sphere intersection, near root
    b = np.dot(ray, C)
    disc = b * b - (np.dot(C, C) - Rs * Rs)
    if disc < 0.0:
        return (None, None) if return_point else None
    t = b - np.sqrt(disc)
    if t <= 0.0:
        return (None, None) if return_point else None

    P = t * ray
    N = (P - C) / Rs
    V = -P / np.linalg.norm(P)          # surface -> camera

    L = V - 2.0 * np.dot(N, V) * N      # source -> surface
    L = L / np.linalg.norm(L)
    return (L, P) if return_point else L


def elevation_correction_deg(radius_highlight_px: float, focal_px: float) -> float:
    """Closed-form size of what the linear-magnification assumption costs.

    Diagnostic only; the vector path above applies the correction exactly.
    """
    return float(np.degrees(radius_highlight_px / focal_px))
