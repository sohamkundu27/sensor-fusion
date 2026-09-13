"""Sensor poses, pinhole projection, and sparse nearest-depth rasterization."""
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from shapely.geometry import MultiPoint, box as rectangle


def sensor_to_global(nusc, sample_data):
    calibration = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    return (transform_matrix(pose['translation'], Quaternion(pose['rotation'])) @
            transform_matrix(calibration['translation'], Quaternion(calibration['rotation'])))


def transform_points(xyz, matrix):
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def filter_visible_points(uvd, image_hw):
    """Recheck representable coordinates after projection casts or image resizing."""
    h, w = image_hw
    valid = np.isfinite(uvd).all(axis=1) & (uvd[:, 2] > 0)
    valid &= (uvd[:, 0] >= 0) & (uvd[:, 0] < w) & (uvd[:, 1] >= 0) & (uvd[:, 1] < h)
    return uvd[valid]


def project_points(xyz_camera, intrinsic, image_hw, min_depth=1.0, max_depth=100.0):
    """Return visible (u, v, camera-Z metres); camera axes are right/down/forward."""
    h, w = image_hw
    valid = np.isfinite(xyz_camera).all(axis=1)
    valid &= (xyz_camera[:, 2] > min_depth) & (xyz_camera[:, 2] <= max_depth)
    xyz = xyz_camera[valid]
    projected = xyz @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    result = np.column_stack((uv[inside], xyz[inside, 2])).astype(np.float32)
    return filter_visible_points(result, image_hw)


def rasterize_depth(uvd, image_hw):
    """Nearest positive Z wins collisions. Empty pixels are zero with mask=False."""
    h, w = image_hw
    depth = np.full(h * w, np.inf, dtype=np.float32)
    if len(uvd):
        u, v = np.floor(uvd[:, :2]).astype(np.int64).T
        if np.any((u < 0) | (u >= w) | (v < 0) | (v >= h) | (uvd[:, 2] <= 0)):
            raise ValueError('Rasterizer expects in-bounds positive-depth points')
        np.minimum.at(depth, v * w + u, uvd[:, 2])
    mask = np.isfinite(depth)
    depth[~mask] = 0
    return depth.reshape(1, h, w), mask.reshape(1, h, w)


def project_box(corners_camera, intrinsic, image_hw, near=0.1):
    """Clip a 3D box against the near plane, then its projected hull to the image."""
    corners = corners_camera.T
    points = [c for c in corners if c[2] >= near]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        p, q = corners[a], corners[b]
        if (p[2] < near) != (q[2] < near):
            points.append(p + (q - p) * ((near - p[2]) / (q[2] - p[2])))
    if len(points) < 3:
        return None
    uvz = np.asarray(points) @ intrinsic.T
    uv = uvz[:, :2] / uvz[:, 2:3]
    h, w = image_hw
    clipped = MultiPoint(uv).convex_hull.intersection(rectangle(0, 0, w, h))
    if clipped.is_empty or clipped.area <= 0:
        return None
    return np.asarray(clipped.bounds, dtype=np.float32)
