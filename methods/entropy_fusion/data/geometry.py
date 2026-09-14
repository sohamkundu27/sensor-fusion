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


def filter_visible_points(uvd, image_hw, return_mask=False):
    """Recheck representable coordinates after projection casts or image resizing."""
    h, w = image_hw
    valid = np.isfinite(uvd).all(axis=1) & (uvd[:, 2] > 0)
    valid &= (uvd[:, 0] >= 0) & (uvd[:, 0] < w) & (uvd[:, 1] >= 0) & (uvd[:, 1] < h)
    return (uvd[valid], valid) if return_mask else uvd[valid]


def project_points(xyz_camera, intrinsic, image_hw, min_depth=1.0, max_depth=100.0,
                   return_indices=False):
    """Return visible (u, v, camera-Z metres), optionally with source row indices.

    Indices survive every depth, field-of-view and float32 boundary filter, so
    intensity/velocity attributes cannot silently move to a different point.
    Camera axes are right/down/forward.
    """
    h, w = image_hw
    valid = np.isfinite(xyz_camera).all(axis=1)
    valid &= (xyz_camera[:, 2] > min_depth) & (xyz_camera[:, 2] <= max_depth)
    xyz = xyz_camera[valid]
    projected = xyz @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    result = np.column_stack((uv[inside], xyz[inside, 2])).astype(np.float32)
    result, representable = filter_visible_points(result, image_hw, return_mask=True)
    if return_indices:
        return result, np.flatnonzero(valid)[inside][representable]
    return result


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


def rasterize_features(uvd, features, image_hw, image_mask=None, vertical=False):
    """Rasterize all attributes from the same nearest-Z return at each pixel.

    With ``vertical=True`` the nearest return in each image column is repeated
    vertically, following the paper's elevation-invariant radar representation.
    An image mask excludes letterboxing. No values are averaged across returns;
    exact depth ties deterministically keep the first source return.
    """
    h, w = image_hw
    if uvd.ndim != 2 or uvd.shape[1] != 3 or features.ndim != 2 or len(features) != len(uvd):
        raise ValueError('Expected matching Nx3 coordinates and NxC features')
    if not np.isfinite(features).all():
        raise ValueError('Rasterizer expects finite point features')
    allowed = np.ones((h, w), dtype=bool) if image_mask is None else np.asarray(image_mask, dtype=bool).reshape(h, w)
    output = np.zeros((features.shape[1], h * w), dtype=np.float32)
    mask = np.zeros(h * w, dtype=bool)
    if len(uvd):
        visible, valid = filter_visible_points(uvd, image_hw, return_mask=True)
        if not valid.all():
            raise ValueError('Rasterizer expects in-bounds positive-depth points')
        u, v = np.floor(visible[:, :2]).astype(np.int64).T
        keys = u if vertical else v * w + u
        order = np.lexsort((np.arange(len(uvd)), uvd[:, 2], keys))
        _, first = np.unique(keys[order], return_index=True)
        winners = order[first]
        if vertical:
            columns = np.zeros((features.shape[1], w), dtype=np.float32)
            column_mask = np.zeros(w, dtype=bool)
            columns[:, u[winners]] = features[winners].T
            column_mask[u[winners]] = True
            output = np.repeat(columns[:, None, :], h, axis=1).reshape(features.shape[1], h * w)
            mask = np.broadcast_to(column_mask, (h, w)).reshape(-1).copy()
        else:
            output[:, keys[winners]] = features[winners].T
            mask[keys[winners]] = True
    mask &= allowed.reshape(-1)
    output[:, ~mask] = 0
    return output.reshape(features.shape[1], h, w), mask.reshape(1, h, w)


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
