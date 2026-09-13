"""Anchor encodings and camera-ray 3D decoding with explicit coordinate frames."""
import numpy as np
import torch
from pyquaternion import Quaternion

ATTRIBUTES = ('cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
              'pedestrian.standing', 'pedestrian.sitting_lying_down',
              'vehicle.moving', 'vehicle.parked', 'vehicle.stopped')


def center_size(boxes):
    return (boxes[:, :2] + boxes[:, 2:]) / 2, (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-4)


def encode_boxes(boxes, anchors):
    center, size = center_size(boxes)
    ac, az = center_size(anchors)
    return torch.cat(((center-ac)/az, (size/az).log()), 1)


def decode_boxes(code, anchors):
    ac, az = center_size(anchors)
    center = code[:, :2] * az + ac
    size = code[:, 2:].clamp(-6, 6).exp() * az
    return torch.cat((center-size/2, center+size/2), 1)


def prepare_targets(target, metadata, device):
    """Convert global annotations to camera center/ray and camera-time ego yaw/velocity."""
    g2c = np.asarray(metadata['global_to_camera'])
    e2g = np.asarray(metadata['ego_to_global'])
    intrinsic = np.asarray(metadata['intrinsic'])
    keep, values, velocity_valid, attributes = [], [], [], []
    for i, ann in enumerate(target['annotations_3d']):
        center = g2c[:3, :3] @ np.asarray(ann['translation']) + g2c[:3, 3]
        if not np.isfinite(center).all() or not 1 < center[2] <= 100:
            continue
        if ann['num_lidar_pts'] + ann['num_radar_pts'] == 0:
            continue
        uv = intrinsic @ center
        uv = uv[:2] / uv[2]
        local_rotation = e2g[:3, :3].T @ Quaternion(ann['rotation']).rotation_matrix
        yaw = np.arctan2(local_rotation[1, 0], local_rotation[0, 0])
        valid = ann['velocity_valid']
        velocity = e2g[:3, :3].T @ np.asarray(ann['velocity_global']) if valid else np.zeros(3)
        values.append([*uv, np.log(center[2]), *np.log(ann['size']), np.sin(yaw), np.cos(yaw), *(velocity[:2]/10)])
        keep.append(i)
        velocity_valid.append(valid)
        names = ann.get('attribute_names', [])
        attributes.append(ATTRIBUTES.index(names[0]) if names and names[0] in ATTRIBUTES else -1)
    index = torch.tensor(keep, dtype=torch.long)
    return dict(boxes=target['boxes'][index].to(device), labels=target['labels'][index].to(device),
                values=torch.tensor(np.asarray(values).reshape(-1, 10), dtype=torch.float32, device=device),
                velocity_valid=torch.tensor(velocity_valid, dtype=torch.bool, device=device),
                attributes=torch.tensor(attributes, dtype=torch.long, device=device))


def encode_3d(values, anchors):
    ac, az = center_size(anchors)
    return torch.cat(((values[:, :2]-ac)/az, values[:, 2:]), 1)


def decode_3d(code, anchors, metadata):
    """Decode global center, w/l/h, yaw quaternion and global XY velocity."""
    ac, az = center_size(anchors)
    uv = code[:, :2] * az + ac
    depth = code[:, 2].clamp(0, np.log(100)).exp()
    intrinsic = torch.as_tensor(metadata['intrinsic'], device=code.device, dtype=code.dtype)
    c2g = torch.as_tensor(np.linalg.inv(metadata['global_to_camera']), device=code.device, dtype=code.dtype)
    e2g = torch.as_tensor(metadata['ego_to_global'], device=code.device, dtype=code.dtype)
    rays = torch.cat((uv, torch.ones_like(uv[:, :1])), 1) @ torch.linalg.inv(intrinsic).T
    camera_xyz = rays * depth[:, None]
    xyz = camera_xyz @ c2g[:3, :3].T + c2g[:3, 3]
    size = code[:, 3:6].clamp(np.log(.1), np.log(30)).exp()
    yaw = torch.atan2(code[:, 6], code[:, 7])
    local_velocity = torch.cat((code[:, 8:10] * 10, torch.zeros_like(code[:, :1])), 1)
    velocity = (local_velocity @ e2g[:3, :3].T)[:, :2]
    # Upright boxes in the local ego frame, transformed using its full rotation.
    rotations = [(Quaternion(matrix=np.asarray(metadata['ego_to_global'])[:3, :3]) *
                  Quaternion(axis=[0, 0, 1], radians=float(y))).elements.tolist() for y in yaw.detach().cpu()]
    return xyz, size, rotations, velocity
