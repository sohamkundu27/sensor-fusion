import numpy as np
import torch
from pyquaternion import Quaternion
from models.boxes import prepare_targets, encode_3d, decode_3d, encode_boxes, decode_boxes


def test_metric_box_roundtrip_with_rotated_translated_camera():
    e2g = np.eye(4)
    e2g[:3, :3] = Quaternion(axis=[0, 0, 1], radians=.8).rotation_matrix
    e2g[:3, 3] = [100, -20, 2]
    c2e = np.eye(4)
    c2e[:3, :3] = Quaternion(axis=[0, 0, 1], radians=.6).rotation_matrix @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    c2e[:3, 3] = [1, .2, 1.5]
    c2g = e2g @ c2e
    center = c2g[:3, :3] @ np.array([2., 1., 20.]) + c2g[:3, 3]
    rotation = Quaternion(matrix=e2g[:3, :3]) * Quaternion(axis=[0, 0, 1], radians=-.4)
    velocity = e2g[:3, :3] @ np.array([3., 2., 0.])
    meta = dict(global_to_camera=np.linalg.inv(c2g), ego_to_global=e2g,
                intrinsic=np.array([[300., 0, 320], [0, 300, 192], [0, 0, 1]]))
    ann = dict(translation=center, size=[2., 4., 1.5], rotation=rotation.elements,
               num_lidar_pts=5, num_radar_pts=0, velocity_valid=True, velocity_global=velocity,
               attribute_names=['vehicle.moving'])
    target = dict(boxes=torch.tensor([[250., 100, 380, 260]]), labels=torch.tensor([1]), annotations_3d=[ann])
    encoded = prepare_targets(target, meta, 'cpu')
    anchors = torch.tensor([[240., 90, 390, 270]])
    xyz, size, rotations, vel = decode_3d(encode_3d(encoded['values'], anchors), anchors, meta)
    np.testing.assert_allclose(xyz[0], center, atol=1e-5)
    np.testing.assert_allclose(size[0], ann['size'], atol=1e-5)
    np.testing.assert_allclose(vel[0], velocity[:2], atol=1e-5)
    np.testing.assert_allclose(Quaternion(rotations[0]).rotation_matrix, rotation.rotation_matrix, atol=1e-6)
    torch.testing.assert_close(decode_boxes(encode_boxes(target['boxes'], anchors), anchors), target['boxes'])
    ann['velocity_valid'] = False
    ann['velocity_global'] = [float('nan')]*3
    masked = prepare_targets(target, meta, 'cpu')
    assert torch.isfinite(masked['values']).all() and not masked['velocity_valid'].any()
