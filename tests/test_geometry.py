import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from data.geometry import project_points, rasterize_depth, transform_points, project_box, sensor_to_global


def test_projection_rejects_behind_camera_and_out_of_frame():
    xyz = np.array([[0, 0, 10], [0, 0, -2], [100, 0, 2], [0, 0, np.nan], [0, 0, 101]])
    k = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1.]])
    np.testing.assert_allclose(project_points(xyz, k, (10, 10)), [[5, 5, 10]])


def test_zbuffer_keeps_nearest_point_independent_of_input_order():
    points = np.array([[2.1, 3.9, 20], [2.9, 3.1, 4], [0, 0, 8]], dtype=np.float32)
    a, mask = rasterize_depth(points, (8, 8))
    b, _ = rasterize_depth(points[::-1], (8, 8))
    np.testing.assert_array_equal(a, b)
    assert a[0, 3, 2] == 4 and mask.sum() == 2
    empty, valid = rasterize_depth(np.empty((0, 3)), (8, 8))
    assert not empty.any() and not valid.any()


def test_timestamp_specific_ego_poses_are_composed():
    class FakeNuScenes:
        records = {
            ('calibrated_sensor', 'lidar'): {'translation': [1, 0, 0], 'rotation': [1, 0, 0, 0]},
            ('calibrated_sensor', 'camera'): {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]},
            ('ego_pose', 'early'): {'translation': [10, 0, 0], 'rotation': Quaternion(axis=[0, 0, 1], angle=np.pi/2).elements},
            ('ego_pose', 'late'): {'translation': [20, 0, 0], 'rotation': [1, 0, 0, 0]},
        }
        def get(self, table, token):
            return self.records[table, token]
    nusc = FakeNuScenes()
    lidar = {'calibrated_sensor_token': 'lidar', 'ego_pose_token': 'early'}
    camera = {'calibrated_sensor_token': 'camera', 'ego_pose_token': 'late'}
    transform = np.linalg.inv(sensor_to_global(nusc, camera)) @ sensor_to_global(nusc, lidar)
    np.testing.assert_allclose(transform_points(np.array([[1., 0, 0]]), transform), [[-10, 2, 0]], atol=1e-10)


def test_box_clipping_handles_near_plane_and_behind_camera():
    k = np.array([[30, 0, 32], [0, 30, 32], [0, 0, 1.]])
    behind = Box([0, 0, -10], [2, 2, 2], Quaternion())
    assert project_box(behind.corners(), k, (64, 64)) is None
    crossing = Box([0, 0, 0.2], [2, 2, 2], Quaternion())
    np.testing.assert_allclose(project_box(crossing.corners(), k, (64, 64)), [0, 0, 64, 64])
