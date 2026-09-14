"""Correspondence and masking checks for the opt-in paper-style sensor inputs."""
from pathlib import Path
import numpy as np
import pytest

from data.geometry import project_points, rasterize_depth, rasterize_features
from data.nuscenes_dataset import NuScenesFusionDataset, LidarPointCloud, RadarPointCloud


def test_projection_source_indices_survive_all_visibility_filters():
    xyz = np.array([[0., 0, -1], [50., 50, 10], [15999.9999, 1000, 10],
                    [np.nan, 0, 10], [100., 200, 10]])
    uvd, indices = project_points(xyz, np.eye(3), (900, 1600), return_indices=True)
    np.testing.assert_array_equal(indices, [1, 4])
    np.testing.assert_allclose(uvd, [[5, 5, 10], [10, 20, 10]])


def test_feature_collision_keeps_complete_nearest_return():
    uvd = np.array([[2.1, 3.9, 20], [2.9, 3.1, 4], [0, 0, 8]], dtype=np.float32)
    features = np.array([[.2, .9, .1], [.04, .3, .8], [.08, .4, .7]], dtype=np.float32)
    encoded, valid = rasterize_features(uvd, features, (8, 8))
    reverse, reverse_valid = rasterize_features(uvd[::-1], features[::-1], (8, 8))
    np.testing.assert_array_equal(encoded, reverse)
    np.testing.assert_array_equal(valid, reverse_valid)
    np.testing.assert_array_equal(encoded[:, 3, 2], features[1])
    depth, depth_valid = rasterize_depth(uvd, (8, 8))
    np.testing.assert_allclose(encoded[:1] * 100, depth)
    np.testing.assert_array_equal(valid, depth_valid)


def test_vertical_radar_uses_nearest_column_return_and_excludes_letterbox():
    uvd = np.array([[2.1, 2.1, 20], [2.9, 4.1, 4], [5.2, 3.1, 8]], dtype=np.float32)
    features = np.array([[.2, .9, .1], [.04, .3, .8], [.08, .4, .7]], dtype=np.float32)
    image_mask = np.zeros((1, 8, 8), dtype=bool)
    image_mask[:, 2:6, 1:7] = True
    encoded, valid = rasterize_features(uvd, features, (8, 8), image_mask, vertical=True)
    assert valid.sum() == 8
    np.testing.assert_array_equal(encoded[:, 2:6, 2], np.repeat(features[1, :, None], 4, axis=1))
    np.testing.assert_array_equal(encoded[:, 2:6, 5], np.repeat(features[2, :, None], 4, axis=1))
    assert not (valid & ~image_mask).any()
    assert not encoded[:, ~valid[0]].any()


def test_empty_sensor_features_have_zero_values_and_no_availability():
    for vertical in (False, True):
        encoded, valid = rasterize_features(np.empty((0, 3)), np.empty((0, 3)), (8, 8), vertical=vertical)
        assert encoded.shape == (3, 8, 8) and valid.shape == (1, 8, 8)
        assert not encoded.any() and not valid.any()
    with pytest.raises(ValueError, match='in-bounds'):
        rasterize_features(np.array([[8., 0, 4.]]), np.ones((1, 3)), (8, 8))


def _sensor_dataset(modality):
    sensor_calibration = {'translation': [0, 0, 1.], 'rotation': [1, 0, 0, 0], 'sensor_token': 'sensor'}
    camera_calibration = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}
    record = {'calibrated_sensor_token': 'sensor_calibration', 'ego_pose_token': 'pose', 'timestamp': 0}
    camera = {'calibrated_sensor_token': 'camera_calibration', 'ego_pose_token': 'pose', 'timestamp': 0}

    class FakeNuScenes:
        def get(self, table, token):
            return {('sample_data', 'token'): record,
                    ('calibrated_sensor', 'sensor_calibration'): sensor_calibration,
                    ('calibrated_sensor', 'camera_calibration'): camera_calibration,
                    ('sensor', 'sensor'): {'modality': modality, 'channel': modality.upper()},
                    ('ego_pose', 'pose'): {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}}[table, token]

    dataset = NuScenesFusionDataset.__new__(NuScenesFusionDataset)
    dataset.nusc, dataset.min_depth, dataset.max_depth = FakeNuScenes(), 1., 100.
    dataset._path = lambda record: Path('unused')
    return dataset, camera


def test_lidar_height_intensity_remain_aligned_with_surviving_points(monkeypatch):
    dataset, camera = _sensor_dataset('lidar')
    # The middle point is behind the camera and must lose its intensity too.
    points = np.array([[0, 0, 2., 128], [0, 0, -3., 255], [1., 0, 3., 32]], dtype=np.float32).T
    monkeypatch.setattr(LidarPointCloud, 'from_file', lambda *a, **kw: LidarPointCloud(points))
    uvd, features, metadata = dataset._project_sensor('token', camera, np.eye(3), (8, 8), np.eye(3),
                                                     return_features=True)
    np.testing.assert_allclose(features, [[.03, .75, 128 / 255], [.04, .875, 32 / 255]])
    np.testing.assert_allclose(uvd[:, 2], [3, 4])
    assert metadata['projected_points'] == 2
    legacy_uvd, _ = dataset._project_sensor('token', camera, np.eye(3), (8, 8), np.eye(3))
    np.testing.assert_array_equal(uvd, legacy_uvd)


def test_radar_uses_compensated_radial_velocity_and_rcs(monkeypatch):
    dataset, camera = _sensor_dataset('radar')
    points = np.zeros((18, 2), dtype=np.float32)
    points[:3] = np.array([[3., 4., 3.], [3., 4., 3.]]).T
    points[5] = [-20., 20.]  # RCS, dBsm.
    points[6:8] = 999.  # Raw velocity deliberately differs; never use these fields.
    points[8:10] = np.array([[3., 4.], [-3., -4.]]).T  # +/-5 m/s radial.
    monkeypatch.setattr(RadarPointCloud, 'from_file', lambda *a, **kw: RadarPointCloud(points))
    _, features, _ = dataset._project_sensor('token', camera, np.eye(3), (8, 8), np.eye(3),
                                             return_features=True)
    np.testing.assert_allclose(features, [[.04, .25, 35 / 60], [.04, .75, 25 / 60]])
