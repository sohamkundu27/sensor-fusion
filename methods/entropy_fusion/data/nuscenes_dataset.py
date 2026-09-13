"""Camera-view nuScenes items with temporally aligned LiDAR/radar depth maps."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name
from .geometry import sensor_to_global, transform_points, project_points, rasterize_depth, project_box, filter_visible_points

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
RADARS = ('RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT')
CLASSES = ('car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian',
           'motorcycle', 'bicycle', 'traffic_cone', 'barrier')


class NuScenesFusionDataset(Dataset):
    """One item per keyframe/camera. Split by scene before expanding camera views.

    RGB is [0,1]; depth inputs are camera-Z/max_depth, zero at missing pixels.
    Availability masks are distinct from real zero values and letterbox padding.
    Labels are 1..10; class 0 is reserved for an eventual SSD background class.
    """
    def __init__(self, root, version='v1.0-mini', split='mini_train',
                 cameras=('CAM_FRONT',), image_hw=(384, 640),
                 min_depth=1.0, max_depth=100.0, nusc=None):
        self.root = Path(root).expanduser().resolve()
        self.version, self.split = version, split
        self.cameras = tuple(cameras)
        self.image_hw = tuple(image_hw)
        self.min_depth, self.max_depth = min_depth, max_depth
        allowed = {'v1.0-mini': ('mini_train', 'mini_val'), 'v1.0-trainval': ('train', 'val')}
        if version not in allowed or split not in allowed[version]:
            raise ValueError('Use mini_train/mini_val with v1.0-mini or train/val with v1.0-trainval')
        if not self.cameras or len(set(self.cameras)) != len(self.cameras) or any(c not in CAMERAS for c in self.cameras):
            raise ValueError('Choose unique supported camera channels')
        if len(self.image_hw) != 2 or any(x <= 0 or x % 16 for x in self.image_hw):
            raise ValueError('image_hw must contain two positive multiples of 16')
        if not 0 < min_depth < max_depth:
            raise ValueError('Expected 0 < min_depth < max_depth')
        self.nusc = nusc or NuScenes(version=version, dataroot=str(self.root), verbose=False)
        if self.nusc.version != version or Path(self.nusc.dataroot).resolve() != self.root:
            raise ValueError('Injected devkit instance has different version/root')
        scene_names = set(create_splits_scenes()[split])
        self.scene_tokens = {s['token'] for s in self.nusc.scene if s['name'] in scene_names}
        samples = sorted((s for s in self.nusc.sample if s['scene_token'] in self.scene_tokens),
                         key=lambda s: (s['scene_token'], s['timestamp']))
        self.items = [(s['token'], c) for s in samples for c in self.cameras]
        if not self.items:
            raise ValueError('No samples found for requested split')

    def __len__(self):
        return len(self.items)

    def _path(self, record):
        path = self.root / record['filename']
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Finish extracting the requested dataset split.')
        return path

    def _project_sensor(self, token, camera_record, intrinsic, original_hw, pixel_transform, output_hw=None):
        record = self.nusc.get('sample_data', token)
        calibration = self.nusc.get('calibrated_sensor', record['calibrated_sensor_token'])
        sensor = self.nusc.get('sensor', calibration['sensor_token'])
        path = self._path(record)
        if sensor['modality'] == 'lidar':
            cloud = LidarPointCloud.from_file(str(path))
        else:
            # Pass explicit defaults; never mutate RadarPointCloud's global filters.
            cloud = RadarPointCloud.from_file(str(path), invalid_states=[0],
                                              dynprop_states=list(range(7)), ambig_states=[3])
        transform = np.linalg.inv(sensor_to_global(self.nusc, camera_record)) @ sensor_to_global(self.nusc, record)
        xyz = transform_points(cloud.points[:3].T, transform)
        uvd = project_points(xyz, intrinsic, original_hw, self.min_depth, self.max_depth)
        if len(uvd):
            homogeneous = np.column_stack((uvd[:, :2], np.ones(len(uvd))))
            uvd[:, :2] = (homogeneous @ pixel_transform.T)[:, :2]
        uvd = filter_visible_points(uvd, original_hw if output_hw is None else output_hw)
        return uvd, {'channel': sensor['channel'], 'token': token,
                     'time_offset_seconds': (record['timestamp'] - camera_record['timestamp']) / 1e6,
                     'sensor_to_camera': transform, 'input_points': cloud.nbr_points(),
                     'projected_points': len(uvd)}

    def __getitem__(self, index):
        token, channel = self.items[index]
        sample = self.nusc.get('sample', token)
        camera = self.nusc.get('sample_data', sample['data'][channel])
        calibration = self.nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
        intrinsic = np.asarray(calibration['camera_intrinsic'], dtype=np.float64)
        with Image.open(self._path(camera)) as image:
            image = image.convert('RGB')
            original_w, original_h = image.size
            h, w = self.image_hw
            scale = min(w / original_w, h / original_h)
            resized_w, resized_h = round(original_w * scale), round(original_h * scale)
            left, top = (w - resized_w) // 2, (h - resized_h) // 2
            canvas = Image.new('RGB', (w, h))
            canvas.paste(image.resize((resized_w, resized_h), Image.Resampling.BILINEAR), (left, top))
            rgb = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        image_mask = np.zeros((1, h, w), dtype=bool)
        image_mask[:, top:top + resized_h, left:left + resized_w] = True
        affine = np.array([[resized_w / original_w, 0, left], [0, resized_h / original_h, top], [0, 0, 1.]])
        kwargs = (camera, intrinsic, (original_h, original_w), affine, self.image_hw)
        lidar, lidar_meta = self._project_sensor(sample['data']['LIDAR_TOP'], *kwargs)
        radar_points, radar_meta = [], []
        for radar in RADARS:
            projected, metadata = self._project_sensor(sample['data'][radar], *kwargs)
            radar_points.append(projected)
            radar_meta.append(metadata)
        radar = np.concatenate(radar_points, axis=0)
        lidar_depth, lidar_mask = rasterize_depth(lidar, self.image_hw)
        radar_depth, radar_mask = rasterize_depth(radar, self.image_hw)
        global_to_camera = np.linalg.inv(sensor_to_global(self.nusc, camera))
        boxes, labels, annotations = [], [], []
        for annotation_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', annotation_token)
            name = category_to_detection_name(ann['category_name'])
            if name is None:
                continue
            corners = self.nusc.get_box(annotation_token).corners()
            corners_camera = transform_points(corners.T, global_to_camera).T
            bbox = project_box(corners_camera, intrinsic, (original_h, original_w))
            if bbox is None:
                continue
            bbox[[0, 2]] = bbox[[0, 2]] * affine[0, 0] + left
            bbox[[1, 3]] = bbox[[1, 3]] * affine[1, 1] + top
            boxes.append(bbox)
            labels.append(CLASSES.index(name) + 1)
            annotations.append({key: ann[key] for key in ('token', 'translation', 'size', 'rotation',
                'attribute_tokens', 'visibility_token', 'num_lidar_pts', 'num_radar_pts')})
            velocity = self.nusc.box_velocity(annotation_token)
            annotations[-1]['velocity_global'] = velocity.tolist()
            annotations[-1]['velocity_valid'] = bool(np.isfinite(velocity).all())
            annotations[-1]['detection_name'] = name
            annotations[-1]['attribute_names'] = [self.nusc.get('attribute', t)['name'] for t in ann['attribute_tokens']]
        pose = self.nusc.get('ego_pose', camera['ego_pose_token'])
        ego_to_global = transform_matrix(pose['translation'], Quaternion(pose['rotation']))
        return {
            'camera': torch.from_numpy(rgb),
            'lidar': torch.from_numpy(lidar_depth / self.max_depth),
            'radar': torch.from_numpy(radar_depth / self.max_depth),
            'camera_mask': torch.from_numpy(image_mask),
            'lidar_mask': torch.from_numpy(lidar_mask),
            'radar_mask': torch.from_numpy(radar_mask),
            'lidar_depth_m': torch.from_numpy(lidar_depth),
            'radar_depth_m': torch.from_numpy(radar_depth),
            'target': {'boxes': torch.as_tensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 4)),
                       'labels': torch.tensor(labels, dtype=torch.int64), 'annotations_3d': annotations},
            'metadata': {'sample_token': token, 'scene_token': sample['scene_token'], 'camera_channel': channel,
                         'camera_token': camera['token'], 'camera_path': str(self._path(camera)),
                         'camera_timestamp': camera['timestamp'], 'original_hw': (original_h, original_w),
                         'image_hw': self.image_hw, 'intrinsic_original': intrinsic,
                         'intrinsic': affine @ intrinsic, 'pixel_transform': affine,
                         'global_to_camera': global_to_camera, 'ego_to_global': ego_to_global, 'sensors': [lidar_meta, *radar_meta]},
            'projections': {'lidar': torch.from_numpy(lidar), 'radar': torch.from_numpy(radar)}
        }


def collate_fusion_batch(items):
    """Stack fixed-sized inputs, retain variable boxes and point lists per item."""
    keys = ('camera', 'lidar', 'radar', 'camera_mask', 'lidar_mask', 'radar_mask', 'lidar_depth_m', 'radar_depth_m')
    batch = {key: torch.stack([item[key] for item in items]) for key in keys}
    batch.update(targets=[x['target'] for x in items], metadata=[x['metadata'] for x in items],
                 projections=[x['projections'] for x in items])
    return batch
