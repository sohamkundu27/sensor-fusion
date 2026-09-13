"""Read one real mini batch, compare devkit projections, and export overlays."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from data import NuScenesFusionDataset, collate_fusion_batch, CLASSES, CAMERAS


def compare_devkit(dataset, item):
    """Independent official projection path catches transform/order mistakes."""
    meta = item['metadata']
    errors = {}
    for sensor in meta['sensors']:
        reference, depths, _ = dataset.nusc.explorer.map_pointcloud_to_image(
            sensor['token'], meta['camera_token'], min_dist=dataset.min_depth)
        reference = reference[:, depths <= dataset.max_depth]
        depths = depths[depths <= dataset.max_depth]
        actual, _ = dataset._project_sensor(sensor['token'],
            dataset.nusc.get('sample_data', meta['camera_token']),
            meta['intrinsic_original'], meta['original_hw'], np.eye(3))
        # Devkit excludes a 1px border; our data loader accepts the whole image.
        h, w = meta['original_hw']
        actual = actual[(actual[:, 0] > 1) & (actual[:, 0] < w-1) &
                        (actual[:, 1] > 1) & (actual[:, 1] < h-1)]
        expected = np.column_stack((reference[:2].T, depths))
        # Devkit mutates float32 points through large global translations;
        # our composed float64 transform avoids that intermediate rounding.
        np.testing.assert_allclose(actual[:, :2], expected[:, :2], atol=0.05, rtol=0)
        np.testing.assert_allclose(actual[:, 2], expected[:, 2], atol=0.002, rtol=0)
        errors[sensor['channel']] = float(np.abs(actual - expected).max()) if len(actual) else 0.0
    return errors


def visualize(item, path):
    image = Image.fromarray((item['camera'].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    w, h = image.size
    panel = Image.new('RGB', (w * 3, h + 28))
    for i, modality in enumerate(('camera', 'lidar', 'radar')):
        view = image.copy()
        draw = ImageDraw.Draw(view)
        if modality == 'camera':
            for box, label in zip(item['target']['boxes'], item['target']['labels']):
                draw.rectangle(box.tolist(), outline='yellow', width=2)
                draw.text(tuple(box[:2].tolist()), CLASSES[label.item()-1], fill='yellow')
        else:
            color = 'cyan' if modality == 'lidar' else 'magenta'
            radius = 1 if modality == 'lidar' else 4
            for u, v, _ in item['projections'][modality].numpy():
                draw.ellipse((u-radius, v-radius, u+radius, v+radius), fill=color)
        panel.paste(view, (i*w, 28))
    draw = ImageDraw.Draw(panel)
    for i, title in enumerate(('Projected GT boxes', 'LiDAR (cyan)', 'Radar (magenta)')):
        draw.text((i*w+8, 6), title, fill='white')
    panel.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path.home() / 'data/nuscenes'))
    parser.add_argument('--split', choices=['mini_train', 'mini_val'], default='mini_train')
    parser.add_argument('--cameras', nargs='+', choices=CAMERAS, default=['CAM_FRONT'])
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('outputs/mini_batch'))
    args = parser.parse_args()
    dataset = NuScenesFusionDataset(args.root, split=args.split, cameras=args.cameras)
    other = NuScenesFusionDataset(args.root, split='mini_val' if args.split == 'mini_train' else 'mini_train',
                                 cameras=args.cameras, nusc=dataset.nusc)
    assert dataset.scene_tokens.isdisjoint(other.scene_tokens)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fusion_batch)
    batch = next(iter(loader))
    for key in ('camera', 'lidar', 'radar'):
        assert torch.isfinite(batch[key]).all()
        assert batch[key].min() >= 0 and batch[key].max() <= 1
    for modality in ('lidar', 'radar'):
        assert torch.equal(batch[modality + '_mask'], batch[modality + '_depth_m'] > 0)
        assert not (batch[modality + '_mask'] & ~batch['camera_mask']).any()
    for target in batch['targets']:
        boxes = target['boxes']
        assert len(boxes) == len(target['labels']) == len(target['annotations_3d'])
        if len(boxes):
            assert ((boxes[:, 2:] - boxes[:, :2]) > 0).all()
            assert boxes.min() >= 0 and boxes[:, [0, 2]].max() <= 640 and boxes[:, [1, 3]].max() <= 384
            assert target['labels'].min() >= 1 and target['labels'].max() <= len(CLASSES)
    item = dataset[0]
    errors = compare_devkit(dataset, item)
    args.output.mkdir(parents=True, exist_ok=True)
    visualize(item, args.output / 'projection_overlay.png')
    cuda_checked = torch.cuda.is_available()
    if cuda_checked:
        inputs = {key: batch[key].cuda() for key in ('camera', 'lidar', 'radar')}
        torch.cuda.synchronize()
        del inputs
    report = {'status': 'PASS', 'split': args.split, 'dataset_items': len(dataset),
              'split_scenes': len(dataset.scene_tokens), 'disjoint_scene_splits': True,
              'shapes': {k: list(batch[k].shape) for k in ('camera', 'lidar', 'radar')},
              'lidar_valid_pixels': batch['lidar_mask'].sum(dim=(1,2,3)).tolist(),
              'radar_valid_pixels': batch['radar_mask'].sum(dim=(1,2,3)).tolist(),
              'boxes_per_item': [len(t['boxes']) for t in batch['targets']],
              'official_projection_max_abs_error': errors, 'gpu_transfer_checked': cuda_checked,
              'sample_tokens': [m['sample_token'] for m in batch['metadata']],
              'camera_channels': [m['camera_channel'] for m in batch['metadata']],
              'workers': args.workers, 'model_built': False, 'training_run': False}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Overlay:', args.output / 'projection_overlay.png')


if __name__ == '__main__':
    main()
