#!/usr/bin/env python3
"""Validate complete trainval input, then explicitly opt into legacy conversion."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def check_input(root):
    version = root / 'v1.0-trainval'
    for name in ('maps', 'samples', 'sweeps', 'v1.0-trainval'):
        if not (root / name).is_dir():
            raise RuntimeError('Missing directory: {}'.format(root / name))
    def table(name):
        with (version / (name + '.json')).open() as stream:
            return json.load(stream)
    samples = table('sample')
    scenes = table('scene')
    if len(scenes) != 850 or len(samples) != 34149:
        raise RuntimeError('Expected full trainval metadata: 850 scenes / 34149 samples; got {} / {}'.format(len(scenes), len(samples)))
    sensors = {x['token']: x for x in table('sensor')}
    calibrations = {x['token']: sensors[x['sensor_token']] for x in table('calibrated_sensor')}
    missing = []
    checked = 0
    for item in table('sample_data'):
        sensor = calibrations[item['calibrated_sensor_token']]
        if sensor['modality'] not in ('lidar', 'camera'):
            continue
        path = root / item['filename']
        checked += 1
        if not path.is_file() or path.stat().st_size == 0:
            if len(missing) < 20:
                missing.append(str(path))
    for item in table('map'):
        path = root / item['filename']
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        raise RuntimeError('Missing/empty files (up to first 20 sensor paths):\n' + '\n'.join(missing))
    print('Input check passed: {} LiDAR/camera files, 850 scenes, 34149 keyframes.'.format(checked))
    print('File presence does not replace downloaded archive checksum verification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/nuscenes', help='Run from repository root; symlinks supported.')
    parser.add_argument('--run', action='store_true', help='Generate trainval info, 2D annotations, and GT database after checks. Never trains.')
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    check_input(root)
    if not args.run:
        print('Check only. Add --run after the download and checksum verification finish.')
        return
    outputs = [root / name for name in ('nuscenes_infos_train.pkl', 'nuscenes_infos_val.pkl', 'nuscenes_dbinfos_train.pkl', 'nuscenes_gt_database')]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise RuntimeError('Move existing preparation outputs aside before regenerating:\n' + '\n'.join(existing))
    # Import this checkout directly: tools/create_data.py parses argv at import time
    # and its normal CLI also requires the separate test split.
    from tools.data_converter.nuscenes_converter import create_nuscenes_infos, export_2d_annotation
    from tools.data_converter.create_gt_database import create_groundtruth_database
    create_nuscenes_infos(str(root), 'nuscenes', version='v1.0-trainval', max_sweeps=10)
    for split in ('train', 'val'):
        path = root / ('nuscenes_infos_' + split + '.pkl')
        export_2d_annotation(str(root), str(path), version='v1.0-trainval')
    create_groundtruth_database('NuScenesDataset', str(root), 'nuscenes', str(root / 'nuscenes_infos_train.pkl'))
    import mmcv
    for split, count in [('train', 28130), ('val', 6019)]:
        actual = len(mmcv.load(str(root / ('nuscenes_infos_' + split + '.pkl')))['infos'])
        if actual != count:
            raise RuntimeError('{} info count {} != {}'.format(split, actual, count))
    print('Prepared 28130 training and 6019 validation infos plus GT database. Training was not started.')


if __name__ == '__main__':
    main()
