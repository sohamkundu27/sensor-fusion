"""Run RCBEVDet's own nuScenes RC converter on v1.0-mini.
tools/create_data_nuscenes_RC.py hard-codes v1.0-trainval and also processes
v1.0-test (not present here), so its functions are called directly with the
mini version instead of editing the script. add_ann_adj_info's body is copied
verbatim with only the version string changed."""
import sys, pickle, numpy as np, mmcv
sys.path.insert(0, '.'); sys.path.insert(0, './tools')
# The released zip lists NuScenesDataset_R in mmdet3d.datasets.__all__ but never
# defines it; the converter only needs its NameMapping. Alias the class that
# does exist rather than editing upstream code.
import mmdet3d.datasets as _mds
if not hasattr(_mds, 'NuScenesDataset_R'):
    _mds.NuScenesDataset_R = _mds.NuScenesDatasetRC
import create_data_nuscenes_RC as rc
from nuscenes import NuScenes
TAG = 'nuscenes_RC_mini'
rc.nuscenes_data_prep(root_path='./data/nuscenes', info_prefix=TAG,
                      version='v1.0-mini', max_sweeps=0)
nus = NuScenes('v1.0-mini', './data/nuscenes/')
for split in ['train', 'val']:
    path = './data/nuscenes/%s_infos_%s.pkl' % (TAG, split)
    d = pickle.load(open(path, 'rb'))
    for i in mmcv.track_iter_progress(range(len(d['infos']))):
        info = d['infos'][i]; sample = nus.get('sample', info['token']); anns = []
        for a in sample['anns']:
            ai = nus.get('sample_annotation', a); v = nus.box_velocity(ai['token'])
            ai['velocity'] = np.zeros(3) if np.any(np.isnan(v)) else v; anns.append(ai)
        info['ann_infos'] = anns; info['ann_infos'] = rc.get_gt(info); info['scene_token'] = sample['scene_token']
    pickle.dump(d, open(path, 'wb')); print(split, len(d['infos']))
