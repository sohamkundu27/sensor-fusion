"""Check all trainval keyframe sensor paths and decode representative full-data batches."""
import json
from pathlib import Path
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from data import NuScenesFusionDataset, CAMERAS, collate_fusion_batch


def main():
    root=Path.home()/'data/nuscenes'
    nusc=NuScenes(version='v1.0-trainval',dataroot=str(root),verbose=True)
    assert len(nusc.scene)==850 and len(nusc.sample)==34149
    checked=0
    for sample in nusc.sample:
        assert len(sample['data'])==12
        for channel,token in sample['data'].items():
            record=nusc.get('sample_data',token)
            path=root/record['filename']
            size=path.stat().st_size
            assert size>0, str(path)
            if channel=='LIDAR_TOP':
                assert size%20==0, str(path)
            checked+=1
    counts={}
    for split in ('train','val'):
        ds=NuScenesFusionDataset(root,version='v1.0-trainval',split=split,cameras=CAMERAS,nusc=nusc)
        for index in np.linspace(0,len(ds)-1,24,dtype=int):
            batch=collate_fusion_batch([ds[int(index)]])
            for key in ('camera','lidar','radar'):
                assert torch.isfinite(batch[key]).all()
        counts[split]=len(ds)
    report=dict(status='PASS',keyframe_sensor_files=checked,items=counts,decoded_views=48,
                limitation='Presence/nonzero size and representative decode checks; archive MD5 discrepancies remain unresolved.')
    out=Path('outputs/full_data_check.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
