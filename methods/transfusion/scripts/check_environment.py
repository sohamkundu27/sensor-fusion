#!/usr/bin/env python3
"""Import/version and synthetic CUDA operator checks; no dataset or training."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import torchvision
    import mmcv
    import mmdet
    import mmdet3d
    from mmcv.ops import nms
    from mmdet3d.ops import Voxelization
    from mmdet3d.ops.spconv import SparseConvTensor, SubMConv3d
    from mmdet3d.models import build_detector
    from tools.data_converter import nuscenes_converter
    from tools.data_converter.create_gt_database import create_groundtruth_database
    import numba
    import numpy
    versions = dict(python=sys.version.split()[0], torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
                    mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmdet3d=mmdet3d.__version__, numpy=numpy.__version__, numba=numba.__version__)
    assert versions['torch'] == '1.7.0+cu110', versions
    assert versions['mmcv'] == '1.2.4' and versions['mmdet'] == '2.10.0', versions
    assert versions['mmdet3d'] == '0.11.0', versions
    assert torch.cuda.is_available(), 'CUDA unavailable'
    versions['gpu'] = torch.cuda.get_device_name(0)
    x = torch.randn(32, 32, device='cuda')
    assert torch.isfinite(x @ x).all()
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device='cuda')
    scores = torch.tensor([0.9, 0.8], device='cuda')
    _, keep = nms(boxes, scores, 0.5)
    assert keep.numel() == 1
    voxel = Voxelization(voxel_size=[0.5, 0.5, 0.5], point_cloud_range=[0, 0, 0, 4, 4, 4], max_num_points=5, max_voxels=100)
    points = torch.tensor([[1., 1., 1., 0.5, 0.], [1.1, 1.1, 1.1, 0.6, 0.]], device='cuda')
    voxels, coords, counts = voxel(points)
    assert counts.sum().item() == 2
    indices = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2], [0, 2, 2, 2]], device='cuda', dtype=torch.int32)
    features = torch.randn(3, 4, device='cuda', requires_grad=True)
    sparse = SparseConvTensor(features, indices, [4, 4, 4], 1)
    layer = SubMConv3d(4, 8, 3, padding=1, bias=False).cuda()
    output = layer(sparse).features
    assert output.shape == (3, 8) and torch.isfinite(output).all()
    output.sum().backward()  # A tiny operator gradient check, no model training/optimizer.
    assert features.grad is not None and torch.isfinite(features.grad).all()
    torch.cuda.synchronize()
    for filename in ('transfusion_nusc_voxel_L.py', 'transfusion_nusc_voxel_LC.py'):
        cfg = mmcv.Config.fromfile(str(ROOT / 'configs' / filename))
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        assert type(model).__name__ == 'TransFusionDetector'
        del model
    print(json.dumps(versions, indent=2))
    print('PASS: imports, both model constructions, CUDA matmul/NMS/voxelization, bundled spconv forward and gradient.')
    print('No dataset was loaded, no full model forward/backward or training was run.')


if __name__ == '__main__':
    main()
