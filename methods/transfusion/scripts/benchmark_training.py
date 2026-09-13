#!/usr/bin/env python3
"""Bounded FP32 mini training benchmark; creates separate mini prep artifacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
import random
import numpy as np
import torch
import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_detector
from tools.data_converter.nuscenes_converter import create_nuscenes_infos
from tools.data_converter.create_gt_database import create_groundtruth_database


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    root = 'data/nuscenes'
    prefix = 'benchmark_mini'
    info = root + '/' + prefix + '_infos_train.pkl'
    dbinfo = root + '/' + prefix + '_dbinfos_train.pkl'
    out = Path('work_dirs/mini_precheck')
    out.mkdir(parents=True, exist_ok=True)
    if not Path(info).exists():
        create_nuscenes_infos(root, prefix, version='v1.0-mini', max_sweeps=10)
    if not Path(dbinfo).exists():
        create_groundtruth_database('NuScenesDataset', root, prefix, info)
    cfg = mmcv.Config.fromfile('configs/transfusion_nusc_voxel_L.py')
    ds = cfg.data.train.dataset.copy()
    ds.ann_file = info
    for step in ds.pipeline:
        if step.type == 'ObjectSample':
            step.db_sampler.info_path = dbinfo
            step.db_sampler.data_root = root
    dataset = build_dataset(ds)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                              num_gpus=1, dist=False, shuffle=True, seed=42)
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.train()
    optimizer_cfg = cfg.optimizer.copy()
    optimizer_cfg.lr = 0.00000625
    optimizer = build_optimizer(model, optimizer_cfg)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    for i in range(12):
        start = time.perf_counter()
        batch = next(iterator)
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        optimizer.zero_grad()
        result = model.train_step(batch, optimizer)
        loss = result['loss']
        if not torch.isfinite(loss).all():
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        row = dict(iteration=i+1, loss=float(loss.detach()), gradient_norm=float(norm),
                   data_seconds=loaded-start, compute_seconds=end-loaded, seconds=end-start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out/'report.json').write_text(json.dumps(dict(settings=dict(batch_size=1, precision='fp32', sweeps=10,
            object_sampling=True, class_balanced_wrapper=False, dataset='v1.0-mini', iterations=12), iterations=rows), indent=2))
    print('PASS: 12 finite forward/backward/AdamW steps. Benchmark weights discarded.', flush=True)

if __name__ == '__main__':
    main()
