# Feasibility timing config: transfusion_nusc_voxel_L.py on nuScenes-mini, batch 1.
# Timing only, not a training recipe. The harness additionally unwraps CBGS and
# drops ObjectSample (its GT database is not built).
_base_ = ['./transfusion_nusc_voxel_L.py']
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(dataset=dict(ann_file='data/nuscenes/nuscenes_mini_infos_train.pkl')),
)
