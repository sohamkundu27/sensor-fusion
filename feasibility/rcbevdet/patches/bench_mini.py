# Feasibility timing config: the released RCBEVDet config on nuScenes-mini, batch 1.
# Timing only. The harness additionally unwraps CBGS (one epoch = true keyframe count).
_base_ = ['./rcbevdet-256x704-r50-BEV128-9kf-depth-cbgs12e-circlelarger.py']
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(dataset=dict(ann_file='data/nuscenes/nuscenes_RC_mini_infos_train.pkl')),
)
