# Initial precheck — September 13, 2026

The isolated entropy-fusion environment passed `pip check`, and all four geometry
unit tests passed. Real nuScenes-mini training input passed at batch size two
(front camera, 640 x 384), and validation input passed at batch size six covering
all six camera views of one sample. Both checks used two loader workers and
successfully transferred inputs to CUDA. Train/validation scene sets are disjoint.

All checked views contained projected LiDAR and radar returns and valid projected
box targets. Independent devkit projection comparisons on the first item of each
batch passed the configured 0.05-pixel / 0.002-metre tolerances. The largest reported
LiDAR projection difference was approximately 0.022 pixels. These are sampled
pipeline checks, not exhaustive validation of every mini or full-data record.

Reports and overlays are in `outputs/mini_batch/` and `outputs/mini_val_batch/`.

A training benchmark cannot run yet: backbone, entropy, fusion and detection-head
modules remain docstring-only placeholders; training and evaluation entry points
are unimplemented. No forward/backward steps, model memory measurements, training
throughput measurements or accuracy evaluation were performed. Input transfer
success must not be interpreted as a model fitting in GPU memory.

To obtain a comparison with TransFusion, first implement the model, anchor matching,
losses and modality dropout, then run a bounded finite-loss/gradient benchmark.
The intended standard nuScenes 3D evaluation also requires 3D predictions and a
global-coordinate export adapter; a plain image-plane SSD head is insufficient.
