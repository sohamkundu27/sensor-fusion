# Single RTX 3080 plan (configs unchanged)

Reviewed `configs/transfusion_nusc_voxel_L.py`, `configs/transfusion_nusc_voxel_LC.py`,
`configs/nuscenes.md`, `tools/train.py`, and the detector implementation.
The local card has 12,288 MiB VRAM; a 10 GiB card needs more headroom.
No data conversion, training, or full model forward/backward has been run during setup.
Actual peak training memory remains to be measured on real samples.

## Start with LiDAR-only

The upstream recipe trains TransFusion-L for 20 epochs, then initializes a
six-epoch LiDAR-camera stage from pretrained components. The fusion config has
`load_from='checkpoints/fusion_voxel0075_R50.pth'`, but that file is not shipped.
The upstream README says trained models are not released. The fusion stage also
needs suitable image backbone/FPN weights (the upstream notes reference nuImages).
Do not simply clear `load_from`: the image backbone and neck default to frozen
(`freeze_img=True` in the detector), and `freeze_lidar_components=True` freezes
LiDAR modules and selected head components in `tools/train.py`. Freezing random
weights would make a misleading experiment. Train the LiDAR stage first, then
assemble/check compatible LiDAR and image weights before the fusion stage.

## Proposed changes for the first run

| Setting | Upstream | Proposed starting point |
| --- | --- | --- |
| GPUs | 8 | One GPU, `--gpu-ids 0 --launcher none` |
| `data.samples_per_gpu` | 2 (global batch 16) | 1; try 2 only after measuring memory |
| `data.workers_per_gpu` | 6 | 2 initially; tune for CPU/RAM/storage throughput |
| AdamW base LR | 0.0001 | 0.00000625 for batch 1 without accumulation, as a linear-scaling starting heuristic |
| Validation batch | Legacy trainer defaults to 1 | Keep 1; validation allows more voxels than training |
| Precision | FP32 | Establish a real-data FP32 baseline before attempting FP16 |

`tools/train.py` overrides the config's eight `gpu_ids` with one by default.
Its `--autoscale-lr` only multiplies by GPU count / 8; it does **not** account for
changing samples per GPU from 2 to 1. Therefore do not combine that flag with the
explicit LR above. LR scaling is a tuning heuristic, not a guarantee of equivalent
convergence. The cyclic schedule has a 10x peak multiplier, so the proposed base
LR peaks at 0.0000625. Batch 2 without accumulation would suggest base LR 0.0000125.

## Mixed precision and accumulation

The pinned legacy runner has an FP16 optimizer hook and the 3D modules contain
FP16 decorators, so adding an `fp16` section is mechanically possible. This is not
a modern universal autocast switch: the detector explicitly calls `img.float()`,
voxelization forces FP32, and sparse convolution kernels/normalization need dtype
validation. FP16 needs an explicit forward/backward finite-loss check after data
is ready; it is not enabled or claimed validated by this setup. The hook accepts `fp16 = dict(loss_scale=512.0)` or `loss_scale="dynamic"`;
a fixed scale must be a float, not an integer. It makes FP32 master-weight copies
and converts model weights to FP16, so explicit FP32 image inputs are a concrete
dtype mismatch risk that may require a code change before fusion FP16 works.

Gradient accumulation is not a built-in config-only option in MMCV 1.2.4's
optimizer hook. It needs a custom hook (or an independently validated framework
upgrade, which risks these pins). To recover effective batch 16 on one GPU with
one sample, accumulate 16 microbatches (8 at batch 2), divide loss appropriately,
step/clip only at accumulation boundaries, and handle an incomplete final window.
Align the LR/momentum schedule with optimizer updates: legacy hooks normally run
on every iteration. Accumulation does not reproduce batch-16 BatchNorm statistics.
It also does not reduce one sample's activation memory.

## Memory pressure to measure

- 10 sweeps, a 108 m by 108 m point-cloud region, 7.5 cm XY voxels, and sparse
  shape `[41, 1440, 1440]` produce a large sparse backbone workload. Limits are
  120,000 training and 160,000 evaluation voxels per sample, with 10 points each.
- The dense BEV neck produces 512 channels at approximately 180 by 180 cells;
  feature activations and backward buffers are substantial.
- 200 queries attend to BEV features; reducing queries can reduce attention
  memory but changes the experiment and does not remove the backbone cost.
- The fusion branch uses six camera views resized within `(800, 448)`, a ResNet50,
  FPN, and image attention. Keep pretrained frozen branches frozen as intended.
- AdamW maintains optimizer state for trainable parameters. Six data-loader workers
  and class-balanced resampling also create CPU RAM/IO pressure, not just VRAM use.

If batch 1 still exceeds memory, first consider smaller train/eval voxel caps
and fewer sweeps, documenting the accuracy tradeoff. Lower image resolution for
fusion if needed. Coarser voxels or a smaller point range require coordinated
changes to sparse shape, train/test grid sizes, bbox coder settings, and pipelines;
do not change `voxel_size` alone. Image backbone checkpointing can help if that
backbone is unfrozen; checkpointing the custom sparse/attention paths takes code
work. The provided pillar variant is another baseline, but changes architecture.

The paper's fade strategy manually disables object sampling after epoch 15 for
the last five LiDAR epochs; it is not automated by the supplied config. Decide
whether reproducing that strategy is part of the first experiment.
