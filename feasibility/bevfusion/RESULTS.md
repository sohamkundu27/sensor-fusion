# Feasibility: BEVFusion (MIT HAN Lab)

Repo: https://github.com/mit-han-lab/bevfusion, HEAD `326653d` ("[ROCm] add support
for ROCm/HIP"). Not the archived no-radar tree the plan assumed: this HEAD ships radar
encoders and camera+radar configs. Env: `feasibility/bevfusion/env2` (micromamba, Py 3.8).
Config timed: `camera+lidar/swint_v0p075/convfuser.yaml` (Swin-T camera + VoxelNet
0.075 m LiDAR, ConvFuser, TransFusion head), via `patches/bench_mini.yaml`.

## Result (measured 2026-09-18 on an idle RTX 3080)

| Field | Value |
|---|---|
| Ran | **Y — after 6 source/config fixes; does NOT run as shipped on a 12 GB card** |
| VRAM @ batch 1 | **6.11 GB peak allocated** (6.74 GB reserved by PyTorch; **9.17 GB of 11.63 GB used device-wide**) |
| sec/iteration | **0.427 s** wall-clock median of 30 (compute-only 0.424 s; data wait 0.7%) |
| Est. hrs/epoch (mini) | **0.038 h** (~2.3 min; 323 keyframes) |
| Est. hrs/epoch (full nuScenes) | **3.34 h** (28,130 keyframes) |
| Biggest blocker | Bundled spconv reserves **6.47 GB** of driver local memory on first launch, which with the model exceeds 12 GB. Fixed by one constant. |

Run-to-run: two clean runs gave 0.4237 and 0.4272 s/iter (<1% apart).

### Why it does not fit as shipped

The first sparse convolution fails with `indice_cuda.cu 124: cuda execution failed
with error 2` (cudaErrorMemoryAllocation) at only 5.02 GB allocated. That line is a
launch check, not an allocation. `getSubMIndicePairsKernel` and two sibling kernels
are instantiated with `KernelMaxVolume = 4096`, giving each thread a local array
`validPoints[4096 * (NDim + 1)]` = 64 KB. The driver reserves local memory for every
resident thread: 70 SMs x 1536 threads x 64 KB ~= 6.6 GB. Measured directly in a
fresh process: **6.47 GB disappears outside PyTorch's allocator after one launch**.
6.5 GB + ~2 GB CUDA context + ~5-6 GB model exceeds the 11.6 GB available.

The header's own default is `KernelMaxVolume = 256`; BEVFusion's call sites override
it to 4096 although its sparse encoder only uses 3x3x3 (27) and 1x1x3 (3) kernels.
Resetting the three call sites to 256 drops the reservation to **0.31 GB**. The
kernel does not bounds-check volume against the template, so this is safe only while
no kernel exceeds 256 — true for this config. On a 24 GB card the original fits,
which is probably why nobody upstream noticed.

### Timing caveats

* LiDAR and camera branches are randomly initialised except the ImageNet Swin-T. The
  published recipe fine-tunes a frozen, pretrained LiDAR branch for 6 epochs; with it
  frozen, backward is cheaper, so 3.34 h/epoch is a conservative upper bound for the
  fusion stage.
* Plain dataset, not CBGS. CBGS resampling makes an upstream "epoch" several times
  longer than 28,130 iterations; multiply accordingly to compare with the paper.
* `ObjectPaste` (GT-sampling) dropped: its database could not be built (see below).
  It is CPU-side; data wait is 0.7%, so its absence barely affects these numbers.

## Everything changed to get it running

All source changes are in `patches/bevfusion-fixes.patch`; the config is
`patches/bench_mini.yaml`.

1. `setup.py` never registered `feature_decorator_ext` though its sources ship.
2. `feature_decorator.cpp` registers a torch op with `int` params; torch accepts only
   `int64_t`/`bool` (static assertion).
3. `radar_encoder.py` imports `flash_attn` unconditionally, breaking every backbone
   import although this config never uses the radar encoder. Made optional.
4. `base.py` was modernised (passes `mats_dict`; `add_depth_features` default True
   -> 6-channel depth) but `DepthLSSTransform` was left behind: wrong
   `get_cam_feats` arity, `__init__` did not forward new kwargs, and `dtransform`
   still hardcodes `Conv2d(1, 8, 1)`. Accepted `mats_dict`, forwarded kwargs, and
   pinned scalar single-channel depth in the bench config.
5. **spconv `KernelMaxVolume` 4096 -> 256** at three call sites (above).
6. `tools/data_converter/nuscenes_converter.py:95` joins the output path with
   `info_prefix` instead of `root_path`, so infos land in `./nuscenes_mini/` with a
   `_radar` suffix and GT-database creation then fails. Infos copied to the names
   the config expects; GT database not built.

My own harness mistake, fixed: the bench YAML initially sat beside
`convfuser.yaml`, but torchpack only inherits parent-directory `default.yaml`
files, so the fuser was silently missing (`assert len(features) == 1`).

## Environment: built successfully

Verified import line:

```
OK torch 1.10.0+cu113 mmcv 1.4.0 mmdet 2.20.0 cuda True
```

Build recipe (`build_env.sh`), all pins from the upstream README:

| Component | Version |
|---|---|
| Python | 3.8 (README asks >=3.8, <3.9) |
| PyTorch | 1.10.0+cu113 (README asks >=1.9, <=1.10.2) |
| torchvision | 0.11.1+cu113 |
| mmcv-full | 1.4.0, from the `cu113/torch1.10.0` OpenMMLab wheel index |
| mmdet | 2.20.0 |
| numpy / numba | 1.23.5 / 0.48.0 |
| torchpack | latest |

## Install issues encountered

1. **No system Python in the supported range.** This machine has only Python
   3.12.3; upstream requires >=3.8,<3.9. Resolved with a per-method micromamba
   env rather than touching the system interpreter.

2. **CUDA 11.3 wheels on a CUDA 13.0 driver.** The box runs driver 580.173.02 /
   CUDA 13.0, while BEVFusion's whole stack is built for cu113. This was the main
   risk to the entire pass and it is **resolved**: `torch.cuda.is_available()` is
   True and the arch list includes `sm_86`, which is the RTX 3080's compute
   capability. Driver backward-compatibility carries the old runtime fine. This
   de-risks TransFusion and SAMFusion, which pin similar stacks.

3. **mmcv-full 1.4.0 needs the version-matched OpenMMLab wheel index.** A plain
   `pip install mmcv-full==1.4.0` attempts a lengthy source build; pointing `-f`
   at `download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html` gets a
   prebuilt wheel. Recorded because the README does not spell this out.

4. CUDA ops: no CUDA toolkit exists on this machine (driver only), so env2 installs a
   self-consistent CUDA 11.3 from the nvidia label first and pins
   `sysroot_linux-64=2.17` (conda's kernel-headers 6.12 define `__s128`, which nvcc
   11.3 cannot parse). Single-GPU timing calls the harness directly, not
   `torchpack dist-run`, so OpenMPI/mpi4py were not needed.

## Data

nuScenes is present locally in full (`/home/soham/data/nuscenes`, 53 GB samples +
342 GB sweeps) along with `v1.0-mini` metadata (404 samples), so no download is
required. mmdet3d-style `.pkl` infos were generated for the mini split (323 train / 81 val
keyframes) — see fix 6 above for the converter's path bug.

## Notes

No radar and no reliability logic in the base config, as expected — this is the
mid-level BEV speed reference for the comparison, not a candidate method.
