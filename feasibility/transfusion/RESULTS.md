# Feasibility: TransFusion

Repo: https://github.com/XuyangBai/TransFusion (shallow clone, 2026-09-17)
Env: `feasibility/transfusion/env2` (micromamba, Python 3.8, CUDA 11.3 toolkit in-env)
Config timed: `configs/transfusion_nusc_voxel_L.py` (TransFusion-L, VoxelNet 0.075 m,
10 sweeps) via `patches/bench_mini_voxel_L.py`. Speed reference only: no radar.

## Result (measured 2026-09-18 on an idle RTX 3080)

| Field | Value |
|---|---|
| Ran | **Y — as shipped (after one compile fix)** |
| VRAM @ batch 1 | **2.56 GB peak allocated** (2.99 GB reserved) — but **11.54 of 11.63 GB used device-wide** as shipped |
| sec/iteration | **0.342 s** wall-clock median of 30 (compute-only 0.341 s; data wait 0.4%) |
| Est. hrs/epoch (mini) | **0.031 h** (~1.8 min; 323 keyframes) |
| Est. hrs/epoch (full nuScenes) | **2.67 h** (28,130 keyframes) |
| Biggest blocker | None fatal. Same spconv local-memory reservation as BEVFusion leaves only ~0.1 GB headroom as shipped. |

Three clean runs: 0.3438, 0.3423 (as shipped) and 0.3419 s/iter (patched spconv).

### Memory: the model is small, the sparse-conv kernel is not

TransFusion bundles the same old spconv as BEVFusion, with index-pair kernels
instantiated at `KernelMaxVolume = 4096` (64 KB of local memory per thread, reserved
by the driver for every resident thread). PyTorch itself needs only 2.56 GB, yet the
device shows 11.54 GB used. Setting the three call sites back to the header default
of 256 (as for BEVFusion) gives:

| Build | sec/iter | torch allocated | device used |
|---|---:|---:|---:|
| As shipped (4096) | 0.342 | 2.56 GB | **11.54 GB** |
| Patched (256) | 0.342 | 2.56 GB | **5.39 GB** |

Identical speed, 6.15 GB less memory. As shipped it fits at batch 1 only because the
model is small; any growth (batch 2, camera branch) would overflow the card.

### Sanity check against the authors' training cost

Authors report ~2 days on 8x V100 or 8x RTX 3090 (~384 GPU-hours). The published
TransFusion-L schedule is 20 epochs with CBGS resampling, which makes an epoch
roughly 4.5x the 28,130 keyframes. At 0.342 s/iter on one 3080 that is about
20 x 4.5 x 28,130 x 0.342 s ≈ **240 h ≈ 10 days** at batch 1. Same order as the
authors' figure on faster cards; everything omitted here (GT-sampling augmentation,
multi-GPU sync, the camera stage) would make real training slower, not faster. No
sign the measurement is off.

## Everything changed to get it running

In `patches/transfusion-fixes.patch`:

1. **`scatter_points_cuda.cu:272`**: `coors_map.index_put_(coors_id_argsort, ...)`
   does not compile on torch >= ~1.7 (`no instance of overloaded function
   "at::Tensor::index_put_"`). Wrapped the index in a list, as upstream mmdet3d did.
2. **spconv `KernelMaxVolume` 4096 → 256** — only for the patched row above; the
   headline figures are as shipped.

Harness-side (no repo edits): mmdet3d 0.11 exposes `build_detector`, not
`build_model`, and `build_dataloader` lives in mmdet; the harness falls back.
`ObjectSample` (GT-sampling) dropped because its database was not built; CPU-side,
data wait is 0.4%. CBGS unwrapped so one epoch = 323 / 28,130 keyframes.

## Install issues encountered

1. **`mmpycocotools` fails to build** under `mmdet==2.11.0`: its sdist ships
   `_mask.pyx` without the generated `_mask.c`. Install `cython==0.29.36` and
   `numpy==1.23.5` first.
2. **Missing runtime deps**: `lyft_dataset_sdk` (imported unconditionally by
   `mmdet3d/core/evaluation`), plus `networkx<2.3`, `plyfile`, `scikit-image`,
   `trimesh==2.35.39`. mmdet3d 0.11 declares `numpy<1.20`; 1.23.5 was kept and works
   for this pass (pip warns).
3. **No CUDA toolkit on the machine** — solved as for BEVFusion: env created with the
   nvidia `cuda-11.3.1` label first (CUDART_VERSION 11030) and
   `sysroot_linux-64=2.17` pinned (kernel-headers 6.12 define `__s128`, which nvcc
   11.3 cannot parse). The first env, built without this, was discarded.
4. Verified import line: `torch 1.10.0+cu113, mmcv 1.4.0, mmdet 2.11.0, cuda True`.

## Not measured

`transfusion_nusc_voxel_LC.py` (camera + LiDAR) was not timed. TransFusion-L is the
LiDAR branch BEVFusion builds on, so it serves the "speed reference" role; LC would
add an image backbone and needs a pretrained image checkpoint.

## Data

Reuses the nuScenes-mini infos generated for BEVFusion (same mmdet3d 0.x format).
The README warns that infos from a newer mmdet3d give wrong mAOE/mASE because of the
coordinate refactor — irrelevant to timing, relevant to any later accuracy work.
