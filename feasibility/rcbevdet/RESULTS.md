# Feasibility: RCBEVDet

Repo: https://github.com/VDIGPKU/RCBEVDet (cloned 2026-09-17)
Paper: CVPR 2024, plus the RCBEVDet++ extension

## Result

| Field | Value |
|---|---|
| Ran | **not yet — env not built** |
| VRAM @ batch 1 | pending |
| sec/iteration | pending |
| Est. hrs/epoch (mini) | pending |
| Est. hrs/epoch (full nuScenes) | pending |
| Biggest blocker | none identified yet; **the expected access gate does not apply** |

## The code is NOT gated any more

This method was queued to be skipped pending an academic-use request. That is out
of date. In the current README the gating sentence is struck through:

> ~~**Note: please sign the [application](...) to obtain the code** **of RCBEVDet.**~~

and the update log records:

> * 2024/06/01 - Code for RCBEVDet is released in the zip file.

The code ships in-repo as `rcbevdet-master.zip` (1.18 MB), unpacked here to
`feasibility/rcbevdet/code/rcbevdet-master/`. It contains `configs/`, `mmdet3d/`,
`tools/` and `setup.py` — a complete mmdet3d-derived tree. Model weights are on
Google Drive, linked openly in the README.

**No manual step is needed from you.** The `RCBEVDet Application.docx` is still in
the repo but is vestigial.

The fusion config present is:
`configs/rcbevdet/rcbevdet-256x704-r50-BEV128-9kf-depth-cbgs12e-circlelarger.py`
— ResNet-50, 256x704 input, BEV 128, 9 keyframes, depth supervision, CBGS, 12
epochs. Note 9 keyframes and CBGS: like BEVFusion, this recipe leans on temporal
accumulation and class-balanced resampling.

## Requirements (from the bundled README)

Two supported combinations are given:

| | Option A (A800/A40, CUDA 12.1) | Option B (other GPUs, CUDA 11.6) |
|---|---|---|
| Python | 3.8.13 | 3.8.13 |
| CUDA | 12.1 | 11.6 |
| PyTorch | 2.0.1+cu118 | 1.12.1+cu116 |
| torchvision | 0.15.2+cu118 | 0.13.0+cu116 |
| mmcv-full | 1.6.0 | 1.6.2 |
| mmdet | 2.28.2 | 2.24.0 |

Option B is the intended path for a consumer card like the 3080. Both need the
repo's own CUDA ops compiled (`mmdet3d/ops/csrc`) and a `spconv` version matched
to the CUDA build.

## Status and expected obstacles

The environment has not been built yet — BEVFusion and TransFusion were taken
first, and the shared CUDA-toolkit problem (below) had to be solved once before
replicating it.

The known machine-level constraint applies here too: **no CUDA toolkit is
installed** (driver 580.173.02 / CUDA 13.0, no `nvcc` under `/usr/local`), so a
matched toolkit must go into the env before `mmdet3d/ops/csrc` will compile. See
`feasibility/bevfusion/RESULTS.md` for the version-pinning trap encountered there.

## Data

nuScenes is present locally in full, so no download is needed. As with the other
mmdet3d methods, `.pkl` infos must be generated for the mini split first.
