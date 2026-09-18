# Feasibility: SAMFusion

Repo: https://github.com/torc-ai/SAMFusion (single commit `4a39d0f`, Apache-2.0)
Paper: ECCV 2024. Env: `feasibility/samfusion/env2` (micromamba, Python 3.8, CUDA 11.3 in-env)
Config timed: `projects/SAMFusion/configs/nuscenes/SAMFusion.py` (camera + LiDAR +
radar; image and LiDAR branches frozen per the config) with the mmengine harness
`feasibility/bench_mmengine.py`.

## Result (measured 2026-09-18 on an idle RTX 3080)

| Field | Value |
|---|---|
| Ran | **Y — at the edge of the card** |
| VRAM @ batch 1 | **7.36 GB peak allocated** (9.45 GB reserved; **11.59 of 11.63 GB used device-wide**, ~40 MB free) |
| sec/iteration | **0.711 s** wall-clock median of 30 (compute-only 0.708 s; data wait 0.4%) |
| Est. hrs/epoch (mini) | **0.064 h** (~3.8 min; 323 keyframes) |
| Est. hrs/epoch (full nuScenes) | **5.56 h** (28,130 keyframes) |
| Biggest blocker | The nuScenes training init checkpoint (`Fusion_0075_refactor.pth`, a DeepInteraction model) is not published. |

Slowest and heaviest of the runnable methods. It fits at batch 1 only with
`freeze_img=True, freeze_pts=True` as configured; batch 2, or training the
DeepInteraction stage it depends on, would not fit on this card. The 2.1 GB gap between
reserved and allocated is caching-allocator slack, so `PYTORCH_CUDA_ALLOC_CONF`
tuning might buy some room — not attempted.

### The checkpoint blocker, precisely

The config sets `load_from = 'projects/SAMFusion/pretrained/Fusion_0075_refactor.pth'`.
The README's published weights are, for nuScenes, only the *finished*
`samfusion_nuscenes_epoch_6.pth` (457 MB, reachable on S3), and for the Dense
dataset `deepinteraction_dense_epoch_6.pth`. There is no nuScenes DeepInteraction
init. Timing is unaffected (the frozen branches cost the same with any weights), but
reproducing SAMFusion training on nuScenes needs either that file from the authors or
training DeepInteraction first — which would not fit at batch 1 here, since it trains
the branches SAMFusion freezes.

## Everything changed to get it running

In `patches/samfusion-fixes.patch`:

1. **`deepinteraction.py`: passes `bbox_head=` to mmdet3d's `MVXTwoStageDetector`,
   which accepts no such argument in mmdet3d 1.2.0 *or* current `main`** (checked
   both). The README pip-installs mmdet3d 1.2.0 then clones mmdetection3d unpinned and
   installs over it, so SAMFusion was evidently built against an unpublished fork.
   With this config `bbox_head` is `None` and the code already falls back to
   `pts_bbox_head`; the fix stops forwarding it and builds it locally only if given.
2. **`tools/dataset_converters/update_infos_to_v2.py:301` hard-codes the authors'
   dataset path** (`/mnt/perception-aisee/Pubblic_Datasets/nuScenes/...`). Replaced
   with `out_dir`, which is what upstream mmdet3d uses there.

Harness-side: CBGS unwrapped (one epoch = 323 / 28,130 keyframes); train `ann_file`
pointed at the mini infos.

## Install issues encountered

1. **`python=3.7.11` is unsolvable** on current conda-forge; used 3.8.
2. Pins turned out consistent: `mmcv 2.0.1` *is* published for cu113/torch 1.10.0.
3. **`flash-attn==0.2.2` is pinned but never imported** by the project code — skipped.
4. **Pillow 10 breaks the old detectron2 wheel** (`Image.LINEAR` removed); a later
   install upgraded Pillow past the pin. Re-pinned `Pillow==9.5.0`.
5. **The plugin JIT-compiles a C++ extension on import**, so `ninja`, `CUDA_HOME`
   and the conda compilers must be on `PATH` at runtime, not only at build time
   (`RuntimeError: Ninja is required to load C++ extensions`).
6. Same CUDA-toolkit / `sysroot_linux-64=2.17` recipe as the other methods.
7. Verified: `mmdet3d 1.2.0, mmcv 2.0.1, mmdet 3.1.0, spconv 2.3.6`, plugin imports.

## Data

`tools/create_data.py --version v1.0-mini --extra-tag nuscenes_mini` produced
`nuscenes_mini_radar_mmdet_v2_infos_{train,val}.pkl` (323 train keyframes) after fix 2.
