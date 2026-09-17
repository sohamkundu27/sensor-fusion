# Feasibility: RAF (Reliability-Aware Fusion)

Repo: https://github.com/parkie0517/RAF (cloned 2026-09-17, shallow)
Paper: Park, Jeong, Yoon — ECCV 2026 (to appear)

## Result

| Field | Value |
|---|---|
| Ran | **N** |
| VRAM @ batch 1 | not measured |
| sec/iteration | not measured |
| Est. hrs/epoch (mini) | not measured |
| Est. hrs/epoch (full nuScenes) | not applicable (method does not use nuScenes) |
| Biggest blocker | **The frozen pretrained LiDAR-RADAR backbone is not distributed** — configs point at the authors' own local absolute paths, with no download URL. |

Unlike method 1, this repo *does* contain real training code (`main_train.py`,
`models/`, `ops/` with CUDA extensions). It is blocked on inputs, not on missing
code. Three independent blockers, in order of how hard they are to clear.

## Blocker 1 (decisive): no pretrained LiDAR-RADAR weights

RAF's whole design is a frozen pretrained L4DR / 3D-LRF backbone plus a trained
camera branch, BEV fusion encoder and head. The checkpoint is referenced only as
author-local absolute paths that were never published:

```
configs/L4DR_stage2_uem_cnn_freeze.yml:212
  PRETRAINED:
    PATH: '/home/user/heejun/L4DR/logs/Simple_Voxel_Encoder/train_simple_voxel_encoder/models/model_8.pt'

configs/RAF_multi_class.yml:212
  PRETRAINED:
    PATH: './logs/Simple_Voxel_Encoder_multi_class/simple_voxel_encoder_multi_class/models/model_28.pt'

configs/RAF_multi_class.yml:19
  PATH_EXP: '/mnt/32THHD/hx/K-Radar-main/logs/PP_RLF/train_35e_0.1denoise_nofilter_nor2l_nomax'
```

There is no release URL, no `download.sh`, and no model-zoo table in the README.
The only downloadable pretrained weight anywhere in the configs is the ImageNet
Swin-Tiny backbone for the *camera* branch
(`swin_tiny_patch4_window7_224.pth`), which is not the LiDAR-RADAR backbone.

Per your instruction I stopped here rather than training an L4DR backbone from
scratch to unblock it.

## Blocker 2: no VoD support in the released code, and no local data

You suggested a small VoD slice instead of nuScenes. **The released code has no
VoD loader.** `datasets/` contains K-Radar loaders only:

```
kradar_detection_v1_0.py  kradar_detection_v2_0.py  kradar_detection_v2_2.py
kradar_detection_v1_1.py  kradar_detection_v2_1.py  kradar_detection_v2_3.py
kradar_detection_v2_2_1.py kradar_detection_v2_4.py kradar_tracking_v1_0.py
```

The active config selects `NAME: 'KRadarDetection_v2_2'`. The paper reports VoD
results, but that pipeline is not in this repository — using VoD would mean
writing a loader, which is outside a feasibility pass.

Neither dataset is on this machine (`/home/soham/data/` holds kitti, nuscenes,
waymo, seeing_through_fog only). Both are application-gated: K-Radar requires a
request to KAIST AVELab and is multi-TB; VoD requires an academic request form
to TU Delft.

## Blocker 3: dependency set is Python 3.8-3.10 era; this machine has only 3.12

A venv was created at `feasibility/raf/venv` (Python 3.12.3) and
`pip install -r repo/requirements.txt` fails immediately. Probing each pin:

| Requirement | Result on Python 3.12 |
|---|---|
| `open3d==0.15.2` | **Unavailable.** Only 0.19.0, 0.20.0 are published for 3.12 |
| `opencv-python==4.2.0.32` | **Unavailable.** Oldest published for 3.12 is 3.4.0.14 / 4.3.0.38 |
| `numba==0.55.1` | **Unavailable.** Requires `>=3.7,<3.11`; first 3.12-compatible is 0.59.x |
| `spconv-cu113` | **Unavailable — no distribution at all.** Built for CUDA 11.3; this box is CUDA 13.0 / driver 580.173.02 |
| `nms` | installs (0.1.6) |
| `setuptools==59.5.0` | installs |

`spconv-cu113` is the hard one: sparse convolution is central to the LiDAR
backbone, the cu113 build line is not published for this CUDA, and the version
that matches CUDA 13 (`spconv-cu120`+) is not ABI-compatible with the pinned
`numba`/`torch` combination this code expects.

Additionally `main_train.py` imports the OpenMMLab stack (`mmcv`, `mmdet`,
`mmdet3d` — it installs log filters for all three), and `ops/setup.py` builds
six CUDA extensions (`bev_pool`, `ingroup_inds`, `iou3d_nms`, `pointnet2`,
`roiaware_pool3d`, `roipoint_pool3d`) that must compile against the same torch.
Per your instruction I logged this rather than working around it; resolving it
means a Python 3.8 interpreter and an older CUDA toolchain, not a pin bump.

## To unblock

1. Obtain the L4DR / 3D-LRF pretrained checkpoint from the authors directly
   (manual — email, not scriptable).
2. Apply for K-Radar access, or write a VoD loader that the repo does not ship.
3. Install Python 3.8-3.10 (deadsnakes or conda) and an older CUDA toolchain for
   `spconv` and the six CUDA ops.

Item 1 alone makes this method un-timeable in this pass.
