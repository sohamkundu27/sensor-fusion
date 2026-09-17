# Feasibility: BEVFusion (MIT HAN Lab)

Repo: https://github.com/mit-han-lab/bevfusion (archived/read-only, cloned 2026-09-17)
Env: `feasibility/bevfusion/env` (micromamba, Python 3.8)

## Result

| Field | Value |
|---|---|
| Ran | **environment builds; timing pending** |
| VRAM @ batch 1 | pending (queued behind the gating ablation) |
| sec/iteration | pending |
| Est. hrs/epoch (mini) | pending |
| Est. hrs/epoch (full nuScenes) | pending |
| Biggest blocker | none fatal so far |

Timing is deliberately deferred: the RTX 3080 is still occupied by the entropy
gating ablation until ~2026-09-18 14:48 UTC. Measuring now would give a
contended, non-comparable sec/iter. `feasibility/run_when_free.sh` starts the
benchmark automatically once the GPU reports zero compute processes.

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

4. Not yet exercised: `python setup.py develop` for the repo's own CUDA ops
   (including `mmdet3d/ops/bev_pool`), and OpenMPI 4.0.4 + mpi4py 3.0.3, which
   upstream needs for `torchpack dist-run`. Single-GPU timing will call
   `tools/train.py` directly rather than through `torchpack`, avoiding the MPI
   dependency for this pass.

## Data

nuScenes is present locally in full (`/home/soham/data/nuscenes`, 53 GB samples +
342 GB sweeps) along with `v1.0-mini` metadata (404 samples), so no download is
required. BEVFusion expects mmdet3d-style `.pkl` infos, which still need to be
generated for the mini split before the first iteration will run.

## Notes

No radar and no reliability logic in the base config, as expected — this is the
mid-level BEV speed reference for the comparison, not a candidate method.
