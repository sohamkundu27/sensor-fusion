# Feasibility: TransFusion

Repo: https://github.com/XuyangBai/TransFusion (cloned 2026-09-17, shallow)
Env: `feasibility/transfusion/env` (micromamba, Python 3.8)

Treated as a **speed reference only**, not a candidate: no radar in the base
architecture.

## Result

| Field | Value |
|---|---|
| Ran | **environment builds; timing pending** |
| VRAM @ batch 1 | pending |
| sec/iteration | pending |
| Est. hrs/epoch (mini) | pending |
| Est. hrs/epoch (full nuScenes) | pending |
| Biggest blocker | none fatal yet; CUDA ops not yet compiled |

## Environment: built successfully

```
OK torch 1.10.0+cu113 mmcv 1.4.0 mmdet 2.11.0 cuda True
```

Upstream compatibility table (`docs/getting_started.md`) allows
`mmdet>=2.5.0` with `mmcv-full>=1.2.4,<=1.4`; the README says the authors used
mmdet 2.10.0 / mmcv 1.2.4. The combination installed here sits inside the
supported range.

## Install issues encountered

1. **`mmpycocotools` fails to build.** `pip install mmdet==2.11.0` pulls
   `mmpycocotools`, whose sdist ships `pycocotools/_mask.pyx` without the
   generated C source:

   ```
   cc1: fatal error: pycocotools/_mask.c: No such file or directory
   error: command '/usr/bin/gcc' failed with exit code 1
   ERROR: Failed building wheel for mmpycocotools
   ```

   Fixed by installing `cython==0.29.36` and `numpy==1.23.5` *before* mmdet, so
   Cython can generate `_mask.c`. Logged rather than worked around silently —
   this bites any fresh install of the mmdet 2.1x line.

2. Same Python-version and CUDA-toolkit constraints as BEVFusion (see that
   method's RESULTS.md): no system Python in range, and **no CUDA toolkit on this
   machine at all** — driver 580.173.02 only, no `nvcc` anywhere under
   `/usr/local`. A matched toolkit has to be installed per-env before any CUDA
   extension will compile.

## Sanity check against the authors' reported training cost

The authors report roughly **2 days on 8x V100 or 8x RTX 3090** for full
training. Taking 8 GPUs x 48 h = ~384 GPU-hours, a single RTX 3080 at roughly
0.7-0.8x a 3090 would need on the order of **500-550 GPU-hours, i.e. about
3 weeks continuous**, for the same schedule. The measured sec/iter will be checked
against this once the GPU frees; if the extrapolated full-nuScenes epoch time
implies wildly less than that, the measurement is wrong.

## Data

nuScenes present locally in full. TransFusion needs mmdet3d-format `.pkl` infos.
Note the README's warning: generating those infos with a *newer* mmdet3d gives
wrong mAOE/mASE because of the mmdet3d coordinate-system refactor. Irrelevant for
timing, but it would matter for any accuracy work later.
