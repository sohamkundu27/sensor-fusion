# Fusion-method results — consolidated for slideshow update

Compiled 2026-09-18 ~19:15 UTC from files on disk only. Nothing was re-run, fixed or
optimized for this report.

**Read this first:** Part 1 is complete. **Part 2 is incomplete** — the gate-off arm of the
ablation is still training (resumed 2026-09-18 18:25 UTC after a crash). Gate-off has results
for epochs 1, 5 and 10 only. There is no epoch-20 result for gate-off and no combined wall-clock
yet. Those gaps are marked below, not estimated.

**Precision note:** values are reproduced exactly as logged. The feasibility harness rounds when it
writes `result.json` (seconds to 4 decimals, memory and mini-epoch hours to 3, full-epoch hours to
2). No per-iteration timings are stored, so those rounded values are the most precise that exist.
Ablation mAP/NDS are full float precision from the evaluator.

**Units:** every "GB" figure in the feasibility files is computed as bytes / 1024³, i.e. **GiB**.
The card reports 11.631 GiB usable to CUDA (nvidia-smi lists 12,288 MiB).

---

# PART 1 — Feasibility triage (7 methods)

Sources: `feasibility/<method>/RESULTS.md` (read in full), `feasibility/<method>/result.json`,
`feasibility/SUMMARY.md`.

Setup for all measured rows: single RTX 3080 (driver 580.173.02), batch size 1, 5 warmup +
30 timed training iterations (forward + backward + optimizer step), nuScenes-mini (323 train
keyframes), otherwise idle GPU, measured 2026-09-18. sec/iter = wall-clock median including
data loading. Epoch hours = sec/iter × 323 (mini) or × 28,130 (full nuScenes), at batch 1,
**without CBGS** resampling (all four runnable methods use CBGS upstream, which makes a
published epoch roughly 4–5× longer), excluding validation and data prep.

## 1.1 Summary table (values as logged)

| # | Method | Ran | VRAM @ batch 1: torch peak allocated | torch peak reserved | device-wide used | sec/iter | hrs/epoch (mini) | hrs/epoch (full nuScenes) | Biggest blocker (as stated in RESULTS.md) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Deep Entropy Fusion / Seeing Through Fog | N | not measured | — | — | not measured | not measured | not applicable (method does not use nuScenes) | The official repo contains no model and no training code. It is a dataset toolkit only. |
| 2 | RAF (Reliability-Aware Fusion) | N | not measured | — | — | not measured | not measured | not applicable (method does not use nuScenes) | The frozen pretrained LiDAR-RADAR backbone is not distributed — configs point at the authors' own local absolute paths, with no download URL. |
| 3 | SAMFusion | Y | 7.364 GiB | 9.447 GiB | 11.589 of 11.631 GiB | 0.711 | 0.064 | 5.56 | The nuScenes training init checkpoint (`Fusion_0075_refactor.pth`, a DeepInteraction model) is not published. |
| 4 | BEVFusion | Y (after source fixes; does not run as shipped on this card) | 6.107 GiB | 6.74 GiB | 9.165 of 11.631 GiB | 0.4272 | 0.038 | 3.34 | Bundled spconv reserves 6.47 GB of driver local memory on first launch, which with the model exceeds 12 GB. Fixed by one constant. |
| 5 | RCBEVDet | Y | 1.771 GiB | 2.059 GiB | 3.88 of 11.631 GiB | 0.4942 | 0.044 | 3.86 | None at runtime. Packaging gaps in the released zip. |
| 6 | TransFusion (LiDAR-only TransFusion-L; speed reference, no radar) | Y (as shipped, after one compile fix) | 2.561 GiB | 2.992 GiB | 11.538 of 11.631 GiB (as shipped) | 0.3423 | 0.031 | 2.67 | None fatal. Same spconv local-memory reservation as BEVFusion leaves only ~0.1 GB headroom as shipped. |
| 7 | AFW-Net | N (estimate only; no code exists) | not measured | — | — | ~0.33 (estimated) | n/a — does not use nuScenes | n/a — does not use nuScenes | No public implementation; and its dataset is the same gated STF data that blocks method 1. |

Secondary logged timing fields:

| Method | sec/iter mean | sec/iter compute-only (median, excl. data loading) | data-wait fraction | iterations timed | dataset length |
|---|---|---|---|---|---|
| SAMFusion | 0.7057 | 0.708 | 0.004 | 30 | 323 |
| BEVFusion | 0.4264 | 0.4244 | 0.007 | 30 | 323 |
| RCBEVDet | 0.4952 | 0.4825 | 0.024 | 30 | 323 |
| TransFusion (as shipped) | 0.3415 | 0.3408 | 0.004 | 30 | 323 |
| TransFusion (spconv patched, see 1.2) | 0.3416 | 0.3405 | 0.004 | 30 | 323 |

## 1.2 Per-method detail

### 1. Deep Entropy Fusion / "Seeing Through Fog"
- Repo: princeton-computational-imaging/SeeingThroughFog, commit `bb57f8c5e29a8d647d5f65b0e277401bf332cbd0`.
- 46 Python files; none define a network, loss, optimizer or training loop. The only
  framework imports use TensorFlow to serialize TFRecords. `environment.yml` is named
  `LabelTool` (python 3.7.1) and installs neither tensorflow nor torch.
- Second, independent blocker: `/home/soham/data/seeing_through_fog/` is empty (12K of stubs).
  The dataset is registration-gated (Princeton form → time-limited link, two-stage 7z
  extraction, recurring download-page downtime). **Manual step, still pending — not done.**
- nuScenes-mini deliberately not substituted: the architecture needs a gated NIR camera, which
  nuScenes lacks.
- This project has its own reimplementation under `methods/entropy_fusion/`; that is a
  different artifact from the official repo.

### 2. RAF (Reliability-Aware Fusion)
- Repo parkie0517/RAF; paper Park, Jeong, Yoon, ECCV 2026. Unlike method 1 it has real
  training code (`main_train.py`, `models/`, `ops/` CUDA extensions).
- Blocker 1: pretrained L4DR/3D-LRF backbone only referenced as author-local paths, e.g.
  `/home/user/heejun/L4DR/logs/Simple_Voxel_Encoder/train_simple_voxel_encoder/models/model_8.pt`,
  `/mnt/32THHD/hx/K-Radar-main/logs/...`. No release URL or model zoo. Only the camera
  branch's ImageNet Swin-Tiny is downloadable. **Manual step pending: request the checkpoint
  from the authors — not done.** Per instruction, no from-scratch training.
- Blocker 2: **no VoD loader in the released code**; `datasets/` is K-Radar only (active config
  `KRadarDetection_v2_2`), although the paper reports VoD. Neither K-Radar nor VoD is on the
  machine; both are application-gated (KAIST AVELab; TU Delft form). **Pending — not done.**
- Blocker 3 (dependency probe on Python 3.12): `open3d==0.15.2`, `opencv-python==4.2.0.32`,
  `numba==0.55.1`, `spconv-cu113` unavailable; `nms` 0.1.6 and `setuptools==59.5.0` install.
  Also needs the OpenMMLab stack and six CUDA extensions (`bev_pool`, `ingroup_inds`,
  `iou3d_nms`, `pointnet2`, `roiaware_pool3d`, `roipoint_pool3d`). See the Blocker 3 flag in
  1.3 — this section is now out of date.

### 3. SAMFusion
- Repo torc-ai/SAMFusion, single commit `4a39d0f`, Apache-2.0; ECCV 2024. Config
  `projects/SAMFusion/configs/nuscenes/SAMFusion.py` (camera + LiDAR + radar), with image and
  LiDAR branches frozen (`freeze_img=True, freeze_pts=True`) as configured.
- Slowest and heaviest runnable method. About 40 MB of the card left free at batch 1, and only
  with the branches frozen; RESULTS.md states batch 2, or training the DeepInteraction stage it
  depends on, would not fit. Reserved-minus-allocated gap (~2.1 GB) is caching-allocator
  slack; `PYTORCH_CUDA_ALLOC_CONF` tuning not attempted.
- Checkpoints: `load_from = 'projects/SAMFusion/pretrained/Fusion_0075_refactor.pth'` is not
  published. Published: nuScenes `samfusion_nuscenes_epoch_6.pth` (the finished model,
  457 MB, reachable on S3) and Dense `deepinteraction_dense_epoch_6.pth`. Timing is
  unaffected. **Manual step pending for real training: obtain that file from the authors.**
- Source fixes (in `feasibility/samfusion/patches/`): (1) `deepinteraction.py` passed
  `bbox_head=` to `MVXTwoStageDetector`, which accepts it in neither mmdet3d 1.2.0 nor current
  `main`. The README installs an unpinned mmdetection3d clone over 1.2.0, so it was evidently
  built against an unpublished fork. (2) `update_infos_to_v2.py:301` hard-coded the authors'
  dataset path `/mnt/perception-aisee/Pubblic_Datasets/nuScenes/...`.
- Install issues: `python=3.7.11` unsolvable on conda-forge (used 3.8); `mmcv 2.0.1` is
  published for cu113/torch 1.10.0 (pins consistent); `flash-attn==0.2.2` pinned but never
  imported (skipped); Pillow 10 breaks the old detectron2 wheel (`Image.LINEAR`) → `Pillow==9.5.0`;
  the plugin JIT-compiles a C++ extension on import, so `ninja`/`CUDA_HOME`/compilers must be
  on `PATH` at runtime. Verified: mmdet3d 1.2.0, mmcv 2.0.1, mmdet 3.1.0, spconv 2.3.6.

### 4. BEVFusion (MIT HAN Lab)
- Repo HEAD `326653d` ("[ROCm] add support for ROCm/HIP"). **Not the archived no-radar tree the
  original plan assumed**: this HEAD ships radar encoders and camera+radar configs. Timed config:
  `camera+lidar/swint_v0p075/convfuser.yaml` (Swin-T camera + VoxelNet 0.075 m LiDAR,
  ConvFuser, TransFusion head).
- Why it does not fit as shipped: the first sparse conv fails with
  `indice_cuda.cu 124: cuda execution failed with error 2` at 5.02 GB allocated. Index-pair
  kernels are instantiated with `KernelMaxVolume = 4096` → 64 KB per-thread local array
  reserved for every resident thread (70 SMs × 1536 threads). Measured 6.47 GB held outside
  PyTorch after one launch. The header default is 256; the model uses only 27- and 3-volume
  kernels. At 256 the reservation measured 0.31 GB.
- Six changes needed (in `feasibility/bevfusion/patches/`): `feature_decorator_ext` not
  registered in `setup.py`; `feature_decorator.cpp` torch op used `int` params (torch needs
  `int64_t`); `radar_encoder.py` imports `flash_attn` unconditionally; `DepthLSSTransform` left
  behind a modernised `base.py` (arity, kwargs, hard-coded 1-channel depth); spconv
  4096 → 256; `nuscenes_converter.py:95` joins with `info_prefix` instead of `root_path`
  (GT database therefore not built; `ObjectPaste` dropped from the timed pipeline).
- Caveats in RESULTS.md: branches randomly initialised except ImageNet Swin-T. The published
  recipe fine-tunes with a frozen pretrained LiDAR branch for 6 epochs, so 3.34 h/epoch is
  described as a conservative upper bound for that stage.
- Env: Python 3.8, torch 1.10.0+cu113, mmcv-full 1.4.0 (needs the version-matched OpenMMLab
  wheel index), mmdet 2.20.0; CUDA 11.3 toolkit installed in-env with
  `sysroot_linux-64=2.17` (the machine has no CUDA toolkit; conda's kernel-headers 6.12 define
  `__s128`, which nvcc 11.3 cannot parse).

### 5. RCBEVDet
- **Access request: not needed, and none was ever submitted.** RESULTS.md: the README's
  academic-use application is struck through; code ships in-repo as `rcbevdet-master.zip`.
  Weights are linked on Google Drive (not downloaded; not needed for timing). Nothing is
  pending from you.
- Config: ResNet-50, 256×704, 9 keyframes, radar, depth supervision (12-epoch CBGS upstream).
- Low memory checked: each sample carries a 54×3×256×704 image tensor (6 cams × 9 frames) plus
  radar points (1,928 × 7 on sample 0). The 8 adjacent frames run under `torch.no_grad()`
  (`bevdet_rc.py:755-769`). It is the only method here on spconv 2.x (2.3.6), so no 6.5 GB
  reservation.
- No source edits to RCBEVDet. Gaps in the zip worked around: `requirements/` missing though
  `setup.py` reads it (recreated from mmdet3d v1.0.0rc4 runtime set); `NuScenesDataset_R`
  exported but never defined (aliased to `NuScenesDatasetRC` in a wrapper); leftover
  `from IPython import embed`; converter hard-codes `v1.0-trainval`/`v1.0-test`.
- Install issues: README pairs `mmcv-full 1.6.2` with `mmdet 2.24.0`, but mmdet 2.24.0
  asserts `mmcv<=1.6.0` → used 1.6.0; torch 1.12 rejects gcc 10.4 with CUDA 11.3 → gcc 9.5.0.
  Verified: torch 1.12.1+cu113, mmcv 1.6.0, mmdet 2.24.0, spconv 2.3.6.

### 6. TransFusion
- Timed `transfusion_nusc_voxel_L.py` (TransFusion-L, VoxelNet 0.075 m, 10 sweeps). The
  camera+LiDAR `voxel_LC` variant was **not timed**.
- Same bundled spconv as BEVFusion (4096). Logged comparison:

  | Build | sec/iter | torch allocated | torch reserved | device used |
  |---|---|---|---|---|
  | As shipped (4096) — `result_as_shipped.json` | 0.3423 | 2.561 GiB | 2.992 GiB | 11.538 GiB |
  | Patched (256) — `result_spconv256.json` | 0.3419 | 2.561 GiB | 2.992 GiB | 5.386 GiB |

- Sanity check stated in RESULTS.md: authors report ~2 days on 8× V100 or 8× RTX 3090
  (~384 GPU-hours). The published 20-epoch CBGS schedule extrapolates to about 240 h (~10 days)
  on this single 3080 at batch 1.
- One compile fix: `scatter_points_cuda.cu:272` `index_put_` with a bare tensor index (fails on
  torch ≥ ~1.7). Install issues: `mmpycocotools` needs Cython first; `lyft_dataset_sdk`
  etc. imported unconditionally; mmdet3d 0.11 declares `numpy<1.20` but 1.23.5 was used.
  Verified: torch 1.10.0+cu113, mmcv 1.4.0, mmdet 2.11.0.

### 7. AFW-Net
- Paper: "Adaptive Sensor Fusion for Robust Perception in Dense Fog: A Gated Vision and LiDAR
  Integration Framework", MDPI Sensors 26(12):3728, June 2026, DOI 10.3390/s26123728
  (PMC13306363). AFW-Net = Adaptive Feature-Weighting Network. No public code; no
  reimplementation attempted.
- **Parameter count: 14.5 M — stated by the authors, not an estimate.** Also stated: batch 16
  on 4× RTX 3090, 120 epochs, ~12 FPS inference on one RTX 3090, 15,000-frame Princeton
  Automated Driving Dataset (= Seeing Through Fog / DENSE).
- Architecture (paper): ResNet-50 gated-image encoder with channel attention; PointNet++
  LiDAR encoder (4 SA layers, radii [0.2, 0.4, 0.8, 1.6] m, groups [32, 64, 128, 256]);
  stride-8, C = 256; channel-wise adaptive fusion (two-layer MLPs, hidden 256) with
  `softplus(Conv1x1)` uncertainty and cross-modal attention; FCOS head, strides [8..128].
- Estimates (labelled as such in RESULTS.md): ~121 GFLOPs forward per frame (~75 of it
  ResNet-50 at 1280×720), ~362 GFLOPs per training step; ~9 FPS inference and ~3.0 it/s ≈
  0.33 s/iter on a 3080; ~1.4 h/epoch over 15,000 STF frames; ~167 h (~7 days) for the
  120-epoch schedule.

## 1.3 Verification of SUMMARY.md against RESULTS.md / result.json — flags

Every SUMMARY.md row agrees with its RESULTS.md and `result.json` on ran / sec/iter /
hours / blocker, except as flagged:

1. **BEVFusion device-wide memory:** RESULTS.md says **9.17 GB**, SUMMARY.md says **9.16 GB**,
   logged value is **9.165**. The files round the same number differently.
2. **BEVFusion RESULTS.md contradicts itself on radar.** The header says HEAD `326653d` "ships
   radar encoders and camera+radar configs", but the closing Notes say "No radar and no
   reliability logic in the base config, as expected". The *timed config* has no radar; the
   *repo* does. "As expected" is stale wording from the original plan.
3. **Run-to-run figures that exist only in text:** BEVFusion RESULTS.md cites a clean run at
   0.4237 s/iter, and TransFusion RESULTS.md cites a 0.3438 s/iter run. Neither is on disk: only
   0.4272 (BEVFusion) and 0.3423 / 0.3419 (TransFusion) are. The TransFusion 0.3438 run also
   overlapped with SAMFusion's CPU-side data prep, although RESULTS.md calls all three runs
   "clean".
4. **RAF Blocker 3 is out of date.** It says resolving the dependency set "means a Python 3.8
   interpreter and an older CUDA toolchain, not a pin bump", and attributes `spconv-cu113`'s
   unavailability to CUDA 13. Later in the same pass, exactly that route (micromamba Python 3.8
   plus an in-env CUDA 11.3 toolkit) worked for four methods, and `spconv-cu113` 2.3.6 installed
   under Python 3.8. So the dependency blocker is solvable. Blockers 1 and 2 (checkpoint, data)
   still stand.
5. **AFW-Net RESULTS.md makes an unmeasured comparison.** It says AFW-Net (14.5 M) and this
   project's model (15.73 M) are "far smaller than the BEV methods in this triage". No parameter
   counts for BEVFusion, TransFusion, RCBEVDet or SAMFusion were recorded anywhere on disk.
6. **SAMFusion emphasis differs.** SUMMARY.md's blocker cell leads with "fits with ~40 MB spare
   only because img+LiDAR branches are frozen". RESULTS.md names the unpublished checkpoint as
   the biggest blocker. Both facts appear in both files.
7. **Units:** SUMMARY.md and RESULTS.md label memory "GB", but it is GiB (see top of file).
8. **Hours columns:** SUMMARY.md recomputes hours from sec/iter at print time. Results agree
   with the harness-logged `hrs_epoch_*` values to the displayed precision.

---

# PART 2 — Entropy-gating ablation (gate active vs. gate fixed to 1)

## 2.1 Where the outputs are

`methods/entropy_fusion/outputs/nuscenes_gating_pair_seed42/` (gitignored, local only):
- `manifest.json` — controlled-pair provenance (init-state hash, data-order hashes, source and
  config hashes).
- `on/`, `off/` — per arm: `status.json` (validation records), `eval_epoch_XX/metrics_summary.json`
  (evaluator output), `epoch_XX.pt` + `best.pt` checkpoints, `train_to_epoch_XX.log`, and
  `training/metrics.jsonl` (per-step training losses).
- `on/status_before_resume_20260918.json` and `off/status_at_sigill_20260918.json` — status
  as it stood before the resume.
- `HANDOFF.md` — run notes.

Every mAP/NDS below was checked: `status.json` equals the evaluator's `metrics_summary.json`
exactly for all 8 evaluated checkpoints.

## 2.2 What the two arms are (from `docs/EXPERIMENT_AUDIT_20260915.md` and `manifest.json`)

Controlled pair on full nuScenes trainval → val: the only config difference is the boolean
`entropy_gating`. "on" multiplies features by learned sigmoid gates; "off" multiplies by 1.
Both keep entropy/coverage concatenation. Both have 15,730,860 parameters and identical
initial-state SHA-256 `446a4dccff9a8c1f99d9d75471f1f904ee341b5ddb4a5d0665fba35cdcd35135`.
Shared: seed 42, same data order, six cameras sampled individually, 384×640, batch 2, 20
epochs, AdamW 2e-4, cosine schedule. One seed only. Validation runs at epochs 1, 5, 10, 15
and 20 only. **No mAP/NDS exists for other epochs.** Model checkpoints (`epoch_XX.pt`) are saved
only at those epochs; `training/metrics.jsonl` logs per-step losses, not mAP/NDS.

## 2.3 Status

| Arm | State at time of writing | Epochs evaluated |
|---|---|---|
| on (gate active) | completed, 20/20 epochs | 1, 5, 10, 15, 20 |
| off (gate fixed to 1) | **training** — crashed at epoch 12 (step 981,026) with `Fatal Python error: Illegal instruction`; resumed 2026-09-18T18:25:22Z from step 981,000; at step 983,072 when checked | 1, 5, 10 |

## 2.4 Every evaluated checkpoint

| Epoch | on mAP | on NDS | off mAP | off NDS |
|---|---|---|---|---|
| 1 | 0.11755891685460469 | 0.180140372925398 | 0.10216772063046183 | 0.1588282952650153 |
| 5 | 0.19769936908037639 | 0.25529054432773796 | 0.20087549463480814 | 0.2507299978169498 |
| 10 | 0.2204309885170797 | 0.2827043022599057 | 0.219469500180354 | 0.2792182170669816 |
| 15 | 0.2199491219427115 | 0.28446953652992735 | not yet available | not yet available |
| 20 | 0.21837072782354844 | 0.28592883446366046 | not yet available | not yet available |

Difference (on − off) at the epochs both arms have:

| Epoch | Δ mAP (on − off) | Δ NDS (on − off) |
|---|---|---|
| 1 | 0.015391196224142867 | 0.021312077660382706 |
| 5 | -0.0031761255544317524 | 0.004560546510788133 |
| 10 | 0.0009614883367257132 | 0.0034860851929241488 |

## 2.5 Final-epoch and best-checkpoint results

| | on (gate active) | off (gate fixed to 1) |
|---|---|---|
| Final epoch (20) mAP | 0.21837072782354844 | not yet available |
| Final epoch (20) NDS | 0.28592883446366046 | not yet available |
| Best checkpoint (selection metric = mAP, per `status.json`) | epoch 10: mAP 0.2204309885170797, NDS 0.2827043022599057 | epoch 10 **so far** (only 1/5/10 evaluated): mAP 0.219469500180354, NDS 0.2792182170669816 |
| Highest NDS among evaluated epochs | epoch 20: 0.28592883446366046 | epoch 10 so far: 0.2792182170669816 |

## 2.6 Wall-clock time

| Item | Value (from logged timestamps) |
|---|---|
| on arm | started 2026-09-15T23:27:26.696921Z, finished 2026-09-17T07:12:01.193075Z; `elapsed_seconds` = 114274.49509443299 s |
| Gap between arms | 4.525386 s (off started 2026-09-17T07:12:05.718461Z) |
| off arm, run to failure | `elapsed_seconds` at failure = 66703.89599652798 s → failure ≈ 2026-09-18T01:43:49Z (last pre-crash training-metrics write 2026-09-18T01:41:52Z) |
| Idle between failure and resume | 60092.418777 s (GPU idle; resumed 2026-09-18T18:25:22.033235Z) |
| off arm since resume | 2933.234101 s at 2026-09-18T19:14:15Z, still running |
| **Full run, both arms combined** | **not available — off arm not finished** |
| Training time logged so far (on + off to failure + off since resume, excludes idle gap) | 183911.62519196098 s at 2026-09-18T19:14:15Z |

Estimates already on disk (`HANDOFF.md`; estimates, not measurements): the original pre-launch
estimate was "~56 h for the pair". The post-resume note estimates off completion around
2026-09-19 08:00–10:00 UTC.

## 2.7 Gate-on vs. gate-off

**Not yet available at epoch 20** (gate-off still training). At the latest epoch both arms
have evaluated (epoch 10): gate-on mAP 0.2204309885170797 vs. gate-off 0.219469500180354
(Δ +0.0009614883367257132); gate-on NDS 0.2827043022599057 vs. gate-off 0.2792182170669816
(Δ +0.0034860851929241488).
