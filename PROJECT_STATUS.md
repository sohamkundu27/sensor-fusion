# Project Status — sensor-fusion

Compiled 2026-09-23 from files on disk. This is a synthesis pass: nothing was re-run,
re-trained or re-evaluated. It supersedes `RESULTS_FOR_CLAUDE.md` (2026-09-18), whose
gate-off ablation numbers are now stale.

---

## Project Status Summary

This repository compares camera/LiDAR/radar fusion methods for 3D object detection on
public driving datasets, using a single RTX 3080. Three workstreams are finished. First,
a feasibility triage of seven published fusion methods: four run on the card (SAMFusion,
BEVFusion after six source fixes, RCBEVDet, and TransFusion-L as a LiDAR-only speed
reference) and three cannot run (the official Seeing Through Fog repo has no model code,
RAF's pretrained backbone and datasets are unavailable, and AFW-Net has no public code).
Second, the controlled entropy-gating ablation on full nuScenes is **complete in both
arms**: gate-off finished on 2026-09-19 08:16 UTC after a crash and resume. At the
pre-registered primary endpoint (epoch 20), gate-on minus gate-off is +0.038 percentage
points (pp) mAP and −0.002 pp NDS, from one seed. Third, the KITTI audit traced the
2.33% moderate car 3D AP to scattered position/depth error, not to calibration,
transforms or box math. The BEVFusion reference comparison then found the same failure
on nuScenes: boxes are the right size but in the wrong place. Seeing Through Fog data
is still not downloaded, and several third-party checkpoints must be requested from
their authors. The current rig and scope come from conversation notes: radar dropped,
single-photon (SP) LiDAR + RGB cameras, CARLA → Mitsuba synthetic data, and a
beta-binomial confidence model for the SP LiDAR. They are recorded under Recent
decisions below and appear in no code, config or other doc in the repo.

---

## Sensor Rig & Dataset Status

### Sensor rig

**No file in the repository describes a physical sensor rig.** A case-insensitive search
of every `.md` file for "rig" and similar terms returns nothing. The intended rig
(single-photon LiDAR + RGB cameras, radar dropped) is described only in Recent
decisions below. None of the datasets on disk contains single-photon LiDAR data. The
only sensor suites in the repo are those of public datasets:

| Source | Sensors |
|---|---|
| nuScenes (primary dataset) | 6 cameras, 1 LiDAR, 5 radars |
| KITTI (interim dataset) | camera + LiDAR (no radar, no gated NIR) |
| Seeing Through Fog / DENSE (target dataset, not downloaded) | RGB, gated NIR, LiDAR, radar |
| Waymo Open v1.4.3 subset | cameras + LiDAR |

The project's own entropy-fusion model (`methods/entropy_fusion/`, per `MODEL_PLAN.md` and
`REIMPLEMENTATION.md`) processes each nuScenes camera view individually at 384×640. It
projects the current LiDAR sweep (depth, height, intensity) and all five current radars
(depth, RCS, compensated radial velocity) into the image plane. It has no gated NIR
stream and no temporal sweep accumulation. It has 15,730,860 parameters.

Compute: one RTX 3080 with 11.631 GiB usable to CUDA (nvidia-smi lists 12,288 MiB),
driver 580.173.02, kernel 7.0.0-31-generic. There is no system CUDA toolkit; each
feasibility method uses its own micromamba env with an in-env CUDA 11.3. System Python
is 3.12.3.

### Datasets (disk state checked 2026-09-23)

| Dataset | On disk | Status | Used by |
|---|---|---|---|
| nuScenes | `~/data/nuscenes`, 402 GB: full `v1.0-trainval` + `v1.0-mini` | Complete. Presence/nonzero-size checks passed (34,149 train/val keyframes). **Publisher MD5 mismatches unresolved.** | Entropy-fusion full runs, gating ablation, feasibility timing (mini) |
| KITTI object | `~/data/kitti_full` (20 GB; 7,481/7,481 frames verified for image_2, velodyne, calib, label_2); `~/data/kitti` (40-frame subset) | Complete; download finished 2026-09-15 19:00 UTC | KITTI car experiment, Phase 1 audit |
| Waymo Open v1.4.3 | `~/data/waymo`, 6.4 GiB, 5 train + 2 val segments | Pipeline-test subset only | No experiment write-up reports results on it |
| Seeing Through Fog / DENSE | `~/data/seeing_through_fog`: 12K of empty directories. Toolkit at `~/data/seeing_through_fog_tools` (commit `bb57f8c`) | **Not downloaded.** Registration-gated manual step pending. Split manifests prepared (`outputs/stf_preparation`) | Blocks the official STF method, AFW-Net, and this project's STF experiment |
| K-Radar, VoD | Not on machine | Application-gated; not requested | RAF only |

### What the planning docs currently claim

Still accurate:

| Doc | Claim |
|---|---|
| `README.md` | Scope: "a research workspace for comparing object-detection and sensor-fusion methods on autonomous-driving datasets." |
| `README.md`, `SEEING_THROUGH_FOG.md` | STF access pending registration; published split files need overlap checks. |
| `README.md` | TransFusion environment and short LiDAR-only benchmark verified; full training pending. |
| `README.md`, `methods/entropy_fusion/README.md`, `REIMPLEMENTATION.md` | Revised entropy-fusion nuScenes run: 21.96% mAP / 28.60% NDS (best checkpoint, epoch 15) vs 15.91% / 24.17% for the first implementation. This is run `paper_v2_full_20260914`, **separate from the gating-ablation arms below**. |
| `README.md` | KITTI "cannot test the paper's radar or gated-camera streams." |

Stale (the disk has moved past them):

| Doc | Claim | Disk state |
|---|---|---|
| `README.md`, `methods/entropy_fusion/README.md`, `KITTI_EXPERIMENT.md` | Full KITTI labeled set "downloading"; training "queued" | Download completed and the 20-epoch run finished on 2026-09-15 |
| `README.md` | Neither adaptation establishes entropy's contribution "without a controlled comparison" | The controlled gating pair has completed (results below); README not updated |
| `EXPERIMENT_AUDIT_20260915.md` (Phase 2) | "Training is prepared, NOT started" | Both arms completed 2026-09-19 |
| `TRAINING.md` | Original full experiment: "full validation and final accuracy remain pending" | Completed 2026-09-14 (15.91% / 24.17%) |
| `methods/transfusion/docs/WAYMO_SUBSET.md` | Only `v1.0-mini` is stored locally; trainval downloaded separately | Full `v1.0-trainval` is local |
| `feasibility/raf/RESULTS.md` Blocker 3 | Dependency set needs "Python 3.8 and an older CUDA toolchain", presented as unresolved | That exact recipe later worked for four methods (see flag 4 below) |

---

## Method Feasibility Table

Sources: `feasibility/SUMMARY.md` (generated 2026-09-18 18:23 UTC), all seven
`feasibility/<method>/RESULTS.md`, all `result*.json`.

Setup for measured rows: RTX 3080, batch size 1, 5 warmup + 30 timed training iterations
(forward + backward + optimizer step), nuScenes-mini (323 train keyframes), idle GPU,
measured 2026-09-18. sec/iter is the wall-clock median including data loading. Hours are
sec/iter × 323 (mini) or × 28,130 (full nuScenes), **without CBGS**. All four runnable
methods use CBGS upstream, which makes a published epoch roughly 4–5× longer. Memory is
in **GiB**: the harness divides bytes by 1024³ (`bench_common.py`, `bench_mmengine.py`),
although the source files label it "GB".

| # | Method | Ran | VRAM @ batch 1: torch peak allocated / device-wide used (of 11.631) | sec/iter | h/epoch mini | h/epoch full nuScenes | Biggest blocker |
|---|---|---|---|---|---|---|---|
| 1 | Deep Entropy Fusion / Seeing Through Fog (official repo) | N | not measured | — | — | n/a (no nuScenes) | Official repo has no model or training code (dataset toolkit only); STF data not downloaded (registration-gated) |
| 2 | RAF (Reliability-Aware Fusion) | N | not measured | — | — | n/a (no nuScenes) | Frozen L4DR/3D-LRF backbone never published (author-local paths only); no VoD loader; K-Radar/VoD not local |
| 3 | SAMFusion | Y, at the edge of the card (~40 MB free) | 7.364 / 11.589 | 0.711 | 0.064 | 5.56 | nuScenes init checkpoint `Fusion_0075_refactor.pth` (DeepInteraction) not published; fits only because image and LiDAR branches are frozen |
| 4 | BEVFusion (MIT HAN Lab) | Y after 6 source/config fixes; **not as shipped** | 6.107 / 9.165 | 0.4272 | 0.038 | 3.34 | Bundled spconv reserves 6.47 GB of driver local memory; fixed by `KernelMaxVolume` 4096 → 256 |
| 5 | RCBEVDet | Y | 1.771 / 3.88 | 0.4942 | 0.044 | 3.86 | None at runtime. The zip is missing `requirements/` and a dataset class. The access gate is gone |
| 6 | TransFusion-L (LiDAR-only speed reference, no radar) | Y as shipped, after 1 compile fix | 2.561 / 11.538 as shipped (5.386 with the spconv fix, same 0.3419 s/iter) | 0.3423 | 0.031 | 2.67 | None fatal; spconv reservation leaves ~0.1 GB headroom as shipped |
| 7 | AFW-Net | N (paper-derived estimate) | not measured | ~0.33 (est.) | n/a | n/a (STF: ~1.4 h/epoch est.) | No public code; its dataset is the same gated STF data as method 1 |

Per-method notes:

- **Seeing Through Fog:** 46 Python files, none defining a network, loss or training
  loop. `environment.yml` installs neither TensorFlow nor PyTorch. nuScenes was
  deliberately not substituted because the architecture needs a gated NIR camera.
- **RAF:** Real training code exists. It is blocked on inputs: the checkpoint and the
  K-Radar/VoD data.
- **SAMFusion:** 2 source fixes (an unsupported `bbox_head=` argument; a hard-coded
  author dataset path). Batch 2, or training the DeepInteraction stage it depends on,
  would not fit.
- **BEVFusion:** The repo HEAD `326653d` ships radar encoders, but the timed config
  (camera+LiDAR ConvFuser) has no radar. The GT database was not built because of a
  converter path bug, so `ObjectPaste` was dropped. Branches are randomly initialised
  except the ImageNet Swin-T, so 3.34 h/epoch is described as a conservative upper
  bound for the fusion stage.
- **RCBEVDet:** 6 cams × 9 frames per sample. Only the key frame carries gradients,
  which is why memory is low. It is the only method on spconv 2.x, so it has no 6.5 GB
  reservation.
- **TransFusion-L:** The camera+LiDAR `voxel_LC` variant was **not timed**. The authors'
  cost (~384 GPU-hours) is consistent with the ~240 h (~10 days) extrapolation for the
  20-epoch CBGS schedule on this card.
- **AFW-Net:** 14.5 M parameters (authors' figure). ~121 GFLOPs forward per frame and
  ~167 h for the 120-epoch schedule are estimates.

### Where SUMMARY.md and RESULTS.md disagree (re-checked against current files)

Every SUMMARY.md row matches its RESULTS.md and `result.json` on ran/not-ran, sec/iter,
hours and blocker, except:

1. **BEVFusion device-wide memory:** RESULTS.md says 9.17 GB, SUMMARY.md says 9.16 GB,
   and the logged value is 9.165. The two files round the same number differently.
2. **BEVFusion radar wording:** The RESULTS.md header says HEAD "ships radar encoders and
   camera+radar configs". Its closing note says "No radar ... in the base config, as
   expected". The timed config has no radar; the repo does. "As expected" is stale
   wording from the original plan.
3. **Timings that exist only in prose:** BEVFusion RESULTS.md cites a 0.4237 s/iter run,
   and TransFusion RESULTS.md cites a 0.3438 s/iter run. Neither is on disk; only 0.4272
   and 0.3423/0.3419 are. RESULTS.md calls all three TransFusion runs "clean". The
   2026-09-18 report said the 0.3438 run overlapped SAMFusion's CPU-side data prep. No
   timestamp for that run is on disk, so this pass could not re-confirm the overlap.
4. **RAF Blocker 3 is out of date.** It says the dependency set needs Python 3.8 and an
   older CUDA toolchain. Later in the same pass, exactly that route worked for four
   methods, and `spconv-cu113` 2.3.6 installed under Python 3.8. Blockers 1 and 2
   (checkpoint, data) still stand.
5. **AFW-Net makes an unmeasured comparison.** It calls AFW-Net (14.5 M) and this
   project's model (15.73 M) "far smaller than the BEV methods". No parameter counts
   for the BEV methods exist on disk.
6. **SAMFusion emphasis:** SUMMARY.md leads with "fits with ~40 MB spare only because
   branches are frozen". RESULTS.md names the unpublished checkpoint as the biggest
   blocker. Both facts appear in both files.
7. **Units:** Both files say "GB", but the values are GiB.
8. SUMMARY.md recomputes hours from sec/iter at print time. The results agree with the
   logged `hrs_epoch_*` values to displayed precision (not a disagreement; noted for
   completeness).

Other cross-document disagreements found during this pass:

- **TransFusion-L speed differs about 2.9× between two write-ups.**
  `methods/transfusion/docs/PRECHECK_RESULTS.md` (2026-09-13) records 0.976 s/iter:
  mean of the last 10 of 12 iterations, torch 1.7.0+cu110 / mmcv-full 1.2.4, object
  sampling on, gradient clipping 0.1. The feasibility pass records 0.3423 s/iter:
  median of 30, torch 1.10.0+cu113 / mmcv 1.4.0, `ObjectSample` dropped. Peak
  allocation agrees (2.563 vs 2.561 GiB). The resulting 20-epoch CBGS extrapolations
  also differ: about 29 days (precheck) vs about 10 days (feasibility). Neither doc
  reconciles the two.
- **Two different "BEVFusion"s.** The feasibility pass timed MIT HAN Lab's BEVFusion.
  The Phase 3 reference audit compares against the ADLab (Liang et al.) BEVFusion
  recipe (`bevf_tf_4x8_6e_nusc.py`). Same name, different method.
- **Two STF registration URLs.** `feasibility/seeing-through-fog/RESULTS.md` points to
  light.princeton.edu; `SEEING_THROUGH_FOG.md` points to the uni-ulm.de DENSE
  registration form.

---

## Entropy-Gating Ablation

Sources: `methods/entropy_fusion/outputs/nuscenes_gating_pair_seed42/` (gitignored, local
only): `manifest.json`, `on/status.json`, `off/status.json`,
`off/status_at_sigill_20260918.json`, `on/status_before_resume_20260918.json`,
`HANDOFF.md`, `launcher.log`, `*/eval_epoch_XX/metrics_summary.json`,
`*/training/report.json`. Design is from `docs/EXPERIMENT_AUDIT_20260915.md`.

**Changed since the last report (2026-09-18):** gate-off finished. Epoch 15 and
epoch 20 are newly evaluated, gate-off's best checkpoint moved from epoch 10 to 15, and
the full wall-clock is now available.

### Design

This is a controlled pair on full nuScenes trainval → val. The only config difference is
the boolean `entropy_gating`: "on" multiplies features by learned sigmoid gates, "off"
multiplies by 1. Both arms keep entropy/coverage concatenation, have 15,730,860
parameters, and share the initial-state SHA-256 `446a4dcc…cd35135`. Shared settings:
seed 42, identical data order (hashed per epoch), 168,780 train views, six cameras sampled
individually, 384×640, batch 2, 20 epochs, AdamW 2e-4 with cosine schedule, selective AMP.
Validation runs at epochs 1, 5, 10, 15 and 20 only. The pre-registered **primary endpoint
is epoch-20 mAP/NDS**; best-validation mAP is secondary.

### Status

| Arm | State | Epochs evaluated | Notes |
|---|---|---|---|
| on (gate active) | **completed**, 20/20 | 1, 5, 10, 15, 20 | Skipped, not retrained, by the 2026-09-18 resume |
| off (gate fixed to 1) | **completed**, 20/20 | 1, 5, 10, 15, 20 | SIGILL crash at step 981,026 (epoch 12), ~01:43 UTC 2026-09-18; resumed 18:25 UTC from step 981,000; finished 2026-09-19 08:16:53 UTC |

`manifest.json` state: `completed`. No training process is running and the GPU is idle.

### Every evaluated checkpoint (percent, rounded to 3 decimals)

| Epoch | on mAP | off mAP | Δ mAP (on − off), pp | on NDS | off NDS | Δ NDS (on − off), pp |
|---|---|---|---|---|---|---|
| 1 | 11.756 | 10.217 | +1.539 | 18.014 | 15.883 | +2.131 |
| 5 | 19.770 | 20.088 | −0.318 | 25.529 | 25.073 | +0.456 |
| 10 | 22.043 | 21.947 | +0.096 | 28.270 | 27.922 | +0.349 |
| 15 | 21.995 | **22.207** | −0.212 | 28.447 | **28.766** | −0.319 |
| **20 (primary)** | **21.837** | **21.799** | **+0.038** | **28.593** | **28.595** | **−0.002** |

All ten `status.json` values equal the evaluator's `metrics_summary.json` (`mean_ap`,
`nd_score`) exactly. The sign of Δ mAP alternates across epochs (+, −, +, −, +); Δ NDS is
positive through epoch 10 and negative at 15 and 20.

Full-precision values as logged (0–1 scale):

| Epoch | on mAP | on NDS | off mAP | off NDS |
|---|---|---|---|---|
| 1 | 0.11755891685460469 | 0.180140372925398 | 0.10216772063046183 | 0.1588282952650153 |
| 5 | 0.19769936908037639 | 0.25529054432773796 | 0.20087549463480814 | 0.2507299978169498 |
| 10 | 0.2204309885170797 | 0.2827043022599057 | 0.219469500180354 | 0.2792182170669816 |
| 15 | 0.2199491219427115 | 0.28446953652992735 | 0.22207166442650367 | 0.28766044265961654 |
| 20 | 0.21837072782354844 | 0.28592883446366046 | 0.21799181035386725 | 0.28595153638246934 |

Deltas at full precision (on − off): epoch 20 Δ mAP = 0.0003789174696811892,
Δ NDS = −0.000022701918808876176; epoch 15 Δ mAP = −0.002122542483792178,
Δ NDS = −0.003190906129689197.

### Primary endpoint and best checkpoints

| | on (gate active) | off (gate fixed to 1) |
|---|---|---|
| Epoch 20 (primary) | 21.837% mAP / 28.593% NDS | 21.799% mAP / 28.595% NDS |
| Best validation mAP (secondary; `best_epoch` in `status.json`) | epoch 10: 22.043% / 28.270% | epoch 15: 22.207% / 28.766% |
| Highest NDS among evaluated epochs | epoch 20: 28.593% | epoch 15: 28.766% |

### Wall-clock time

| Item | Value |
|---|---|
| on arm | started 2026-09-15 23:27:26 UTC, finished 2026-09-17 07:12:01 UTC; `elapsed_seconds` 114,274.5 s (31.74 h). The current `on/status.json` `finished_utc` (2026-09-18 18:25:13) was rewritten by the resume; the original is in `status_before_resume_20260918.json`. |
| Gap between arms | 4.5 s (off started 2026-09-17 07:12:05 UTC) |
| off arm | `elapsed_seconds` 116,595.2 s (32.39 h). This excludes the idle gap: 66,703.9 s to the crash + 49,891.3 s from resume to finish. It includes the ~1–1.5 h loader fast-forward after the resume. |
| Idle after the crash | 60,092.4 s (16.69 h), GPU idle until the manual resume |
| **Both arms, logged run time** | **230,869.7 s (64.13 h)** |
| Calendar span, on start → off finish | 290,966.6 s (80.82 h) = run time + idle gap + 4.5 s |
| Estimates on disk vs actual | Pre-launch `HANDOFF.md` estimate: "~56 h for the pair" (actual logged 64.13 h). Post-resume ETA 2026-09-19 08:00–10:00 UTC (actual 08:16:53). |

Final-stage peak VRAM (`training/report.json`, last 5 epochs only): on 0.669 GiB,
off 0.656 GiB allocated.

### Interpretation limits (from `HANDOFF.md` and `manifest.json`)

- One paired seed: exploratory, not a variance estimate. Repeat matched seeds before a
  strong claim.
- CUDA kernels are not forced deterministic, so the same seed does not mean
  bitwise-identical training.
- This measures multiplicative gating *given* retained entropy concatenation. A
  no-entropy control is needed to test the entropy pathway as a whole.
- Gate-off's gate parameters stay registered but receive no gradient.
- AMP skipped-update counts can diverge because of the treatment. The per-step
  `skipped_optimizer_step` flag is logged in `training/metrics.jsonl`, but no per-arm
  total has been written to any file yet.

---

## KITTI Root-Cause Findings

Sources: `methods/entropy_fusion/docs/EXPERIMENT_AUDIT_20260915.md` (Phase 1),
`docs/KITTI_EXPERIMENT.md`, `outputs/kitti_full_20260915/status.json`,
`docs/REFERENCE_GAP_AUDIT_20260915.md` (Phase 3). There is no top-level `docs/`; these
live under `methods/entropy_fusion/docs/`.

### The KITTI number

Camera+LiDAR, car-only, 3,712/3,769 train/val split, 384×1248, 20 epochs from ImageNet
camera init. The run finished 2026-09-15 19:00 UTC. Best checkpoint: epoch 20. Official
devkit (SHA-256-pinned), AP_R40 at IoU 0.7:

| | Easy | **Moderate** | Hard |
|---|---|---|---|
| 2D | 86.066% | **77.639%** | 68.515% |
| BEV | 8.197% | **7.349%** | 5.824% |
| **3D** | 2.605% | **2.333%** | 1.834% |

Moderate 3D AP by validation epoch (1/5/10/15/20): 0.002 / 0.779 / 0.520 / 2.120 / 2.333%.

### What was ruled out

- **Calibration:** On 10 evenly spaced validation frames (40 GT cars, 189,542 visible
  LiDAR points), an independent official `P2 · R0_rect · Tr_velo_to_cam` projection
  exactly matches the loader's float32 UV/Z.
- **Transforms and box math:** GT → target encoding → decoding → KITTI export round trip
  has maximum errors of 1.312e-5 m (bottom center), 4.0e-7 m (dimensions) and 1.2e-7 rad
  (heading). Corners match the official devkit construction to 1.406e-5 m. This rules out
  rectification, P2 baseline, h/w/l ordering, bottom vs geometric center, and heading
  convention errors. The audit states this "is not proof of every possible inference bug."
- **Dead 3D loss:** The 3D loss is active and supplies 39.46% of the total at epoch 20.
  No batch has positive anchors with zero 3D loss, and nonzero 3D gradients reach the
  predictor, head and fusion projection. 15 AMP-skipped updates (0.040%); 90.48% of
  updates were gradient-clipped, which the audit flags as a stability/tuning concern
  rather than evidence of a disabled loss.
- **Evaluation protocol:** `eval_kitti.py` exports the separately regressed SSD 2D box,
  not the projected 3D box the devkit expects. This is a real mismatch but not the
  cause: evaluating with projected boxes moved moderate BEV AP from 7.349% to 7.277% and
  3D AP from 2.333% to 2.304%. Projected-box 2D AP is 45.01% vs 77.64% for the SSD head.

### The actual root cause: position (mainly depth), then heading

Diagnostic over all 3,769 validation frames: 6,659 prediction-GT pairs retained out of
7,876 moderate GT cars (score ≥ 0.1, Hungarian 2D matching, 2D IoU ≥ 0.7). This is a
conditional diagnostic, not AP.

| Measurement | Result |
|---|---|
| Median absolute center error X / Y / Z (camera frame; Z = depth) | 0.1416 / 0.0579 / **0.4521 m** |
| 90th percentile absolute error X / Y / Z | 0.6073 / 0.1540 / **1.9504 m** |
| Median Euclidean center error | 0.5158 m |
| Median predicted/GT height, width, length | 1.0124 / 0.9924 / 1.0018 (sizes are right) |
| Median / 90th percentile heading error (mod π) | 6.68° / **31.35°** |

The errors are scattered, not a fixed offset, scale or 90° rotation. Examples: frame
001095 has +4.29 m depth error with a plausible image projection; frame 006508 has a
51.8° heading error.

### GT-substitution test

For the same 6,659 pairs, selected predicted components were replaced with GT and 3D IoU
recomputed. These are diagnostic fractions, not AP or attainable accuracy.

| Substitution | Pairs with 3D IoU ≥ 0.7 |
|---|---|
| None | 16.32% |
| GT vertical center only | 20.45% |
| GT dimensions only | 22.75% |
| GT heading only | 24.48% |
| GT depth only | 33.91% |
| **GT full position** | **62.97%** |
| **GT position + heading** | **94.28%** |

Conclusion stated in the audit: position is the largest source of lost overlap, and
heading is secondary.

### BEVFusion error-profile comparison (Phase 3)

The audit compared our revised nuScenes run (`paper_v2_full_20260914`, epoch 15: 21.956%
mAP / 28.600% NDS) against the **ADLab (Liang et al.) BEVFusion** recipe, whose reference
result is 69.2 / 67.9. Reference configs were fetched verbatim and hashed. Note that this
is not the MIT HAN Lab BEVFusion timed in the feasibility pass.

- **Missing augmentation does not explain the gap.** The reference fusion stage uses no
  CBGS and no geometric augmentation; all of that lives in its frozen LiDAR
  pretraining. The real differences are 10-sweep point accumulation, a 20-epoch
  CBGS-balanced LiDAR-pretrained frozen BEV backbone at 1440×1440 voxels, effective
  batch 32, and a BEV-space head. We have none of these.
- **The error profile is qualitatively different, not a scaled-down copy:**
  1. **Velocity is forfeited.** mAVE 1.3943 gives a TP score of exactly 0.0, losing all
     10% of NDS that velocity carries, because there is no temporal input.
  2. **The class ranking is inverted.** traffic_cone 40.10 and barrier 37.38 AP rank
     above car 37.34, which is a camera-detector profile rather than a LiDAR-BEV one.
  3. **Scale is healthy; translation and orientation are not.** mASE 0.3399 and mAAE
     0.2164 are normal, but mATE 0.8424 and mAOE 0.8390 score only 0.158 and 0.161.
     This is the same dissociation Phase 1 measured on KITTI: "the box is the right
     size and in the wrong place."
  4. **Large, rare classes collapse.** trailer 2.56 and construction_vehicle 3.12 AP.
- **Verdict (from the audit):** the gap is architectural and input-level, not a
  training-budget shortfall. About 10% of NDS (velocity) is structurally unavailable
  and about 17% is lost to translation/orientation. The reference number is a 6-epoch
  fine-tune on an inherited, frozen LiDAR detector, so it measures a different
  quantity.
- **Caveat stated in the audit:** BEVFusion's published per-class TP error table was
  not among the fetched artifacts and could not be verified offline. The comparison is
  against the recipe's structure and our own internal dissociations.

---

## Recent decisions (not yet in repo)

*Source: conversation notes, pasted 2026-09-23. Nothing below is reflected in any code,
config or other doc in the repository. A search for CARLA, Mitsuba, single-photon,
SPAD, transient, photon and beta-binomial finds only unrelated mentions of a "transient"
systemd unit.*

**Scope change:** radar dropped from the sensor rig. Fusion scope is now
single-photon LiDAR + RGB cameras only, not the original LiDAR+camera+radar
triple. This deprioritizes any radar-dependent method (RCBEVDet, and the
radar branch of SAMFusion/RAF) and makes camera+LiDAR-only architectures
(TransFusion, BEVFusion) the more direct fit.

**Data generation plan:** CARLA for world/scene simulation, feeding into
Mitsuba for physically accurate rendering of both RGB and SP-LiDAR
transients (fog scattering included), off the same underlying scene, so
both modalities stay geometrically and temporally consistent. This is the
intended fix for the annotated-training-data blocker that shows up
repeatedly in Part 1 above: perfectly matched, freely-annotated ground
truth for our exact sensor rig, generated synthetically. The CARLA-to-
Mitsuba-XML bridge is itself being treated as a contribution, building on
prior Unreal-to-Mitsuba work, not just infrastructure. Full transient
rendering is expensive (potentially weeks at full temporal/spatial
resolution; reducible by dropping resolution).

**Uncertainty-quantification approach (the core theoretical contribution):**
per-measurement confidence for the SP LiDAR, decomposed into three sources:
(1) shot noise from photon-counting statistics (binomial sampling over
arrival rates), (2) scene-level uncertainty from the physical scattering
interaction (single-scattering forward model for fog), (3) uncertainty in
the forward model's own approximations, handled by treating the model's
rate parameter as a random variable rather than fixed, which turns the
binomial into a beta-binomial and yields a calibrated confidence value.
Measurement itself is a generalized likelihood ratio test: divergence
between observed photon counts and what the forward model predicts under
corruption-only.

**Two evaluation axes:** downstream detection metrics (mAP/NDS-style) for
whether the confidence signal helps perception, and a separate model-
validation metric (a deflection-style statistic) for whether the forward
model matches reality, independent of downstream task performance.

**Open risk, stated plainly:** no way currently exists to validate the
forward model against real transient data. Real fog is too inconsistent to
collect reliably, public datasets never ship raw photon histograms (only
point clouds or coarse "echoes"), and the one hardware option that would
give full transients is ~$40k and unlikely to be purchased. Validation is
simulation-only for now; that's a defensible choice but should be stated
as a choice, not discovered as a gap later.

**Hypothesis, stated at the strength it was actually made:** the confidence
signal is expected to help training convergence speed. Whether it improves
final detection accuracy is explicitly uncertain, not assumed.

---

## Open Blockers

Every "pending", "not done" or "blocked" item from the sections above, deduplicated.
Items tagged **[radar]** are deprioritized by the scope change in Recent decisions.

1. **Seeing Through Fog / DENSE data not downloaded.** Registration-gated manual step,
   not done; local directories are empty. This one item blocks the official STF method,
   AFW-Net, and this project's own STF experiment. The docs cite two registration URLs
   (light.princeton.edu and the uni-ulm.de DENSE form).
2. **STF split overlaps unresolved and evaluation protocol not frozen.** `rain.txt`
   shares 23 frames with clear training, 96 with clear validation, 22 with clear test
   and 20 with light fog. AP interpolation, IoU thresholds, ignored-label handling and
   2D-vs-3D output for Table 2 are also unestablished. Must be resolved before any STF
   training.
3. **No STF detector to train.** The official repo ships no model. The four-stream
   dataset adapter and paper-matched detector are not implemented
   (`SEEING_THROUGH_FOG.md` next steps 3–5).
4. **[radar] RAF pretrained L4DR/3D-LRF checkpoint** (the frozen LiDAR-radar backbone):
   must be requested from the authors; not done.
5. **[radar] RAF data:** K-Radar (KAIST) or VoD (TU Delft) access not requested. The
   released code has no VoD loader.
6. **RAF environment:** the Python 3.8 + CUDA 11.3 route is now known to work, but it
   has not been attempted for RAF (six CUDA ops plus the OpenMMLab stack).
7. **SAMFusion nuScenes init checkpoint** (`Fusion_0075_refactor.pth`): must be requested
   from the authors; not done. Training the DeepInteraction stage locally would not fit
   at batch 1.
8. **AFW-Net has no public code.** No reimplementation attempted.
9. **TransFusion beyond the LiDAR-only timing:** the camera+LiDAR `voxel_LC` variant is
   not timed. The fusion init checkpoint `fusion_voxel0075_R50.pth` is not shipped (a
   missing prerequisite). Full-trainval infos and GT database are not generated. Full
   training is pending, and full-data/validation memory is unmeasured.
10. **GT-sampling databases not built in the feasibility envs:** BEVFusion (converter
    path bug; `ObjectPaste` dropped) and TransFusion (`ObjectSample` dropped). This is
    irrelevant to timing but needed for real training.
11. **nuScenes archive MD5 mismatches unresolved.** Only presence and size were checked;
    content-level integrity (including radar) is not verified.
12. **TransFusion-L speed discrepancy unreconciled:** 0.976 s/iter (precheck) vs 0.342
    s/iter (feasibility).
13. **Gating ablation follow-ups from `HANDOFF.md`, not done:** per-arm AMP
    skipped-update counts have not been collected, and the consolidated three-phase
    write-up has not been written.
14. **Gating ablation evidence limits:** single seed; matched repeat seeds are needed
    before a strong claim. A no-entropy control is needed to test the entropy pathway as
    a whole.
15. **Machine stability unexplained:** the gate-off SIGILL on 2026-09-18 (hardware or
    memory suspected, unconfirmed; `dmesg` needs permissions) and the 2026-09-13 kernel
    page fault (cause unknown). The handoff notes it "may recur."
16. **KITTI export protocol mismatch:** `eval_kitti.py` exports the regressed SSD 2D box
    instead of the projected 3D box. It is measured as not the cause, but no doc says
    the exporter was changed.
17. **Phase 3 reference table unverified:** BEVFusion's published per-class TP errors
    were not fetched, so per-class gap sizes are unverified.
18. **Docs out of date:** see the "Stale" table under Sensor Rig & Dataset Status
    (README KITTI and controlled-comparison lines, KITTI_EXPERIMENT, EXPERIMENT_AUDIT
    Phase 2, TRAINING, WAYMO_SUBSET, RAF Blocker 3, BEVFusion radar note). In addition,
    `README.md` ("camera, LiDAR, and radar measurements") and
    `methods/entropy_fusion/README.md` ("camera images with LiDAR and radar") still
    describe the pre-scope-change sensor set.
19. **Nothing from the new plan exists in the repo yet:** no CARLA scenes, no Mitsuba
    rendering, no CARLA-to-Mitsuba-XML bridge, no SP-LiDAR data or forward model, and no
    uncertainty-quantification or GLRT code or docs.
20. **No real-data validation path for the SP-LiDAR forward model (stated as a deliberate
    choice):** real fog is too inconsistent to collect, public datasets ship no raw
    photon histograms, and full-transient hardware (~$40k) is unlikely to be purchased.
    Validation is simulation-only for now.
21. **Transient rendering cost:** potentially weeks at full temporal/spatial resolution;
    reducible by dropping resolution.

Marked "pending" in older docs but **already done on disk** (not blockers): full KITTI
download and 20-epoch KITTI run (2026-09-15); original full nuScenes experiment
(2026-09-14); paper_v2 revised run (2026-09-15); both gating-ablation arms (2026-09-19);
post-reboot driver repair and recovery (`FULL_EXPERIMENT.md`, 2026-09-13).
