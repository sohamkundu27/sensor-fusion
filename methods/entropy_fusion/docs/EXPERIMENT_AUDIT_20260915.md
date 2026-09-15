# Entropy-fusion experiment audit — 15 September 2026

This audit separates measured defects from model limitations. Original checkpoints,
training code, and reported evaluation files were preserved. No new training was
started. Artifacts are under `outputs/kitti_audit_20260915/` (local, ignored by Git).

## Phase 1: KITTI inference, geometry and losses

Evaluated checkpoint: `kitti_full_20260915/epoch_20.pt`, original validation AP_R40
at IoU 0.7: moderate 2D 77.63911%, BEV 7.348965%, 3D 2.3334%.

### Visual inspection and selection

Eight frames were selected at evenly spaced indices among validation frames with
accepted matches, without selecting based on the quality of the 3D result:
000004, 001095, 002158, 003231, 004337, 005335, 006508, 007480. Each PNG contains
independently projected 3D image overlays and BEV boxes over raw LiDAR, green GT
and red predictions. `contact_sheet.jpg` collects all eight.

The quantitative audit covers all 3,769 validation frames. Predictions require
score >=0.1; Hungarian assignment maximizes 2D IoU against Car labels; retained
pairs require 2D IoU >=0.7 and GT moderate difficulty (height >=25 pixels,
occlusion <=1, truncation <=0.3). There are 6,659 retained pairs out of 7,876
moderate GT cars. This is a conditional paired diagnostic, not AP: missed and
poor-2D detections are excluded, and official matching differs.

| Measurement | Result |
|---|---:|
| Mean signed center error, rectified camera X/Y/Z | +0.0449 / +0.0077 / +0.0363 m |
| Median absolute X/Y/Z error | 0.1416 / 0.0579 / 0.4521 m |
| 90th percentile absolute X/Y/Z error | 0.6073 / 0.1540 / 1.9504 m |
| Median Euclidean center error | 0.5158 m |
| Median predicted/GT height, width, length | 1.0124 / 0.9924 / 1.0018 |
| Median / 90th percentile heading error, modulo pi | 6.68 / 31.35 degrees |

Errors are scattered rather than a fixed direction/scale/90-degree rotation.
Frame 001095 has a +4.29 m depth error despite a plausible image projection;
frame 003231 has a -0.11 m depth error and 0.81 3D IoU. Frame 006508 includes a
51.8-degree heading error. These are different failure modes, not one offset.

### Independent calibration and coding checks

On ten evenly spaced validation frames (40 GT cars; 189,542 visible LiDAR points),
independent official `P2 * R0_rect * Tr_velo_to_cam` projection exactly agrees
with the loader's float32 UV/Z. GT -> actual target encoding -> decoding -> KITTI
export maximum errors: bottom center 0.00001312 m, dimensions 0.00000040 m,
heading 0.00000012 radians; corner error versus official devkit construction
0.00001406 m. This rules out the tested rectification, P2 baseline, h/w/l ordering,
bottom/geometric center, and heading convention errors; it is not proof of every
possible inference bug. See `geometry/geometry_report.json` and its audit script.

### Controlled diagnostic substitutions (not new model results)

For the same 6,659 matched pairs, replace selected predicted components with GT
and recompute independent 3D IoU, keeping other components fixed:

| Substitution | Pairs with 3D IoU >=0.7 |
|---|---:|
| None | 16.32% |
| GT depth only | 33.91% |
| GT vertical center only | 20.45% |
| GT dimensions only | 22.75% |
| GT heading only | 24.48% |
| GT full position | 62.97% |
| GT position and heading | 94.28% |

Position is the largest measured source of lost overlap, with heading secondary.
These fractions are not AP and cannot be reported as attainable trained accuracy.

### Loss contribution

The 3D loss is active. Epoch-20 mean losses across logged training microbatches:
CE 0.17412; 2D regression 0.00823; 3D code regression 0.07011; metric-center loss
0.96387 * 0.25 = 0.24097; quality loss 0.58998 * 0.5 = 0.29499; total 0.78842.
The direct 3D terms contribute 39.46% of total (excluding quality). The contribution
is 29.33% in epoch 1 and 40.52% in epoch 10. Mean training center error in epoch 20
is still 1.347 m, including modality dropout and all positive anchors.

There are no logged batches with positive anchors but zero 3D-code or center loss.
All 37,120 totals reconstruct the weighted formula to within 8e-7. 561 batches
(1.51%) have no positives, consistent with 452/3,712 train frames lacking eligible
Car labels. Velocity and attribute losses are intentionally zero for KITTI.
15 updates were skipped by AMP (0.040%); 90.48% were gradient-clipped, a training
stability/tuning concern but not evidence of disabled loss.

An independent validation-batch backward pass, without an optimizer step, measured
nonzero weighted-3D gradient norms at the 3D predictor (36.095), head tower (0.489),
and fusion projection (0.345). UV/depth/dimension/heading terms all have gradients.
See `loss/loss_audit.json`.

### Evaluation protocol issue

The official devkit readme requires the 2D box accompanying a 3D prediction to be
its projected 3D box, for visibility and height filtering. `eval_kitti.py` currently
exports the independently regressed SSD 2D box. This is a real protocol mismatch;
it does not by itself explain the measured physical overlap errors. A diagnostic copy of all 3,769 frames was evaluated with projected 3D bboxes,
near-plane and image clipping. It removed 60/18,829 no-visible-area detections.
Moderate BEV AP changed 7.348965% -> 7.2772475%; 3D AP changed 2.3334% ->
2.3038725%. This mismatch is not the cause of poor 3D accuracy. The same diagnostic
2D AP is 45.006865%, showing that the separate SSD 2D head is substantially more
accurate than the projection of predicted 3D boxes. Original results are preserved.


## Phase 2: controlled nuScenes gating ablation

Both historical models already have entropy gates. The full difference inventory
is [ORIGINAL_REVISED_DIFF.md](ORIGINAL_REVISED_DIFF.md): input channels/rasterization,
entropy representation, progressive feedback, FPN/head/anchors/depth prior,
localization quality, loss terms and negative ratio, batch size, LR/schedule,
weight decay, augmentation, precision, inference scores/thresholds and runtime
recovery all changed. Best-to-best improvement cannot identify the gate's effect.
The historical concat mode also removes entropy input channels and 187,968
parameters; it is not a same-architecture gating-only control.

The prepared configs are `configs/nuscenes_gating_on.json` and
`configs/nuscenes_gating_off.json`. Their only JSON difference is the boolean
`entropy_gating`. Both retain all layers and concatenate entropy/coverage. The
on arm multiplies features by learned sigmoid gates; the off arm uses ones.
This isolates multiplicative gating conditional on entropy feature inputs.
The off arm's unused gate parameters receive no gradients, by design.

Both use 15,730,860 registered parameters, identical initial state SHA-256
`446a4dccff9a8c1f99d9d75471f1f904ee341b5ddb4a5d0665fba35cdcd35135`, seed 42,
full nuScenes train/val, six individually sampled views, 384x640, batch 2,
20 epochs, AdamW 2e-4, weight decay 0.0005, 1000-step LR warmup/cosine,
5000-step center-loss warmup, single-sensor dropout probability 0.5, brightness/
contrast augmentation, and selective AMP. Neither arm initializes from a trained
checkpoint. The same ImageNet camera initialization is used in both.

Validation epochs: 1/5/10/15/20; same score threshold 0.001, quality scores,
image-IoU NMS and cross-view merge. Primary comparison: epoch-20 mAP/NDS.
Best-validation checkpoint is secondary. No mAP early-stop threshold is used.
The launcher runs the two experiments sequentially on the same GPU.

`outputs/nuscenes_gating_pair_seed42/manifest.json` records initial-state hashes,
full per-epoch ordering hashes, 168,780 train views, source/config hashes,
dataset JSON hashes and evaluation protocol. The launcher refuses source,
metadata or Python/PyTorch/CUDA/package drift before either arm. Raw sensor
files are not fully rehashed. Existing train resume now rejects an on/off switch;
old checkpoints default to enabled. 30 focused tests passed, including strict
loading of the existing KITTI checkpoint and same-state/gradient/masking checks.

Training is prepared, NOT started. Preparation and later execution are separate:

```bash
cd /home/soham/sensor-fusion/methods/entropy_fusion
/home/soham/venvs/entropy-fusion/bin/python scripts/run_gating_ablation.py
/home/soham/venvs/entropy-fusion/bin/python scripts/run_gating_ablation.py --run
# For an interrupted pair, use --run --resume instead.
```

CUDA kernels are not forced deterministic, so identical seeds do not promise
bitwise training reproducibility. AMP overflow/skipped updates can differ because
of the treatment; compare those logs. Timing/thermals/OS scheduling differ, and
one seed cannot estimate between-seed variation. Repeat matched seeds before a
strong research claim. Retaining entropy concatenation means a separate no-entropy
information control would be needed to test the entire entropy pathway.
