# Paper-informed nuScenes reimplementation

This revision follows a direct reading of `../entropyfusion.pdf` (Bijelic et al.,
CVPR 2020) and its [official supplement](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/figures/AdverseWeatherFusion_Supplement.pdf).
It implements the paper's progressive, entropy-steered image-plane fusion idea
more closely. It remains a three-modality **nuScenes 3D adaptation**, with no gated
NIR sensor, rather than a reproduction of their detector or results. The official
SeeingThroughFog repository provides dataset utilities, not a complete released
training implementation that we could simply port.

## What the paper reports

Table 2 reports car average precision on their own weather dataset, with KITTI
easy/moderate/hard difficulty categories and clear-weather training. There is no
nuScenes NDS or ten-class nuScenes mAP in that table.

| Weather | Easy AP | Moderate AP | Hard AP |
|---|---:|---:|---:|
| Clear | 89.84% | 85.57% | 79.46% |
| Light fog | 90.54% | 87.99% | 84.90% |
| Dense fog | 87.68% | 81.49% | 76.69% |
| Snow/rain | 88.99% | 83.71% | 77.85% |

These numbers cannot be compared directly with the original local run's 15.91%
nuScenes mAP / 24.17% NDS. That completed baseline and its checkpoints remain intact.

## Findings and implemented changes

The original branches were independently extracted and fused only at their
outputs. The revised branches exchange joint features at strides 8, 16 and 32,
and feed them back into subsequent feature extraction. Gates use all modalities'
256-bin, per-channel entropy from 16x16 patches, and concatenate entropy with gated
features. Missing values are zero-filled in entropy histograms; measured coverage
is an additional input. The selected recipe drops at most one modality with probability 0.5 per
sample, preserving the original request. Independent dropout conditioned on
retaining one sensor remains an explicit alternative; the paper does not specify
this detail. Dropped LiDAR is also removed from the geometric depth reference.

LiDAR now carries depth, ego-frame height and intensity. Radar carries depth,
RCS and compensated radial velocity and repeats each nearest column return
vertically within the camera image. DHI and vertical radar follow the paper;
the radar attributes and normalization are explicit nuScenes adaptations.
Nearest-return rasterization keeps all attributes attached to the same point.

A top-down feature pyramid feeds six SSD detection scales. Smaller anchor sizes
address poor small-object coverage. A diagnostic on the same 96 validation views
and 455 targets found the fraction without an IoU>=0.5 anchor decreased from
38.24% to 16.70%; near-zero matches (<0.1 IoU) decreased from 10.55% to 0.44%.
This is anchor coverage, not a measured mAP improvement. Our 36 size/aspect
combinations are hand-selected; the paper uses 21 clustered combinations.

The baseline's mean translation error was 1.003m and mean velocity error 1.796m/s.
Car AP ranged from 1.04% at the strict 0.5m distance threshold to 57.18% at 4m.
The old log-depth and velocity/10 losses barely penalized physical errors.
The new head predicts a learned residual around measured local LiDAR log-depth,
with a 20m fallback when no measurement exists. It does not equate a measured
surface with an object's center. Added losses supervise camera-frame XYZ in
metres and velocity in m/s. A localization-quality score is learned from detached
center errors. Global ground-plane suppression is implemented as an alternative, but the
selected recipe retains image-IoU suppression because the mini diagnostic favored
it. Cross-view suppression still uses global ground-plane distance.

## Training recipe and limitations

`configs/paper_v2.json` selects the new architecture explicitly. Training uses
all six cameras, 384x640 inputs, ImageNet ResNet18 camera initialization, random
sensor branches, batch two, one optimizer update per batch (effective batch two), AMP,
AdamW at 2e-4 with 1,000-step warmup and a 20-epoch cosine schedule, weight decay
0.0005, and camera brightness/contrast augmentation. The added metric-center
loss weight ramps from 0.05 to 0.25 over the first 5,000 batches, giving early
classification gradients more influence. Workers use spawn and a
120-second loader timeout. Full-run validation remains after epochs 1, 5, 10,
15 and 20; best checkpoint selection uses mAP.

The paper instead uses half-width VGG branches, training from scratch, constant
learning rate, and its own sensor/dataset setup. Our depth reference, 3D losses,
quality scoring, top-down pyramid, optimizer/schedule and image augmentation are
nuScenes-specific design choices. This revision is not evidence that entropy
helps: `fusion_mode=concat` supplies a matching progressive-fusion control with
entropy inputs/gates removed. A controlled comparison remains necessary.

## Verification

The repeated-two-training-view, 80-update learning check reduced loss from 14.79
to 1.87 and positive-anchor mean center error from 8.49m to 1.35m. Every sensor
branch, exchange module and detection head received gradients; decoded outputs
were finite. This is a learning-mechanics check, not held-out accuracy. The model
has 15,730,860 parameters and this FP32 check peaked at 0.610 GiB allocated VRAM.
The initial 200-step AMP/accumulation check had no skipped optimizer updates,
used 0.636 GiB peak allocated VRAM, and averaged 0.033 seconds per two-view batch.
Checkpoint resume passed. The first pilot used independent multi-sensor dropout
and effective batch four; its three-epoch mAP was only 0.39%. Removing quality
calibration worsened it to 0.028%, so confidence calibration alone was not the
cause. At the same epoch-seven checkpoint and score threshold 0.001, image-IoU
suppression scored 2.65% mAP versus 1.84% for metric suppression. The classifier
was still biased towards background. That pilot was stopped, retaining artifacts,
and the selected recipe above was tested from fresh initialization.

Evaluation admits revised detections at confidence 0.001 before ranking and
suppression, since scores multiply class probability and localization quality.
The threshold and suppression settings are recorded in every evaluation's run.json.
This changes postprocessing compared with the original baseline; gains cannot be
attributed to entropy alone. All original pilot artifacts remain in
`outputs/paper_v2_mini_preflight_20260914/`.

The planned full runner checks the first validation result and stops if mAP is
below 2%, preserving the checkpoint for investigation instead of spending the
remaining epochs on a clearly degenerate model. This is a failure-detection floor,
not an accuracy target; the baseline to improve upon remains 15.91% mAP.

## Selected pilot and full run

The adjusted recipe completed five mini epochs (4,845 optimizer updates), with
no skipped optimizer updates. Official mini validation produced **4.02% mAP and
4.64% NDS**; car AP was 8.07%, pedestrian AP 7.59%, and cone AP 24.55%.
This demonstrates recovery from the degenerate initial pilot, not superiority to
the original 20-epoch mini run (4.76% mAP / 8.33% NDS). It is not directly
comparable to the original full-validation result either. Pilot artifacts are
`outputs/paper_v2_mini_recipe_check_20260914/`. The revised architecture must be
judged on the full validation split and controlled ablations.

Thirty tests passed. Full-data DHI checks passed on 24 train and 24 validation
views, including the previously crashing CAM_BACK sample; see the local
`outputs/paper_v2_full_data_check.json`. There were no new kernel faults.

The new full experiment uses `outputs/paper_v2_full_20260914/` and the persistent
user service `entropy-fusion-v2.service`. It starts from fresh ImageNet camera
weights and random sensor/exchange/head weights, not either mini checkpoint or
the old full-run weights. Twenty epochs cover 3,375,600 camera-view presentations,
1,687,800 two-view batches/optimizer updates. The original completed service is
disabled; its files and completed checkpoints are preserved.

```bash
systemctl --user status entropy-fusion-v2.service --no-pager
tail -n 1 outputs/paper_v2_full_20260914/training/metrics.jsonl
cat outputs/paper_v2_full_20260914/status.json
```

The service requires a working NVIDIA driver, uses automatic checkpoint resume,
and remains enabled across login/logout and reboot. It does not restart repeatedly
on failure. A checkpoint is saved every 1,000 batches and at epoch completion.
Validation runs after epochs 1, 5, 10, 15 and 20 and selects best.pt by mAP.
If epoch-one mAP is below 2%, the runner stops and records the reason; intervention
is required before continuing that failed experiment. Keeping the machine awake
and powered on is still necessary.

Launched September 14, 2026 at 14:43 CDT from commit `87bb07b`. Full training
passed 1,938 batches at startup verification, with zero skipped updates. The first
step-1,000 checkpoint reopened with finite weights and the selected recipe
(single-modality dropout, effective batch two, AMP, DHI inputs and image NMS).
GPU utilization was sampled at 75%, temperature 53 C. Local verification is in
`outputs/paper_v2_full_20260914/startup_verification.json`. These observations
confirm startup only; inspect live metrics and validation for current progress.
