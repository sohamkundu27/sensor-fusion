# Phase 3: reference comparison audit — 15 September 2026

This compares our revised nuScenes run (`paper_v2_full_20260914`, epoch 15,
21.955672% mAP / 28.600194% NDS) against the BEVFusion (ADLab/Liang et al.)
fusion-stage recipe. Reference configs were fetched verbatim and hashed in
`outputs/reference_audit_20260915/sources.json`. No training or model changes
were made for this audit.

## What the reference fusion stage actually does

Reference stage 1, LiDAR-only pretraining (`transfusion_nusc_voxel_L.py` on
`nus_tf.py`), 20 epochs, batch 2/GPU x 8 GPUs, AdamW 1e-4, wd 0.01:

| Setting | Value |
|---|---|
| Dataset wrapper | `CBGSDataset` (class-balanced resampling) |
| `LoadPointsFromMultiSweeps` | `sweeps_num=10` |
| `GlobalRotScaleTrans` | `rot_range=[-0.785, 0.785]`, `scale_ratio_range=[0.9, 1.1]`, `translation_std=[0.5, 0.5, 0.5]` |
| `RandomFlip3D` | `flip_ratio_bev_horizontal=0.5`, `flip_ratio_bev_vertical=0.5` |
| `ObjectSample` / `db_sampler` | present but **commented out** |
| Voxel grid | `[1440, 1440, 40]` |

Reference stage 2, fusion (`bevf_tf_4x8_6e_nusc.py` on `nusc_tf.py`), 6 epochs,
batch 4/GPU x 8 GPUs (effective 32), AdamW 1e-3, wd 0.05, LR steps [4, 5],
`load_from = work_dirs/transfusion_nusc_voxel_L/epoch_20.pth`,
`freeze_img=True`, `freeze_lidar_components=True`, `no_freeze_head=True`:

| Setting | Value at the fusion stage |
|---|---|
| Dataset wrapper | plain `NuScenesDataset` — **no CBGS** |
| `LoadPointsFromMultiSweeps` | `sweeps_num=10` — **retained** |
| `GlobalRotScaleTrans` | **absent** |
| `RandomFlip3D` | **absent** |
| `ObjectSample` | **absent** |
| Remaining train pipeline | `PointsRangeFilter`, `ObjectRangeFilter`, `ObjectNameFilter`, `PointShuffle`, `MyResize`, `MyNormalize`, `MyPad` |

This confirms the earlier finding rather than overturning it: the fusion stage
itself uses **no CBGS and no geometric (rot/scale/trans/flip) augmentation**. Its
only train-time stochasticity is `PointShuffle`. The geometric augmentation lives
entirely in the frozen LiDAR pretraining that the fusion stage loads.

The optional augmented variant (`bevf_tf_4x8_10e_nusc_aug.py` on
`nusc_tf_aug.py`, 10 epochs, LR 1e-4, `load_from = models/lidar_tf.pth`) does add
BEV augmentation at the fusion stage — `GlobalRotScaleTransBEV(resize_lim=(0.9,
1.1), rot_lim=(-0.785, 0.785), trans_lim=0.5)` and `RandomFlip3DBEV`, with a
`lidar_aug_matrix` collected so the camera branch can be de-augmented. It still
uses no CBGS and no `ObjectSample`. So augmentation is a variant switch at the
fusion stage, not the headline recipe.

**Therefore "we are missing augmentation" does not explain our gap.** The
differences that are real at the fusion stage are: 10-sweep temporal point
accumulation, a 20-epoch CBGS-balanced LiDAR-pretrained and frozen BEV backbone
at 1440x1440 voxel resolution, effective batch 32, and a BEV-space detection
head. We have none of these: one current LiDAR record and one current record per
radar with no sweep accumulation, no pretraining of any sensor branch, batch 2,
and image-plane detection at 384x640 with regressed depth.

## Our per-class AP and TP errors (epoch 15)

| Class | AP % | mATE | mASE | mAOE | mAVE | mAAE |
|---|---:|---:|---:|---:|---:|---:|
| car | 37.34 | 0.620 | 0.189 | 0.255 | 1.773 | 0.218 |
| truck | 13.72 | 0.888 | 0.278 | 0.442 | 1.659 | 0.292 |
| bus | 16.05 | 1.011 | 0.275 | 0.442 | 2.894 | 0.335 |
| trailer | 2.56 | 1.248 | 0.356 | 1.005 | 0.873 | 0.129 |
| construction_vehicle | 3.12 | 1.193 | 0.496 | 1.501 | 0.333 | 0.386 |
| pedestrian | 34.59 | 0.691 | 0.318 | 1.246 | 0.917 | 0.229 |
| motorcycle | 18.27 | 0.771 | 0.370 | 1.050 | 2.013 | 0.110 |
| bicycle | 16.44 | 0.685 | 0.365 | 1.164 | 0.692 | 0.032 |
| traffic_cone | 40.10 | 0.668 | 0.386 | n/a | n/a | n/a |
| barrier | 37.38 | 0.650 | 0.365 | 0.444 | n/a | n/a |

Aggregate TP errors: mATE 0.8424, mASE 0.3399, mAOE 0.8390, mAVE 1.3943,
mAAE 0.2164. The corresponding nuScenes TP scores are 0.1576, 0.6601, 0.1610,
**0.0000**, 0.7836.

NDS decomposition reconstructs exactly: (5 x 0.219557 + 1.7623) / 10 = 0.286002.

## Error profile: same shape scaled down, or qualitatively different?

**Qualitatively different.** Four independent signatures, all from our own
numbers:

1. **Velocity is not merely worse, it is forfeited.** mAVE 1.3943 clips to a TP
   score of exactly 0.0, so velocity contributes none of its 10% of NDS. Car
   1.773, bus 2.894 and motorcycle 2.013 m/s are worse than predicting zero
   velocity for most objects. This is not undertraining: we feed one current
   LiDAR record and one current radar record with no sweep accumulation, so
   there is no temporal baseline from which velocity is recoverable. The
   reference keeps `sweeps_num=10` at the fusion stage precisely for this. A
   scaled-down-but-same-shape model would still score something here.

2. **The class ranking is inverted relative to any LiDAR-centric BEV detector.**
   Our best classes are traffic_cone (40.10) and barrier (37.38), *above* car
   (37.34). LiDAR-BEV models rank car/bus/truck at the top and cone/barrier in
   the middle. Cone and barrier are small, static, dense, near the ego vehicle,
   and exempt from orientation and velocity scoring — exactly the classes an
   image-plane detector with regressed depth can win. This is the profile of a
   camera detector with sensor side-channels, not of a fusion model scaled down.

3. **Scale is healthy while translation and orientation are not.** mASE 0.3399
   (score 0.660) and mAAE 0.2164 (score 0.784) are in a normal range, but mATE
   0.8424 and mAOE 0.8390 score 0.158 and 0.161. Appearance-derived quantities
   work; metric-geometry quantities do not. This is the **same dissociation
   Phase 1 measured on KITTI**, where median dimension ratios were 1.01/0.99/1.00
   while substituting GT position alone lifted 3D IoU>=0.7 from 16.32% to 62.97%.
   Two datasets, two codebases, one failure mode: the box is the right size and
   in the wrong place.

4. **Large, rare, extent-dominated classes collapse.** trailer 2.56 and
   construction_vehicle 3.12 with mATE 1.248/1.193 and mAOE 1.005/1.501. These
   are the classes that require reasoning about metric extent in BEV, and they
   are the ones CBGS resampling is designed to protect during LiDAR pretraining.

Caveat, stated plainly: BEVFusion's own published per-class TP error table is
not among the fetched artifacts and could not be verified offline, so the
comparison above is made against the *structure* the reference recipe implies
(temporal sweeps, CBGS-pretrained BEV backbone) and against our own internal
dissociations, not against quoted reference TP numbers. Rows 1-4 do not depend
on the reference table; a claim about the exact size of each per-class gap would.

## Verdict

**Underperforming is the wrong frame, and so is on track.** For the budget we
actually spent — 20 epochs from scratch, batch 2, no pretrained sensor branch,
no temporal sweeps, image-plane detection — 21.96% mAP / 28.60% NDS is about
what this configuration can produce, and more epochs of the same configuration
will not close the distance to 69.2/67.9.

The gap to the reference is **architectural and input-level, not a training
budget shortfall**:

* ~10% of NDS is structurally unavailable to us (velocity, score 0.0) because we
  have no temporal input. No amount of training recovers it.
* A further ~17% of NDS is lost to translation and orientation (scores 0.158 and
  0.161), which Phase 1 independently traced to scattered position/depth error
  rather than a coordinate or calibration bug.
* The reference's 69.2/67.9 is a 6-epoch fine-tune on top of a 20-epoch
  CBGS-balanced, 10-sweep, 1440x1440-voxel LiDAR detector that is then frozen.
  Nearly all of its geometric competence is inherited, not learned at the fusion
  stage. Comparing our from-scratch fusion number against it compares different
  quantities.

So: our number is not evidence that entropy fusion fails, and it is not evidence
that it works. It is evidence that we are measuring a camera-led image-plane
detector against a LiDAR-led BEV detector. The honest comparison would need a
camera-plus-sensor baseline trained under our own budget — which is exactly what
the Phase 2 gating ablation provides.
