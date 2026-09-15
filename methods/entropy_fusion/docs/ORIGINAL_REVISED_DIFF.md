# nuScenes original versus revised: ablation audit

**Both completed models already use entropy gating.** Their difference is not entropy-on versus entropy-off, and the observed gain cannot be attributed to entropy gating.

Audit evidence is the saved full-run configs, reports and evaluation metadata, plus historical source at `af8830d` and `87bb07b` and the precision repair at `86a139c`. The current `mini.json` and `paper_v2.json` templates default to mini; actual full-run overrides were read from saved configs. No training or model changes were made for this audit.

## Completed results

| Run | Selected epoch | mAP | NDS |
|---|---:|---:|---:|
| Original |20|15.91%|24.17%|
| Revised best |15|21.96%|28.60%|
| Revised final |20|21.62%|28.49%|

The best-to-best difference is +6.04 percentage points mAP and +4.43 NDS; epoch-20 versus epoch-20 is +5.71 mAP and +4.32 NDS. Neither comparison isolates gating.

## Every substantive recipe difference

| Area / setting | Original | Revised |
|---|---|---|
| input: sensor channels | One channel per sensor: camera-Z/100 | Three channels per sensor: LiDAR Z/100, clipped (ego-height+3)/8, clipped intensity/255; radar Z/100, clipped (RCS+40)/80, clipped (compensated radial velocity+30)/60 |
| input: radar rasterization | Sparse nearest return per image pixel | Nearest radar return per image column, repeated vertically inside camera image mask |
| entropy mechanism: histogram features | 16 bins; camera luminance and one sensor depth channel; unmeasured pixels excluded from histogram; one entropy plus one coverage channel per modality | 256 bins per each of three channels; zero-filled missing measurements included; three entropy plus one coverage channel per modality |
| entropy mechanism: gates | One independent 2-channel entropy/coverage convolution per sensor at each of 3 scales | One joint 12-channel all-sensor entropy/coverage convolution per scale; entropy features also concatenated with gated features |
| architecture: exchange location | Backbones run independently through all stages; only completed output feature maps fused | Fused context feeds back via learned 1x1 projections to each available stream after each stage, residual scale 0.1 |
| architecture: sensor backbone first convolution | 1 input channel; 293,712 parameters each | 3 input channels; 294,000 parameters each |
| architecture: fused feature width | 96 | 128 |
| architecture: head | 3 scales; one shared 96-channel tower; no top-down pyramid; 254,406 parameters | Top-down lateral/smoothing FPN on 3 scales, 3 extra downsampled scales, 6 separate 128-channel towers; 2,057,292 parameters |
| architecture: anchor bases as fraction of min(image_height,image_width) | [0.1, 0.25, 0.5] | [0.04, 0.09, 0.18, 0.32, 0.55, 0.8] |
| architecture: depth initialization and input prior | Direct log-depth code initialized to log(20) for all anchors | Learned log-depth residual initialized to 0, plus measured mean log-depth in central half of anchor; fallback log(20); prior removed when LiDAR dropped |
| architecture: localization quality head | Absent | Added scalar per anchor, trained from detached exp(-center_error/2), also used for score ranking |
| architecture: total trainable parameters | 13,205,670 | 15,730,860 |
| loss: hard-negative ratio | 3 | 5 |
| loss: 3D code velocity terms | SmoothL1 on velocity/10 when valid | Velocity terms weight 0 inside base boxes3d loss; replaced with metric velocity SmoothL1 in m/s weighted 0.1 |
| loss: physical camera-center loss | Absent | Added XYZ-metre SmoothL1(beta 1) with weight 0.05->0.25 linearly over 5,000 microbatches |
| loss: quality loss | Absent | Added BCE on positives and same hard negatives, weight 0.5 |
| optimization: batch and microbatch count | batch 1, accumulation 1: 3,375,600 microbatches over 20 epochs | batch 2, accumulation 1: 1,687,800 microbatches over 20 epochs; skipped AMP updates possible |
| optimization: learning rate | AdamW 1e-4 constant | AdamW peak 2e-4; 1,000-microbatch linear warmup; cosine over 20 epochs to 5% of peak (1e-5) |
| optimization: weight decay | 0.01 | 0.0005 |
| optimization: numeric precision | FP32 throughout; AMP disabled | AMP enabled; first 6,000 retained steps before FP32 camera/exchange recovery; resumed remainder uses FP32 camera/exchange and FP16-autocast sensor/head; losses FP32 |
| augmentation: camera photometric | None | Per-sample independent brightness and contrast uniform [0.8, 1.2], clipped [0, 1], masked to camera image |
| runtime: workers | 2 workers with platform-default fork; later timeout 120 s recovery | 2 workers spawn, 120 s timeout |
| evaluation: score threshold | 0.05 | 0.001 |
| evaluation: confidence scoring | Foreground softmax probability | Foreground probability times sigmoid localization quality |
| evaluation: reported best checkpoint | Epoch 20 selected by validation mAP: 15.912606% mAP, 24.168525% NDS | Epoch 15 selected by validation mAP: 21.955672% mAP, 28.600194% NDS; epoch 20: 21.619519% mAP, 28.487799% NDS |
| runtime history: mid-run code/environment changes | Started af8830d; projection-boundary repair c3bc5bc; resume plumbing 328cb53; later kernel/driver repair with current driver 580.173.02 reported in docs | Started 87bb07b; loss nonfinite at 6,592; restored step-6,000 checkpoint using precision fix 86a139c; 1 skipped AMP update documented in initial resumed 1,309 steps |

Source functions and exact values are recorded per row in `../outputs/ablation_audit_20260915/config_diff.json`.

## Shared controls already present

* Same nuScenes train/val scenes, ten classes, six independently sampled camera views, 384x640 letterboxed inputs, 20 epochs, ImageNet ResNet18 camera and random sensor/head initialization.
* Same 168,780 camera-view items/epoch (28,130 keyframes), 3,375,600 view presentations over 20 epochs; validation 36,114 views (6,019 keyframes). These are camera-view passes, not 6-camera joint batches.
* Same at-most-one-sensor dropout probability 0.5, seed 42, accumulation 1, AdamW family, gradient clip 10, base 3D code coordinates, forced anchor assignment and attribute weight 0.2. Optional revised independent dropout was not used in the completed full run.
* Same per-view image IoU NMS at 0.5 (revised metric_suppression=false), top 100 from 1,000 candidates, cross-view class-distance merging to 500/sample, official detection_cvpr_2019 evaluation and validation epochs 1, 5, 10, 15, 20. Revised scores and score cutoff differ as above.
* Camera BN running statistics frozen in both; camera affine parameters train. No CBGS in either; one current LiDAR record and one current record per radar, no temporal sweep accumulation.

## The existing concat switch is not a strict gating-only control

Existing `paper_v2/entropy` has 15,730,860 parameters; `paper_v2/concat` has 15,542,892, which is 187,968 fewer. It removes both entropy gate convolutions and entropy-concatenation channels, altering project convolutions and random initialization of later modules. Same seed alone does not match shared weights or training random draws.

For the requested gating-only pair, preserve all modules, initial tensors, entropy concatenation and data/loss/inference paths and toggle only multiplication by learned sigmoid gates versus multiplication by one. State clearly that this measures multiplicative gating conditional on retaining entropy features; a separate entropy-information ablation would answer a different question.

## Reproducibility limits

* Same seed in historical runs does not align random draws because architecture, batch shape, augmentation and precision differ.
* Historical runs were interrupted and altered under distinct code/software regimes; cannot retroactively make them controlled experiments.
* One paired seed gives no seed-variance uncertainty; multiple matched seeds are needed to support a robust effect.
* GPU reductions/kernel behavior and AMP overflow may differ between gate variants even when data, RNG and configs match; record skipped updates and full software/hardware hashes.
* Fixed final epoch 20 should be primary endpoint; choosing each best epoch adds validation selection variance. Report best checkpoint only secondary.

Do not reuse either completed checkpoint to initialize a supposedly from-scratch controlled arm. Pair the initialization, data order, stochastic augmentation/dropout draws and validation protocol. Freeze a single code revision and environment for both arms; report fixed final epoch 20 as the primary endpoint and per-seed paired differences if repeated.
