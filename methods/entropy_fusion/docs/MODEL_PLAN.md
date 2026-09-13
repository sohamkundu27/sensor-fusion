# Implemented 3D baseline

This implementation adapts the entropy-guided fusion idea from Bijelic et al.,
[Seeing Through Fog Without Seeing Fog](https://arxiv.org/html/1902.08913v3), CVPR 2020.
It is not a reproduction of the original architecture, training recipe or adverse-
weather results. The paper uses local measurement entropy, stream dropping and
single-shot detection; this project adds a camera-ray 3D head for nuScenes.

## Inputs and fusion

One item is one camera view at a keyframe, letterboxed to 640 x 384. Each view uses
the current LiDAR cloud and all five current radar clouds projected with calibrated
sensor poses. There is no temporal sweep accumulation or radar-velocity input.
The six views are processed independently with shared weights.

The camera stream is torchvision ResNet18 with ImageNet-1K V1 weights and ImageNet
channel normalization, retaining the detector image geometry rather than applying
classification crops. BatchNorm running statistics stay frozen for small batches;
affine parameters and convolutions train. Two randomly initialized compact CNNs
use GroupNorm for LiDAR and radar. Feature strides are 8, 16 and 32.

Each 16 x 16 input patch produces normalized Shannon entropy from a 16-bin
histogram and an observed-pixel coverage channel. RGB uses luminance; sensor depth
uses camera-Z divided by 100 m. Missing returns and padding do not enter histogram
counts. A measured constant patch has zero entropy but positive coverage; an empty
patch has both zero. Quantization, masked normalization, and the coverage channel
are explicit design choices, distinct from the paper's 8-bit input representation.

At each resolution, a convolution of entropy plus coverage produces channel-wise
sigmoid gates. Gated modality features are concatenated and projected to 96 channels.
The backbones remain separate; these fusion outputs feed the multi-scale detector
and are not fed back into the individual backbone stages.

During training, with probability 0.5 per view, one of three modalities is selected
uniformly for dropping. Its input, entropy/coverage, and gate availability become
zero. A naturally absent selected stream makes this a no-op; the final available
stream is never deliberately removed. This resolves the original dropout ambiguity
without allowing deliberate all-stream dropout. Evaluation disables random dropping.

## Anchors, targets, and losses

Six anchors per cell combine aspect ratios 0.5/1/2 with two size multipliers, 1 and sqrt(2). Base sizes are 0.1, 0.25,
and 0.5 times the shorter image dimension at strides 8/16/32. At the default input
size this produces 30,240 anchors. Padding-centered anchors are ignored.

The shared head predicts 11 class logits (including background), four image-box
regression values, ten 3D values and eight attribute logits per anchor. The 3D code
contains projected center offsets relative to the anchor, log camera depth,
log width/length/height, sine/cosine of heading relative to the camera's horizontal
viewing direction, and velocity in that same horizontal frame divided by 10.
Decoding uses intrinsics and poses to return global centers, orientation quaternions
and velocity. Boxes are upright in the camera-time ego frame, so pitch/roll are not
independently regressed. Depth decoding is limited to 1–100 m and dimensions to
0.1–30 m. These bounds and the 3D parameterization are baseline design choices.

Targets come from the matched original global annotations; centers outside 1–100 m
camera depth and boxes without any annotated LiDAR/radar points are excluded.
Unknown velocity components and absent attributes are masked in their losses.
Image targets remain projections of 3D boxes, not hand-labeled 2D silhouettes.

IoU >= 0.5 marks positive anchors, < 0.4 marks negatives, and the gap is ignored.
Greedy unique forced matches ensure valid ground-truth boxes receive distinct
positive anchors. Cross-entropy uses 3:1 hard negative mining; empty scenes use up
to 100 background anchors with a separate normalization. Smooth-L1 supervises
image boxes and the 3D code. Attribute cross-entropy has weight 0.2; the other three
loss terms have weight 1. This differs from the paper's 5:1 mining recipe.

## Training and evaluation

AdamW defaults to constant LR 1e-4 and weight decay 0.01, with gradient clipping at
10. No class-balanced resampling, scheduler, or image/weather augmentation is used
in this initial baseline. FP16 uses dynamic loss scaling with initial scale 1024;
overflowed updates are logged and skipped, with repeated overflow raising an error.
Accumulation normalizes each window, including a final partial window.

Checkpoints include model, optimizer, scaler, RNG states and an epoch/batch cursor.
The loader uses an epoch-seeded shuffle. Resume requires matching training settings;
it is tested with the current deterministic loader, not arbitrary future random
worker augmentations. Format version 1 identifies the camera-relative 3D encoding.

Per-view class-aware image NMS is followed by same-class distance suppression in
global XY across cameras, capped at 500 predictions per sample. The merge radii
are explicit constants in `models/predictions.py`; close same-class objects may be
merged incorrectly and this needs validation. Attributes are restricted to valid
classes. Every evaluation sample has a result entry, including empty detections.
Full-split exports are passed to the [official nuScenes detection evaluator](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md)
for 3D mAP and NDS. Partial export checks deliberately do not report official metrics.

This establishes executable training and evaluation, not useful detection accuracy.
Depth accuracy, rare classes, multi-view merging, modality ablations and adverse-
weather robustness all require proper experiments. Compare methods on identical
splits with each method's inputs, sampling and compute unit explicitly reported.
