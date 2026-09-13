# Next milestone: model design (not implemented)

The requested design is inspired by Bijelic et al.,
[Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather](https://openaccess.thecvf.com/content_CVPR_2020/html/Bijelic_Seeing_Through_Fog_Without_Seeing_Fog_Deep_Multimodal_Sensor_Fusion_CVPR_2020_paper.html),
CVPR 2020. Authors: Mario Bijelic, Tobias Gruber, Fahim Mannan, Florian Kraus,
Werner Ritter, Klaus Dietmayer, and Felix Heide. This is a proposed nuScenes
camera/LiDAR/radar adaptation, not an exact reproduction of their sensor setup.

Planned modules (currently docstring-only placeholders):

- `models/backbone.py`: ImageNet-pretrained ResNet18 for RGB; smaller randomly
  initialized CNNs for the single-channel LiDAR/radar depth inputs.
- `models/entropy.py`: per-modality 16×16-patch Shannon entropy. Define histogram
  bins and treatment of sparse no-return pixels/padding before implementing;
  missing data must not be confused with low-entropy measured data.
- `models/fusion.py`: entropy-conditioned sigmoid convolution gates and feature
  concatenation at several backbone resolutions.
- `models/head.py`: SSD-style multi-scale anchors, classification, and regression.
- `train.py`: cross-entropy plus smooth-L1 and stochastic modality dropout.
  Clarify whether p=0.5 selects one random modality per sample or applies
  independently to each modality (which can drop all three); specify how masks,
  entropy inputs, and gated features respond. No claim of exact paper equivalence
  is made until the schedule and mechanism are checked against its implementation.

## Resolve the output task before implementing the head

[Standard nuScenes detection evaluation](https://www.nuscenes.org/object-detection)
is **3D**, using global-coordinate boxes and ground-plane center-distance matching,
not image-plane SSD IoU mAP. Official submission records also carry orientation,
size, velocity, class, score, and attributes. `eval.py` therefore deliberately fails
with an explanatory message instead of silently reporting 2D AP as nuScenes AP.

Choose either an explicitly separate 2D projected-box benchmark, or extend the
head with 3D decoding/regression and a global-coordinate prediction adapter. The
latter must reconcile duplicate predictions across camera views and produce
sample-token-indexed outputs for official evaluation (`mini_val` during development).
Camera-view targets already retain their matching global 3D annotation fields.

Do not load pretrained weights, build model classes, run losses, train, or report
mAP in the data-pipeline milestone. The next implementation starts after review
of the verified batch and the output-task decision.
