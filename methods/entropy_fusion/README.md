# Entropy-steered 3D fusion

This method combines camera images with LiDAR and radar projected into the image
plane to detect nuScenes objects in 3D. It is inspired by Mario Bijelic, Tobias
Gruber, Fahim Mannan, Florian Kraus, Werner Ritter, Klaus Dietmayer, and Felix
Heide’s CVPR 2020 paper, [Seeing Through Fog Without Seeing Fog](https://arxiv.org/abs/1902.08913).
Our three-modality, camera-ray 3D baseline is an adaptation of their ideas, with
no gated NIR stream and no claim of reproducing their reported results.

An ImageNet-pretrained ResNet18 processes RGB, while smaller CNNs process sparse
LiDAR and radar depth maps. Masked patch entropy and sensor coverage steer feature
gates at three resolutions. An SSD-style head predicts classes, image boxes, metric
3D centers and dimensions, orientation, velocity, and attributes. Per-sample
modality dropout encourages the detector to use complementary inputs.

The project supports training checkpoints, resume, mixed precision, gradient
accumulation, and official nuScenes 3D evaluation after merging detections from
all six camera views. Development uses the ten-scene nuScenes-mini split; raw
data and checkpoints remain outside Git. A [20-epoch mini experiment](docs/MINI_EXPERIMENT.md) completed successfully,
with 4.76% validation mAP. Held-out accuracy remains low, and adverse-weather
robustness has not been established.

See the [architecture and design choices](docs/MODEL_PLAN.md),
[data contract](docs/DATA_PIPELINE.md), [training instructions](docs/TRAINING.md),
and [measured checks](docs/PRECHECK_RESULTS.md). [TransFusion](../transfusion/)
is the separate baseline method in this research workspace.
