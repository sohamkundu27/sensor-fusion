# sensor-fusion

A research workspace for comparing object-detection and sensor-fusion methods on autonomous-driving datasets. Each method has its own folder so we can develop and evaluate different approaches in the same repository.

## Datasets

### nuScenes

[nuScenes](https://www.nuscenes.org/) contains 1,000 driving scenes recorded with six cameras, one LiDAR, and five radars, with annotated 3D objects. We use its camera, LiDAR, and radar measurements to explore how complementary sensors can improve object detection. The 10-scene mini split is our starting point for verifying data loading and sensor alignment before moving to the full training and validation splits.

### Waymo Open Dataset

The [Waymo Open Dataset Perception dataset](https://waymo.com/open/) provides camera images, LiDAR measurements, and object annotations for autonomous-driving research. We have a small subset of version 1.4.3: five training segments and two validation segments in TFRecord format, totaling approximately 6.4 GiB. This subset supports pipeline testing; it is not a representative evaluation benchmark.

### Seeing Through Fog

[Seeing Through Fog](https://github.com/princeton-computational-imaging/SeeingThroughFog) contains synchronized camera, gated near-infrared, LiDAR and radar measurements in clear weather, fog, snow and rain, with object annotations and scene metadata. We are preparing it to evaluate entropy fusion in the setting used by Bijelic and colleagues. Dataset access is pending registration, and the published split files need overlap checks before a reproduction experiment.

## Methods

### TransFusion

[TransFusion](methods/transfusion/) is a transformer-based approach to LiDAR-camera 3D object detection introduced by Xuyang Bai and colleagues at CVPR 2022. It uses LiDAR features to initialize object queries and camera features to refine detections through attention. We are preparing the [original implementation](https://github.com/XuyangBai/TransFusion) as a baseline, including its LiDAR-only and camera-fusion variants; its environment and a short LiDAR-only training benchmark are verified, while full training remains pending.

### Entropy-steered multi-modal fusion

Our [entropy-steered fusion method](methods/entropy_fusion/) is inspired by Mario Bijelic and colleagues’ CVPR 2020 paper, [Seeing Through Fog Without Seeing Fog](https://arxiv.org/abs/1902.08913). It combines camera images with LiDAR and radar projected onto the image plane, using local input entropy and sensor coverage to guide feature fusion at multiple resolutions. This three-modality adaptation uses a lightweight SSD-style head extended to predict 3D boxes, velocity, and attributes, with official nuScenes evaluation across all six camera views. The revised full run achieved 21.96% mAP and 28.60% NDS, compared with 15.91% and 24.17% for the first implementation. A four-modality Seeing Through Fog experiment is being prepared to check fidelity to the original method; entropy's contribution still needs a controlled comparison.
