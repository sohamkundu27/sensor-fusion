# Entropy-steered 3D fusion

This method combines camera images with LiDAR and radar projected into the image
plane to detect nuScenes objects in 3D. It is inspired by Mario Bijelic, Tobias
Gruber, Fahim Mannan, Florian Kraus, Werner Ritter, Klaus Dietmayer, and Felix
Heide’s CVPR 2020 paper, [Seeing Through Fog Without Seeing Fog](https://arxiv.org/abs/1902.08913).
Our three-modality, camera-ray 3D baseline is an adaptation of their ideas, with
no gated NIR stream and no claim of reproducing their reported results.

The revised model exchanges entropy-steered features between sensor branches
throughout feature extraction. It uses LiDAR depth, height and intensity; radar
depth, reflectivity and compensated velocity; and a six-scale SSD head with
metric 3D supervision. The original completed full nuScenes baseline achieved
15.91% mAP and 24.17% NDS. The [paper-informed revision](docs/REIMPLEMENTATION.md)
achieved 21.96% mAP and 28.60% NDS at its best validation checkpoint, epoch 15.
Entropy's contribution and adverse-weather robustness still require controlled
evaluation. The [Seeing Through Fog experiment](docs/SEEING_THROUGH_FOG.md) is
being prepared with the paper's four sensor modalities; full dataset access is
pending registration and the public split overlaps need resolution.

See the [architecture and design choices](docs/MODEL_PLAN.md),
[data contract](docs/DATA_PIPELINE.md), [training instructions](docs/TRAINING.md),
and [measured checks](docs/PRECHECK_RESULTS.md). [TransFusion](../transfusion/)
is the separate baseline method in this research workspace.
