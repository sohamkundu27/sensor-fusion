# sensor-fusion

Research workspace for LiDAR-camera 3D object detection, based on
[TransFusion](https://github.com/XuyangBai/TransFusion).

## Original work and credit

This project builds on **TransFusion: Robust LiDAR-Camera Fusion for 3D Object
Detection with Transformers** (CVPR 2022), by **Xuyang Bai, Zeyu Hu, Xinge Zhu,
Qingqiu Huang, Yilun Chen, Hongbo Fu, and Chiew-Lan Tai**.
See the [paper](https://arxiv.org/abs/2203.11496) and
[original repository](https://github.com/XuyangBai/TransFusion).
The original [LICENSE](LICENSE) is preserved without changes. The imported source
is based on upstream commit `73c596f7bd3460c17cbcc58dd9bcc5a0896774a8`;
[provenance and config checksums](research/upstream.json) record that snapshot.
The upstream Git history was removed from the staging clone; this repository
retains its own existing history. The
[original README](docs/TRANSFUSION_UPSTREAM_README.md) is retained for reference.

## Project notes

Preparation only: no nuScenes conversion or training has been started.
Upstream model configs remain unchanged. See:

- [Environment setup and validation](docs/ENVIRONMENT.md)
- [Single RTX 3080 changes to consider](docs/RTX3080_PLAN.md)
- [nuScenes preparation and launch checklist](docs/NUSCENES_READY.md)
- [Existing dataset notes (Waymo and nuScenes mini)](docs/WAYMO_SUBSET.md)

The active checkout is `~/sensor-fusion/methods/transfusion`, the isolated environment is
`~/venvs/transfusion`, and `data/nuscenes` links to `~/data/nuscenes` locally.
Datasets, checkpoints, compiled binaries, and environments are not committed.
Old upstream CI/release workflows are archived under `docs/upstream-workflows/`;
they use a retired runner and are not enabled as this project's automation.

Add research objectives, experiment notes, and results below this section.
