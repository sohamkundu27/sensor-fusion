# sensor-fusion

Camera, projected LiDAR, and projected radar data for an entropy-steered
multi-modal object detector on nuScenes.

Inspired by **Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor
Fusion in Unseen Adverse Weather** (Bijelic et al., CVPR 2020):
[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bijelic_Seeing_Through_Fog_Without_Seeing_Fog_Deep_Multimodal_Sensor_Fusion_CVPR_2020_paper.html).
This project adapts the idea to three nuScenes modalities; it does not use gated NIR.

## Current milestone: data pipeline only

The mini dataloader projects sensors through their timestamp-specific ego poses
into each camera image, produces sparse depth maps and validity masks, and returns
projected boxes plus matching original 3D annotations. It uses official scene
splits, with no train/validation scene overlap. No preprocessing info files are required.

```text
data/                     # Loader, geometry, batching; source code, not raw data
models/                   # backbone/entropy/fusion/head placeholders only
scripts/verify_batch.py   # Real mini batch + official projection comparison + overlay
train.py, eval.py          # Explicit not-implemented entry points
tests/                   # Synthetic geometry tests
legacy/transfusion/       # Previous project, including its unchanged original LICENSE
```

Model implementation, pretrained weight downloads, entropy gates, modality dropout,
training, and mAP evaluation are **not implemented or run** in this milestone.
Verification passed on a two-item front-camera training batch and a six-view
validation batch, including GPU transfer and official-devkit projection comparisons.
Four synthetic geometry tests passed. [Recorded results](docs/verification.json).

See the [data contract](docs/DATA_PIPELINE.md) and [model plan](docs/MODEL_PLAN.md).

## Run on this machine

The separate environment is `~/venvs/entropy-fusion` (Python 3.12, PyTorch
2.9.1+cu128, torchvision 0.24.1+cu128, nuScenes devkit 1.2.0, NumPy 1.26.4).
The previous environments are unchanged.

```bash
cd ~/sensor-fusion
source ~/venvs/entropy-fusion/bin/activate
python -m pytest -q
python -m scripts.verify_batch --root ~/data/nuscenes --workers 2
```

This reads one batch of two front-camera items from `mini_train` and saves an
image overlay and JSON report under `outputs/mini_batch/`. It does not train.
Use `--split mini_val` for validation and `--cameras` to select any of the six views.

Raw data stays outside Git:

```text
~/data/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/
```

On another machine with a CUDA 12.8-compatible NVIDIA driver:

```bash
python3.12 -m venv ~/venvs/entropy-fusion
source ~/venvs/entropy-fusion/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The loader also works on CPU. For CPU-only use install the matching PyTorch wheels
from the CPU index instead. Raw data, generated overlays, weights, and environments
are ignored by Git. Data-loader code under `data/` is tracked.

## Evaluation decision still required

An SSD-style **2D** head alone cannot produce standard nuScenes **3D detection mAP**.
Before implementing the head, choose a 2D benchmark or add 3D predictions and the
required global-coordinate export. Both 2D and matching 3D targets are retained.

## Previous work

The complete previous project is in [legacy/transfusion](legacy/transfusion/README.md).
Its original LICENSE and configs are preserved, and the existing editable install
was repointed locally. Run its commands from `~/sensor-fusion/legacy/transfusion`.
The legacy environment/model checks passed after relocation. The relocation was
committed and pushed before starting this new pipeline.
