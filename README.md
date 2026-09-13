# sensor-fusion

A workspace for comparing object-detection and sensor-fusion methods on nuScenes
and Waymo. Each method has its own folder, dependencies, documentation, and experiments.

| Method | Folder | Current status |
| --- | --- | --- |
| TransFusion | [methods/transfusion](methods/transfusion/README.md) | Environment and CUDA checks verified; upstream model configs available |
| Entropy-steered camera/LiDAR/radar fusion | [methods/entropy_fusion](methods/entropy_fusion/README.md) | nuScenes-mini data pipeline verified; model implementation pending |

```text
methods/
├── transfusion/
└── entropy_fusion/
```

Raw datasets are shared outside the repository under `~/data/`. Each method keeps
its own environment; run commands from the corresponding method directory.

## TransFusion

```bash
cd ~/sensor-fusion/methods/transfusion
source scripts/activate_transfusion.sh
python scripts/check_environment.py
```

The original TransFusion LICENSE and paper attribution remain in its method folder.

## Entropy-steered fusion

```bash
cd ~/sensor-fusion/methods/entropy_fusion
source ~/venvs/entropy-fusion/bin/activate
python -m pytest -q
python -m scripts.verify_batch --workers 2
```

These commands verify the existing setup and data pipeline; they do not train models.
