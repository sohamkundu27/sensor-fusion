# Single-GPU feasibility triage: 7 camera/LiDAR/radar fusion methods

Generated 2026-09-17 22:28 UTC by `make_summary.py`.
GPU: NVIDIA RTX 3080, 12288 MiB, driver 580.173.02 / CUDA 13.0.
All timings taken at batch size 1, 20-50 iterations, on an otherwise idle GPU
(the gating ablation had finished). Timing only -- no accuracy or convergence claim.

| method | ran | VRAM @ batch 1 | sec/iter | est. hrs/epoch (mini) | est. hrs/epoch (full nuScenes) | biggest blocker |
|---|---|---|---|---|---|---|
| Deep Entropy Fusion (Seeing Through Fog) | N | — | — | — | — | Official repo has no model/training code (dataset toolkit only); STF data not downloaded and registration-gated |
| RAF (Reliability-Aware Fusion) | N | — | — | — | — | Frozen pretrained L4DR/3D-LRF backbone never distributed (author-local paths only); no VoD loader in released code |
| SAMFusion | ? | — | — | — | — | — |
| BEVFusion | ? | — | — | — | — | — |
| RCBEVDet | ? | — | — | — | — | — |
| TransFusion | ? | — | — | — | — | — |
| AFW-Net | N (estimate only) | — | — | — | — | No public code (MDPI Sensors 2026); parameter/FLOP figures are paper-derived estimates, not measured |

Epoch extrapolation: mini = 323 iterations, full nuScenes = 28130 iterations,
both at batch size 1, sec/iter x iterations. Excludes validation and data-prep.

Per-method detail, including every install/dependency issue encountered, is in
each `feasibility/<method>/RESULTS.md`.
