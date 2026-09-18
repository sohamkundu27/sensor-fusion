# Single-GPU feasibility triage: 7 camera/LiDAR/radar fusion methods

Generated 2026-09-18 18:23 UTC by `make_summary.py`. RTX 3080, 11.63 GB usable (nvidia-smi: 12,288 MiB), driver 580.173.02.

All measurements: batch size 1, 5 warmup + 30 timed training iterations (forward, backward, optimizer step) on nuScenes-mini, on an otherwise idle GPU. sec/iter is the wall-clock median including data loading. Timing only — no accuracy or convergence claim.

| method | ran (Y/N) | VRAM @ batch 1 | sec/iter | est. hrs/epoch (mini) | est. hrs/epoch (full nuScenes) | biggest blocker |
|---|---|---|---|---|---|---|
| Deep Entropy Fusion (Seeing Through Fog) | N | — | — | — | — | Official repo has no model or training code (dataset toolkit only); STF data not downloaded, registration-gated |
| RAF | N | — | — | — | — | Frozen L4DR/3D-LRF backbone never published (author-local paths only); no VoD loader in released code; K-Radar/VoD not local |
| SAMFusion | Y | 7.36 GB torch / 11.59 GB device | 0.711 | 0.064 | 5.56 | Fits with ~40 MB spare only because img+LiDAR branches are frozen; nuScenes init checkpoint (DeepInteraction) not published; 2 source fixes |
| BEVFusion | Y | 6.11 GB torch / 9.16 GB device | 0.427 | 0.038 | 3.34 | Does not fit as shipped: bundled spconv reserves 6.5 GB; one-constant fix (4096→256) + 5 other source/config fixes |
| RCBEVDet | Y | 1.77 GB torch / 3.88 GB device | 0.494 | 0.044 | 3.86 | None at runtime (access gate is gone); zip is missing requirements/ and a dataset class |
| TransFusion (LiDAR-only, speed ref.) | Y | 2.56 GB torch / 11.54 GB device | 0.342 | 0.031 | 2.67 | Fits as shipped but uses 11.5 of 11.6 GB (same spconv reservation; 5.4 GB with the fix); 1 compile fix |
| AFW-Net | N (estimate) | — | ~0.33 (est.) | n/a (STF: ~1.4 h/epoch est.) | n/a | No public code; figures are paper-derived estimates (14.5 M params stated by authors); uses gated STF data |

**Epochs:** mini = 323 train keyframes, full = 28,130, at batch 1, *without* CBGS. BEVFusion, TransFusion, RCBEVDet and SAMFusion all train with CBGS upstream, which makes a published epoch roughly 4-5x longer; multiply before comparing with paper schedules. Validation and data preparation are excluded.

**VRAM:** "torch" = `torch.cuda.max_memory_allocated`; "device" = everything in use on the card at the end of the run (`cudaMemGetInfo`), including CUDA context and memory CUDA extensions reserve outside PyTorch. The two differ hugely for BEVFusion and TransFusion because of a sparse-conv kernel — see their RESULTS.md.

Per-method detail, including every install and dependency issue and every source change, is in `feasibility/<method>/RESULTS.md`; source changes are in `feasibility/<method>/patches/`.
