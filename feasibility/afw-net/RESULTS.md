# Feasibility: AFW-Net (Adaptive Feature-Weighting Network)

Paper: "Adaptive Sensor Fusion for Robust Perception in Dense Fog: A Gated Vision
and LiDAR Integration Framework", *MDPI Sensors* 26(12):3728, June 2026.
DOI: 10.3390/s26123728 · PMC13306363

**No public code.** Per instruction, no reimplementation was attempted. Everything
below is derived from the paper's architecture section and is an **estimate, not a
measurement**.

## Result

| Field | Value |
|---|---|
| Ran | **N — estimate only, no code exists** |
| VRAM @ batch 1 | not measured; see estimate below |
| sec/iteration | **~0.33 s (estimated)** |
| Est. hrs/epoch (mini) | n/a — does not use nuScenes |
| Est. hrs/epoch (full nuScenes) | n/a — does not use nuScenes |
| Biggest blocker | No public implementation; and its dataset is the same gated STF data that blocks method 1 |

## Architecture, as stated in the paper

| Component | Specification |
|---|---|
| Gated image encoder | ResNet-50 + channel attention after each residual block |
| LiDAR encoder | PointNet++, 4 set-abstraction layers, radii [0.2, 0.4, 0.8, 1.6] m, groups [32, 64, 128, 256] |
| Feature level | stride-8 pyramid, C = 256 |
| Fusion | two-layer MLPs (hidden 256) for channel-wise weights; uncertainty via `softplus(Conv1x1(Fg))`; cross-modal attention with learnable scale β |
| Head | FCOS, anchor-free, strides [8, 16, 32, 64, 128] |
| **Parameters** | **14.5 M (stated by the authors)** |
| Training | batch 16 across 4x RTX 3090, 120 epochs |
| Inference | ~12 FPS on a single RTX 3090 |
| Dataset | Princeton Automated Driving Dataset, 15,000 frames |

The 14.5 M parameter count is the authors' own figure, not my estimate. For scale,
this project's own entropy-fusion model is 15.73 M — the two are comparable, and
both are far smaller than the BEV methods in this triage.

## FLOP estimate (mine, rough)

Derived by scaling published per-component costs to the stated configuration at
STF's native 1280x720 gated resolution:

| Component | Estimate |
|---|---:|
| ResNet-50 @ 1280x720 (18.4x the 224² cost of ~4.1 GFLOPs) | ~75 GFLOPs |
| FPN + FCOS head, 5 levels (~50% of backbone) | ~38 GFLOPs |
| PointNet++, 4 SA layers | ~4 GFLOPs |
| Adaptive fusion MLPs, C=256 at 160x90 | ~4 GFLOPs |
| **Total forward** | **~121 GFLOPs/frame** |
| Training step (~3x forward) | ~362 GFLOPs |

Treat these as order-of-magnitude. The dominant term is ResNet-50 at full gated
resolution; if the authors downscale before the backbone (the paper does not say),
the true figure could be several times lower.

## Timing estimate (mine, rough)

Anchored on the authors' own 12 FPS inference on an RTX 3090:

* RTX 3080 at ~75% of 3090 throughput → ~9 FPS inference
* Training step ~3x inference → **~3.0 it/s ≈ 0.33 s/iteration** at batch 1
* STF 15,000 frames → **~1.4 h/epoch**
* Their full 120-epoch schedule → **~167 h ≈ 7 days** on this single 3080

That last figure is the useful one: even though AFW-Net is the *smallest* model in
this triage by parameter count, reproducing the published training schedule on one
3080 is a week of continuous compute.

## Blockers

1. **No public code.** The paper provides no repository link. A reimplementation
   from the architecture section is possible in principle — the description is
   unusually complete — but that is a project, not a feasibility check.
2. **Same data blocker as method 1.** The "Princeton Automated Driving Dataset" is
   Seeing Through Fog / DENSE. It needs gated NIR imagery, the local STF
   directories are empty, and access is registration-gated. Even a perfect
   reimplementation could not be trained here today.

## Sources

- [Adaptive Sensor Fusion for Robust Perception in Dense Fog (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306363/)
- [doi.org/10.3390/s26123728](https://doi.org/10.3390/s26123728)
