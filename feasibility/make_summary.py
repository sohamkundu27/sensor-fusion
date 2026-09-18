#!/usr/bin/env python3
"""Builds feasibility/SUMMARY.md from each method's result.json + curated notes."""
import json, os, datetime
F = os.path.dirname(os.path.abspath(__file__))
MINI, FULL = 323, 28130

M = [  # (folder, display name, blocker/notes when it ran or didn't)
 ("seeing-through-fog", "Deep Entropy Fusion (Seeing Through Fog)",
  "Official repo has no model or training code (dataset toolkit only); STF data not downloaded, registration-gated"),
 ("raf", "RAF",
  "Frozen L4DR/3D-LRF backbone never published (author-local paths only); no VoD loader in released code; K-Radar/VoD not local"),
 ("samfusion", "SAMFusion",
  "Fits with ~40 MB spare only because img+LiDAR branches are frozen; nuScenes init checkpoint (DeepInteraction) not published; 2 source fixes"),
 ("bevfusion", "BEVFusion",
  "Does not fit as shipped: bundled spconv reserves 6.5 GB; one-constant fix (4096→256) + 5 other source/config fixes"),
 ("rcbevdet", "RCBEVDet",
  "None at runtime (access gate is gone); zip is missing requirements/ and a dataset class"),
 ("transfusion", "TransFusion (LiDAR-only, speed ref.)",
  "Fits as shipped but uses 11.5 of 11.6 GB (same spconv reservation; 5.4 GB with the fix); 1 compile fix"),
 ("afw-net", "AFW-Net",
  "No public code; figures are paper-derived estimates (14.5 M params stated by authors); uses gated STF data"),
]

def row(folder, name, note):
    p = os.path.join(F, folder, "result.json")
    if folder == "afw-net":
        return f"| {name} | N (estimate) | — | ~0.33 (est.) | n/a (STF: ~1.4 h/epoch est.) | n/a | {note} |"
    if not os.path.exists(p):
        return f"| {name} | N | — | — | — | — | {note} |"
    d = json.load(open(p))
    if not str(d.get("ran", "")).startswith("Y"):
        return f"| {name} | {d.get('ran','N')} | — | — | — | — | {note} |"
    vram = f"{d['vram_gb']:.2f} GB torch / {d['device_used_gb']:.2f} GB device"
    return (f"| {name} | Y | {vram} | {d['sec_per_iter']:.3f} | "
            f"{d['sec_per_iter']*MINI/3600:.3f} | {d['sec_per_iter']*FULL/3600:.2f} | {note} |")

lines = [
 "# Single-GPU feasibility triage: 7 camera/LiDAR/radar fusion methods", "",
 f"Generated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC} by `make_summary.py`. "
 "RTX 3080, 11.63 GB usable (nvidia-smi: 12,288 MiB), driver 580.173.02.", "",
 "All measurements: batch size 1, 5 warmup + 30 timed training iterations (forward, backward, "
 "optimizer step) on nuScenes-mini, on an otherwise idle GPU. sec/iter is the wall-clock median "
 "including data loading. Timing only — no accuracy or convergence claim.", "",
 "| method | ran (Y/N) | VRAM @ batch 1 | sec/iter | est. hrs/epoch (mini) | est. hrs/epoch (full nuScenes) | biggest blocker |",
 "|---|---|---|---|---|---|---|",
] + [row(*m) for m in M] + [
 "",
 f"**Epochs:** mini = {MINI} train keyframes, full = {FULL:,}, at batch 1, *without* CBGS. "
 "BEVFusion, TransFusion, RCBEVDet and SAMFusion all train with CBGS upstream, which makes a "
 "published epoch roughly 4-5x longer; multiply before comparing with paper schedules. "
 "Validation and data preparation are excluded.",
 "",
 "**VRAM:** \"torch\" = `torch.cuda.max_memory_allocated`; \"device\" = everything in use on the card "
 "at the end of the run (`cudaMemGetInfo`), including CUDA context and memory CUDA extensions reserve "
 "outside PyTorch. The two differ hugely for BEVFusion and TransFusion because of a sparse-conv "
 "kernel — see their RESULTS.md.",
 "",
 "Per-method detail, including every install and dependency issue and every source change, is in "
 "`feasibility/<method>/RESULTS.md`; source changes are in `feasibility/<method>/patches/`.",
]
open(os.path.join(F, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
