#!/usr/bin/env python3
"""Builds feasibility/SUMMARY.md from per-method result JSONs + static blockers."""
import json, os, datetime, glob
F = os.path.dirname(os.path.abspath(__file__))

# Methods with no runnable code/inputs -- findings already established, see each RESULTS.md
STATIC = {
 "seeing-through-fog": dict(order=1, name="Deep Entropy Fusion (Seeing Through Fog)", ran="N",
   blocker="Official repo has no model/training code (dataset toolkit only); STF data not downloaded and registration-gated"),
 "raf": dict(order=2, name="RAF (Reliability-Aware Fusion)", ran="N",
   blocker="Frozen pretrained L4DR/3D-LRF backbone never distributed (author-local paths only); no VoD loader in released code"),
 "samfusion": dict(order=3, name="SAMFusion", ran="?", blocker=""),
 "bevfusion": dict(order=4, name="BEVFusion", ran="?", blocker=""),
 "rcbevdet": dict(order=5, name="RCBEVDet", ran="?", blocker=""),
 "transfusion": dict(order=6, name="TransFusion", ran="?", blocker=""),
 "afw-net": dict(order=7, name="AFW-Net", ran="N (estimate only)",
   blocker="No public code (MDPI Sensors 2026); parameter/FLOP figures are paper-derived estimates, not measured"),
}
MINI_ITERS = 323    # nuScenes-mini train keyframes (v1.0-mini, 404 samples, ~80/20 split)
FULL_ITERS = 28130  # nuScenes trainval keyframes

rows = []
for key, meta in sorted(STATIC.items(), key=lambda kv: kv[1]["order"]):
    r = dict(meta); r["key"] = key
    j = os.path.join(F, key, "result.json")
    if os.path.exists(j):
        try:
            d = json.load(open(j))
            r["ran"] = d.get("ran", "?")
            r["vram"] = d.get("vram_gb")
            r["sec"] = d.get("sec_per_iter")
            r["blocker"] = d.get("blocker", r.get("blocker", ""))
        except Exception as e:
            r["blocker"] = f"result.json unreadable: {e}"
    rows.append(r)

def fmt(r):
    v = f'{r["vram"]:.2f} GB' if r.get("vram") else "—"
    s = f'{r["sec"]:.3f}' if r.get("sec") else "—"
    if r.get("sec"):
        mini = f'{r["sec"]*MINI_ITERS/3600:.2f}'
        full = f'{r["sec"]*FULL_ITERS/3600:.1f}'
    else:
        mini = full = "—"
    return f'| {r["name"]} | {r["ran"]} | {v} | {s} | {mini} | {full} | {r.get("blocker","") or "—"} |'

out = [
 "# Single-GPU feasibility triage: 7 camera/LiDAR/radar fusion methods",
 "",
 f"Generated {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} by `make_summary.py`.",
 "GPU: NVIDIA RTX 3080, 12288 MiB, driver 580.173.02 / CUDA 13.0.",
 "All timings taken at batch size 1, 20-50 iterations, on an otherwise idle GPU",
 "(the gating ablation had finished). Timing only -- no accuracy or convergence claim.",
 "",
 "| method | ran | VRAM @ batch 1 | sec/iter | est. hrs/epoch (mini) | est. hrs/epoch (full nuScenes) | biggest blocker |",
 "|---|---|---|---|---|---|---|",
] + [fmt(r) for r in rows] + [
 "",
 f"Epoch extrapolation: mini = {MINI_ITERS} iterations, full nuScenes = {FULL_ITERS} iterations,",
 "both at batch size 1, sec/iter x iterations. Excludes validation and data-prep.",
 "",
 "Per-method detail, including every install/dependency issue encountered, is in",
 "each `feasibility/<method>/RESULTS.md`.",
]
open(os.path.join(F, "SUMMARY.md"), "w").write("\n".join(out) + "\n")
print("wrote SUMMARY.md")
