# Mitsuba / mitransient smoke test

Ran 2026-09-23 on this machine. **Both renders completed.** The clear histogram
shows one peak per object at the expected two-way time of flight. The foggy
histogram shows that same peak plus a near-range rising/decaying scattered-light
tail — the backscatter signature this pipeline is meant to capture.

Every LiDAR / fog / film number below is a **placeholder** for a cheap first
test, not a sensor or atmosphere specification. They are marked `(placeholder)`
where they appear.

---

## Bottom line

| Check | Result |
|---|---|
| Isolated env + `mitsuba==3.8.0` + `drjit` + `mitransient==1.3.0` | Installed in `~/venvs/mitsuba` (drjit resolved to **1.3.1**, the pin of Mitsuba 3.8.0) |
| Scene | `sim/mitsuba_smoketest/scene.xml` (reuse this) |
| Variant | **`cuda_ad_mono`** for the numbers below. Card was idle (1 MiB, 0% util, no compute apps) at 17:02. An earlier pair on **`llvm_ad_mono`** (GPU held by CARLA) is kept in `outputs/llvm/` and compared in the timing table. |
| Clear: one clean peak per object at expected ToF? | **Yes.** After a 1 ns (placeholder) pulse, 100% / 92% / 100% of pure pixels on A / B / C have a single peak, and 100% / 94% / 100% of raw argmaxes sit in the ground-truth bin window. Median error 0.3–0.5 bins. Energy before the window is **0**. Sky pixels are exactly 0. |
| Fog: that peak **plus** a backscatter tail? | **Yes.** On object pixels, 92–98% of the energy arrives *before* the surface ToF (near-range backscatter). The surface peak is still there at the same bin. Fog-only pixels (no surface) are 0 in clear and a decaying continuum in fog, peaking at 0.26 m one-way / 1.75 ns, matching an analytic single-scattering curve at close range. |
| NaNs / negatives | 0 / 0 in both transients |

If this had failed, the output would have been reported as-is. It did not fail.

Figures: `sim/mitsuba_smoketest/figures/{overview,histograms,backscatter_vs_single_scatter}.png`.

---

## Environment

```
python3.12 -m venv ~/venvs/mitsuba
~/venvs/mitsuba/bin/pip install mitsuba==3.8.0 drjit mitransient==1.3.0 numpy matplotlib nvidia-ml-py
```

| Package | Installed |
|---|---|
| mitsuba | 3.8.0 |
| drjit | 1.3.1 |
| mitransient | 1.3.0 |
| numpy | 2.5.3 |
| matplotlib | 3.11.2 |

Reuse:

```
cd sim/mitsuba_smoketest
~/venvs/mitsuba/bin/python render.py --variant cuda_ad_mono --sigma-t 0.0  --out outputs/clear.npz
~/venvs/mitsuba/bin/python render.py --variant cuda_ad_mono --sigma-t 0.15 --out outputs/fog.npz
# or: ./run.sh 0.15 16384   (cuda_ad_mono if the GPU is free, else llvm_ad_mono)
```

`run.sh` deletes `~/.drjit` first so the cold time includes JIT compile.

---

## Scene (for reuse)

Path: **`sim/mitsuba_smoketest/scene.xml`**

Load after setting a `*_mono` variant and `import mitransient` (that import
registers `transient_prbvolpath`, `transient_hdr_film`, `angulararea`). Tunables
are XML `$defaults`, overridable at load:

```
scene = mi.load_file("scene.xml", sigma_t=0.15, albedo=0.95, g=0.85, spp=16384)
```

| Element | What was used | Placeholder? |
|---|---|---|
| Ground | Rectangle y=0, 20 m × 12 m | — |
| Object A | Sphere, r=0.5 m, centre (−1.4, 0.5, 3.8). One-way range on pure pixels **3.58–4.11 m** (median 3.69 m, ToF 24.6 ns) | geometry for this test |
| Object B | 1 m cube, front face z=5.0. Range **5.10–5.65 m** (median 5.26 m, ToF 35.1 ns) | geometry for this test |
| Object C | 1.6×1.5×1.0 m box, front face z=6.5. Range **6.50–6.60 m** (median 6.53 m, ToF 43.5 ns) | geometry for this test |
| Medium | Homogeneous, HG phase, `sigma_t` / `albedo` / `g` | **yes** — `albedo=0.95`, `g=0.85`; clear `sigma_t=0`; fog `sigma_t=0.15 /m` |
| Laser stand-in | `angulararea` on a 1 cm disk at (0, 1, −0.001), facing +z. `cutoff_angle=beam_width=30` (half-angle ⇒ **60° full cone**) | **yes** — 60° FOV, radiance `10000` |
| Wavelength | `cuda_ad_mono` / `llvm_ad_mono` carry no λ. Labelled “905 nm equivalent” only; reflectances stand in for NIR | **yes** — 905 nm, ρ_ground=0.2, ρ_object=0.5 |
| Pulse | mitransient 1.3.0 emits a **delta at t=0**. The ~1 ns width is a **post-render 1 ns FWHM Gaussian** in `analyze.py` | **yes** — 1 ns |
| Integrator | `transient_prbvolpath`, `max_depth=8`, `rr_depth=5` | depth is a cost knob |
| Film | `transient_hdr_film` **64×64**, **512** bins, `bin_width_opl = c × 100 ps = 0.0299792458 m`, `start_opl=0` | **yes** — 64×64, 512, 100 ps. Window = 15.35 m OPL ≈ **7.67 m one-way**. Deliberately short, not the real target range. |
| Sensor | Perspective, 60° HFOV, origin (0, 1, 0), look +z | **yes** — 60° FOV, 1 m height |
| spp | 16384, independent sampler, seed 0 | cost knob |

Time in the film is **optical path length in metres**, not seconds. All emitters
fire at t=0. Two-way ToF in seconds is `OPL / c`; one-way range is `OPL / 2`.

### Two implementation constraints that are not placeholders

1. **Sensor cannot start inside a medium.** mitransient 1.3.0 still has
   `TODO: support sensors inside media`. Fog is a null-BSDF cube whose front
   face is 5 cm in front of the sensor (`z ∈ [0.05, 12.05]`). The first 5 cm of
   each camera ray (and ~5.1 cm of each laser ray) are vacuum. That is ~0.1 m of
   extra OPL, a few bins, and is why Beer–Lambert comparisons subtract
   `2 × 0.05 m`.
2. **Mitsuba merges the two cubes on XML load** into one unnamed mesh
   (union bbox). They still render as two boxes. IDs `obj_B_box` / `obj_C_box`
   do not survive `mi.load_file`. Analysis classifies hits by AABB, not shape
   id. Separate BSDF instances did not stop the merge. The XML still names them
   separately for reuse / editing.

---

## Time and memory (measured)

Wall clock is `time.perf_counter()` around `integrator.render` + `dr.sync_thread()`.
Each process rendered twice; **warm** is the second pass (kernel already compiled).
`~/.drjit` was deleted before the CUDA pair. CPU peak is `RUSAGE_SELF.ru_maxrss`.
GPU numbers are NVML, polled every 10 ms. Quote **this-process** GPU, not
device-wide (device-wide includes the ~378 MiB idle reserve).

### `cuda_ad_mono` — timed pair (GPU idle)

| | Clear (`σ_t = 0`) | Fog (`σ_t = 0.15 /m`) |
|---|---|---|
| Timestamp | 2026-09-23 17:02:27 −0500 | 2026-09-23 17:02:37 −0500 |
| Card at start | 1 MiB, 0% util, no compute apps | same (our clear process had exited) |
| spp / resolution | 16384, 64×64×512 | same |
| Wall, cold (incl. JIT) | **1.10 s** | **2.86 s** |
| Wall, warm | **0.69 s** | **2.85 s** |
| Process total (import + 2 renders + save) | 2.51 s | 6.51 s |
| CPU user time | 4.81 s | 7.78 s |
| Peak CPU RSS | **475 MiB** | **432 MiB** |
| This process, peak GPU (NVML) | **498 MiB** | **498 MiB** |
| Device-wide peak (NVML) | 884 MiB (baseline 378 MiB) | 884 MiB (baseline 378 MiB) |
| NaNs / negatives | 0 / 0 | 0 / 0 |

Same scene, same seed, same sanity-check signature as the LLVM pair (clear
argmaxes 100% / 94% / 100%; fog pre-window energy 93% on A).

### `llvm_ad_mono` — earlier pair (GPU busy)

Kept in `sim/mitsuba_smoketest/outputs/llvm/`. CARLA held 5563 MiB at 99% util
when the clear half started; it had exited by the fog half. Both halves stayed
on LLVM so they are comparable to each other.

| | Clear | Fog |
|---|---|---|
| Wall, cold / warm | 2.55 s / **2.28 s** | 5.79 s / **5.81 s** |
| Peak CPU RSS | 293 MiB | 271 MiB |
| This process, peak GPU | 224 MiB (CUDA context only) | 224 MiB |

Warm CUDA vs warm LLVM: clear **3.3×** (0.69 s vs 2.28 s), fog **2.0×**
(2.85 s vs 5.81 s). Fog is less GPU-bound at this film size (more work per
ray, same 64×64×512 wavefront).

Film arithmetic: 64 × 64 × 512 × 4 B ≈ 8 MiB per float32 channel. The ~500 MiB
process GPU is the wavefront + OptiX/CUDA context, not the film.

---

## Sanity check, in numbers

Ground-truth ToF is a 4×4 camera-ray walk (stepping the null fog-box faces)
plus the Euclidean return to the laser disk centre. The “GT window” is that
sub-pixel OPL range ± 1 bin.

### Clear

| | A sphere | B cube | C far box |
|---|---|---|---|
| Pure pixels | 161 | 131 | 144 |
| Median one-way range | 3.69 m | 5.26 m | 6.53 m |
| Raw argmax in GT window | **100%** | **94%** | **100%** |
| Median \|argmax − mean GT\| | 0.53 bin | 0.26 bin | 0.26 bin |
| Energy in GT window (median) | 97.5% | 95.5% | 93.9% |
| Energy *before* the window | **0** | **0** | **0** |
| Single pulsed peak (local, >5% of max) | 100% | 92% | 100% |

Sky pixels inside the 30° half-cone: **clear total energy max = 0**.

Object B’s 6% misses are silhouette pixels (max error 15 bins) where a 4×4
sub-ray mix spans a wide depth. Interior pixels are clean.

### Fog (`σ_t = 0.15 /m`, albedo 0.95, g = 0.85 — all placeholders)

| | A | B | C |
|---|---|---|---|
| Object peak still in GT window | 91% | 84% | 74% |
| Energy in GT window (median) | 5.2% | 1.7% | 0.7% |
| Energy *before* the window | **92%** | **97%** | **98%** |
| Pre / peak energy | 18× | 56× | 127× |
| Strongest *pulsed* return is the object | 5% | 0.8% | 0% |

The object return is still a peak at the same ToF (see
`figures/histograms.png`). It is no longer the brightest thing in the
histogram: the near-range backscatter is. After the 1 ns (placeholder) pulse,
fog-only pixels peak at **0.26 m / 1.75 ns** and decay (0.9% of peak at 1 m,
0.02% at 3 m). Clear is identically zero on those pixels.

Monte Carlo vs a first-order single-scattering prediction (point emitter,
same HG, same `σ_t`) on fog-only pixels is the same shape near the sensor
(~1–6×, the model is not a radiometric match — disk vs point, missing
multiple scatter). Pre-object energy on A/B/C is ~2.3× the single-scatter
prediction, i.e. the extra is multiple scattering, not a missing peak.

A naive Beer–Lambert ratio of *window energy* fog/clear is ~1.7–1.8, not
`e^{-σ_t L}` (~0.34 / 0.21 / 0.14). The window is polluted by in-bin scatter,
so that ratio is not a transmittance measurement. The ToF *location* of the
surface peak is the check that passed.

Small far-bin bumps (~7 m, ~10⁻⁶) on the fog curves are ~1000× below the
backscatter peak; treated as noise, not a fourth object.

---

## What the output would have looked like if this had failed

It did not. For the record, a failure would have been: clear energy on sky
pixels, object argmaxes nowhere near the GT bins, or fog indistinguishable
from a scaled copy of clear (no pre-peak tail). None of those happened.

---

## Files

| Path | Role |
|---|---|
| `sim/mitsuba_smoketest/scene.xml` | Scene for reuse |
| `sim/mitsuba_smoketest/render.py` | One-render driver, time + memory JSON |
| `sim/mitsuba_smoketest/analyze.py` | ToF + backscatter checks, figures |
| `sim/mitsuba_smoketest/run.sh` | Variant pick, both renders, analyze |
| `sim/mitsuba_smoketest/outputs/{clear,fog}.{npz,json}` | CUDA timed pair + metadata (gitignored via `outputs/`) |
| `sim/mitsuba_smoketest/outputs/llvm/` | Earlier LLVM pair (same scene, GPU was busy) |
| `sim/mitsuba_smoketest/outputs/results.json` | Numeric sanity-check dump (from the CUDA pair) |
| `sim/mitsuba_smoketest/figures/*.png` | overview / per-object histograms / SS overlay |
