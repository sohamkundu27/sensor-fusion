# CARLA → Mitsuba pipeline readiness

Checked 2026-09-23. Investigation only: nothing was installed, built, downloaded or
changed. "Checked" means inspected on this machine; "per docs" means taken from the
vendor documentation or PyPI metadata and **not verified locally**, because neither
tool is installed.

---

## Bottom line

- **Nothing is installed.** No CARLA (server or Python client), no Unreal Engine, no
  Mitsuba 2 or 3, no Dr.Jit, and no `mitransient`. None of the 17 Python interpreters
  on the machine has any of them, and nothing in this repo references them.
- **CARLA 0.9.16 (Unreal Engine 4.26) is the realistic version for this GPU.** Its
  docs recommend 8 GB of VRAM; the card has 12 GB. CARLA 0.10.0 (Unreal Engine 5.5)
  recommends at least 16 GB plus a separate GPU for ML. **Neither line officially
  supports Ubuntu 24.04**, which this machine runs.
- **Headless rendering is supported** (`-RenderOffScreen`, Vulkan, no display server).
  The NVIDIA Vulkan driver is present. The Unreal *editor*, which is needed to get
  meshes out, is a GUI app, and running it on this display-less box is unsolved.
- **CARLA's Python API cannot export meshes or read materials.** It gives transforms,
  bounding boxes, semantic labels, the road network, weather/fog parameters, lights,
  sensor poses and walker bone poses. Geometry and textures need a source build of
  CARLA and its Unreal editor: about 130 GB, a 4+ hour build, and an Epic-linked
  GitHub account.
- **Transient rendering is installable, with a version pin.** `mitransient` 1.3.0
  provides a time-resolved volumetric integrator (`transient_prbvolpath`). It
  requires `mitsuba>=3.6,<3.9`, so pip would pull Mitsuba 3.8.0 rather than the latest
  3.9.1. Mitsuba 3 itself has volumetric path tracers and homogeneous/heterogeneous
  media for fog.
- **Run CARLA and Mitsuba one after the other, not at the same time, and not
  alongside a heavy training job.** CARLA's guidance already assumes most of an 8 GB
  card, and a dense transient film can take gigabytes on its own (see the GPU section).
- **Disk: 178 GB free of 915 GB (80% used).** A CARLA package (~20 GB) fits easily.
  A source build (~130 GB) would leave about 48 GB. Dense transient outputs are the
  real disk risk.

---

## 1. Installed vs missing

| Component | Status | Evidence |
|---|---|---|
| CARLA server (any version) | **Missing** | Full-filesystem scan for `*carla*`, `CarlaUE4*`, `CarlaUnreal*` (dataset folders excluded) found no install. The only hits were Ubuntu software-catalog icons for the unrelated "Carla" audio plugin host. No apt or snap package, no Docker image. |
| `CARLA_ROOT` / `CARLA_HOME` / `UE4_ROOT` env vars | **Not set** | Not in the shell environment; not in `~/.bashrc`, `~/.profile`, `~/.bash_profile`, `~/.zshrc`, `~/.bash_aliases` or `/etc/environment` |
| `carla` Python client | **Missing** | 17 interpreters checked: system Python 3.12.3, `~/venvs/{entropy-fusion,nuscenes,torch,transfusion}`, the 7 feasibility micromamba envs plus the RAF venv, `~/.venv`, two venvs under `~/github`, and one agent-tool venv |
| Unreal Engine (UE4 or UE5 editor) | **Missing** | No `UE4Editor`, `UnrealEditor` or `UnrealEngine*` anywhere |
| Mitsuba 3 / Dr.Jit | **Missing** | Not importable in any of the 17 interpreters; no `mitsuba*` or `drjit` path anywhere on disk. No Mitsuba 2 either. |
| `mitransient` | **Missing, and not referenced** | Not installed; zero mentions in the repo (including envs and cloned third-party repos) or elsewhere on disk |
| NVIDIA driver | 580.173.02 (open), CUDA 13.0 | Meets CARLA UE5's ≥550 and Mitsuba/mitransient GPU's ≥495.89 |
| Vulkan (for CARLA on Linux) | Present: `libvulkan1` 1.3.275 + NVIDIA ICD (`nvidia_icd.json`, API 1.4.312) | `vulkaninfo` isn't installed, so there was no end-to-end Vulkan test |
| OptiX (for Mitsuba `cuda_*` variants) | Present: `libnvoptix.so.1` (ships with the driver) | |
| LLVM (for Mitsuba `llvm_*` CPU variants; needs ≥11.1) | Present: `libLLVM` 18.1, 19.1, 20.1 | |
| NVIDIA EGL/GLX | Present: `libEGL_nvidia.so.0`, `libGLX_nvidia.so.0` | |
| Docker | Installed; 4 local images (ROS, VINS-Fusion), 6.4 GB | **No NVIDIA container toolkit**, so GPU containers (e.g. `carlasim/carla`) won't run until it's installed, which needs sudo |
| sudo | Not passwordless | Any apt install (container toolkit, `vulkaninfo`, build deps) needs you |
| Display server | None on the GPU | `gdm3` is running but there is no Xorg, Xwayland or compositor process. Sessions are tty/ssh. `nvidia-smi` shows no GPU processes. |

### Machine

| | |
|---|---|
| OS | Ubuntu 24.04.3 LTS, kernel 7.0.0-31-generic, glibc 2.39 |
| CPU | Intel i9-12900K, 24 threads |
| RAM | 62 GiB (58 GiB available at check time), 8 GiB swap |
| GPU | One RTX 3080 12 GB (12,288 MiB, compute 8.6). Idle at check time: 1 MiB used, no processes. It is also the only GPU for training. |

---

## 2. CARLA

### Versions available (per GitHub releases and PyPI)

| Line | Latest numbered release | Engine | Notes |
|---|---|---|---|
| UE4 | **0.9.16** | Unreal Engine 4.26 (CARLA fork) | `carla==0.9.16` on PyPI (uploaded 2025-09-14) has **cp310/cp311/cp312 manylinux_2_31** wheels, which suit this machine's Python 3.12 and glibc 2.39 |
| UE5 | **0.10.0** (2024-12-19) | Unreal Engine 5.5 | Packages for Ubuntu 22 and Windows 11. Python 3.8–3.12. The client wheel ships inside the package (PyPI stops at 0.9.16). Newer fixes exist only on the `ue5-dev` nightly. |

CARLA says both lines will "coexist for the foreseeable future"; some UE4 assets and
features are not yet migrated to UE5.

### Requirements vs this machine

| Requirement | 0.9.16 package | 0.9.16 source build (needed for mesh export) | 0.10.0 package | This machine |
|---|---|---|---|---|
| OS | Ubuntu 20.04 / 22.04 | Ubuntu 20.04 / 22.04; "not tested internally in Ubuntu 24.04 … recommend … a maximum of Ubuntu 22.04" | "minimum of Ubuntu 22.04" | **24.04.3: outside the tested range for UE4; not explicitly listed for UE5** |
| GPU | 2070-class, **≥ 8 GB** VRAM | RTX 2000+, ≥ 6 GB, preferably 8 GB | RTX 3000+, **≥ 16 GB** VRAM; "a dedicated GPU, separate from the GPU used for CARLA, is highly recommended" for ML | RTX 3080 12 GB, single GPU: **meets 0.9.16, below 0.10.0's recommendation** |
| Driver | — | — | ≥ 550 | 580 ✓ |
| Disk | ~20 GB | ~130 GB (Unreal Engine ~91 + CARLA ~31) | 130 GB | 178 GB free |
| Python | 3.7–3.12 | 3.8+ | 3.8–3.12 | 3.12.3 ✓ |
| Other | TCP ports 2000/2001 | GitHub account linked to Epic Games; build takes "4 hours or more"; i7 with 4+ cores | TCP ports 2000/2001 | i9-12900K ✓ |

**Can this GPU run CARLA alongside everything else?** Only 0.9.16 fits the card per
the docs, and then only if CARLA has the GPU to itself or nearly so. The docs
recommend 8 GB for CARLA alone. If actual use matches that, about 4 GB is left on
this card (not measured). Recent workloads on this card for comparison: the
feasibility pass measured device-wide use of 11.59 GiB for SAMFusion and 11.54 GiB
for TransFusion as shipped, so either would conflict with CARLA. The entropy-fusion
ablation peaked at 0.67 GiB PyTorch-allocated, so it might coexist, but
compute contention was not measured.

### Headless / off-screen rendering

- **Supported since 0.9.12** via `./CarlaUE4.sh -RenderOffScreen` (UE5:
  `CarlaUnreal.sh -RenderOffScreen`). Unreal renders normally and camera sensors
  return data, with no display. On Linux, UE 4.26 uses **Vulkan only**, and the NVIDIA
  Vulkan driver is installed here.
- This is different from `no_rendering_mode` (a world setting), where nothing is
  drawn and cameras return empty data.
- The Docker route (`carlasim/carla:<ver>` with `-RenderOffScreen -nosound`) also
  needs the NVIDIA container toolkit, which is missing and needs sudo. Its images are
  Ubuntu 22-based, which could sidestep the 24.04 support gap.
- **Gap:** off-screen mode covers the *simulator*. The **Unreal editor** is a GUI
  application, and the docs don't describe running it without a display. Mesh
  export needs the editor, so it needs either a remote desktop / virtual display on
  this box or a different machine.

### What CARLA can export for Mitsuba

Source: CARLA 0.9.16 Python API and sensor reference.

| What Mitsuba needs | Python API (works with a packaged CARLA) | Needs the Unreal editor / a plugin | Detail |
|---|---|---|---|
| **Actor transforms** (vehicles, walkers, props) | **Yes** | — | `Actor.get_transform()`, `get_velocity()`, `bounding_box`; `WorldSnapshot` per tick. Synchronous mode with `fixed_delta_seconds` gives deterministic stepping. |
| **Sensor (camera/LiDAR) poses** | **Yes** | — | Sensors are actors, so they have world transforms and parent attachment |
| **Camera intrinsics** | **Partial** | — | Only blueprint attributes (`image_size_x/y`, horizontal `fov`, lens distortion params). The K matrix has to be derived from the FOV; the API doesn't return it. |
| **LiDAR configuration and point clouds** | **Yes, but not physically based** | — | `channels`, `range`, `points_per_second`, `rotation_frequency`, `upper/lower/horizontal_fov`. Output is x, y, z plus intensity, and the semantic LiDAR adds object id and semantic tag. It is **ray-cast**, with intensity = exp(−a·d) from a user-set `atmosphere_attenuation_rate`: **no scattering, no time-of-flight histogram**. Usable as a geometry cross-check, not as SP-LiDAR ground truth. |
| **Walker (pedestrian) pose** | **Yes (bone transforms)** | Skeletal mesh itself: yes | `Walker.get_bones()` returns every bone's transform (world, component and parent-relative), so animated pedestrians can be posed per frame if the skeletal mesh is exported separately |
| **Static environment objects** | **Partial** | — | `World.get_environment_objects()` gives id, name, semantic type, transform and bounding box (8 corners via `get_world_vertices`). `get_level_bbs()`; objects can be toggled on and off. **No vertices or triangles.** |
| **Road network** | **Yes** | — | `Map.to_opendrive()` / `save_to_disk()` gives the OpenDRIVE 1.4 `.xodr`: lanes, road geometry, landmarks, crosswalk polygons. This is a road description, not the rendered road mesh. |
| **Geometry probing** | **Yes** | — | `cast_ray`, `project_point`, `ground_projection` return hit points with semantic labels. That's point sampling, not a mesh. |
| **Static mesh geometry** | **No** | **Yes** | In the editor (source build): `File → Carla Exporter` (the `CarlaExporter` plugin) writes the selected meshes to one `.obj` in `Unreal/CarlaUE4/Saved`. It's documented for pedestrian-navigation meshes, and the docs don't describe material export. Unreal's own FBX/OBJ export is also editor-only. |
| **Vehicle / pedestrian meshes** | **No** | **Yes** | Asset export (FBX) from the editor. Pedestrians would also need per-frame skinning from `get_bones()`. |
| **Materials and textures** | **No (write-only)** | **Yes** | The API can only *apply* textures (`apply_textures_to_object`, etc.) to objects listed by `get_names_of_all_objects()`. `MaterialParameter` names Unreal's slots (Diffuse, Normal, AO/Roughness/Metallic/Emissive, Emissive) but nothing reads them back. Unreal material graphs would need a mapping to Mitsuba BSDFs. |
| **Weather / fog** | **Yes (read and write)** | — | `WeatherParameters`: `fog_density` (0–100), `fog_distance` (m), `fog_falloff`, `scattering_intensity`, `mie_scattering_scale`, `rayleigh_scattering_scale`, plus clouds, precipitation, wetness and sun angles. These are **Unreal artistic fog/sky controls, not physical extinction or scattering coefficients**. |
| **Lights** | **Yes** | — | `LightManager` / `carla.Light`: location, color, intensity (lumens), on/off, group. Enough to place Mitsuba emitters for street lights. |

**Existing Unreal → Mitsuba tooling (web search).** I found no maintained converter
from Unreal Engine to Mitsuba 3. The closest are the official
[`mitsuba-blender`](https://github.com/mitsuba-renderer/mitsuba-blender) add-on
(Blender ↔ Mitsuba XML, so a possible route is Unreal FBX → Blender → Mitsuba) and
[`Fbx2Mitsuba`](https://github.com/Poudingue/Fbx2Mitsuba), an older converter for 3ds
Max ASCII FBX. Mitsuba 3 loads `.obj`/`.ply` meshes directly, so a CarlaExporter
`.obj` would load without conversion. Materials would still need their own mapping.

---

## 3. Mitsuba and transient rendering

### Versions (per PyPI; nothing installed)

| Package | Latest | What `mitransient` would pull | Notes |
|---|---|---|---|
| `mitsuba` | **3.9.1** (2026-08-07), wheels cp310–cp314, manylinux_2_28 x86_64, ~63 MB | **3.8.0** (2026-02-23) | `mitransient` 1.3.0 requires `mitsuba>=3.6.0,<3.9.0`, so the latest Mitsuba is excluded |
| `drjit` | 1.5.0 (pinned by 3.9.1) | **1.3.1** (pinned by 3.8.0), ~4.5 MB | |
| `mitransient` | **1.3.0** (2026-03-10), pure Python | — | Optional extras: numpy, matplotlib, opencv-python |

Variants shipped in the pip wheels: `scalar_rgb`, `scalar_spectral`,
`scalar_spectral_polarized`, `llvm_ad_{rgb,mono,mono_polarized,spectral,spectral_polarized}`
and `cuda_ad_{rgb,mono,mono_polarized,spectral,spectral_polarized}`. Any other
variant (e.g. non-AD `cuda_mono`, or `_double`) needs a source build.

### Transient (time-resolved) rendering

`mitransient` ([repo](https://github.com/diegoroyo/mitransient),
[docs](https://mitransient.readthedocs.io), arXiv 2510.25660) is a Python add-on for
Mitsuba 3. Plugins, per its README:

| Type | Plugin | Relevance |
|---|---|---|
| Integrator | `transient_path` | Time-resolved path tracing, surfaces only |
| Integrator | **`transient_prbvolpath`** | **Time-resolved volumetric path tracing** (Path Replay Backpropagation, so it is differentiable). This is the one that handles fog. |
| Integrator | `transient_nlos_path` | Non-line-of-sight; not needed here |
| Film | `transient_hdr_film` | Output shape (width, height, `temporal_bins`, channels). Parameters `start_opl`, `bin_width_opl`, `temporal_bins` (default 2048). Time is expressed as optical path length. |
| Film | `phasor_hdr_film` | Frequency-domain output |
| Emitter | `angulararea` | Area emitter restricted to a cone (a possible laser/illuminator stand-in) |
| Sensor | `nlos_capture_meter` | Not needed here |

GPU use needs NVIDIA driver ≥ 495.89 (have 580). CPU vectorized use needs LLVM
≥ 11.1 (have 18–20). Both are satisfied.

### Volumetric / participating-media support (for fog)

Per the Mitsuba 3 docs:

- **Integrators:** `volpath`, `volpathmis` (spectral MIS, for spectrally varying
  extinction) and `prbvolpath`. Note that `path`, `direct` and `ptracer` do *not*
  handle media.
- **Media:** `homogeneous` and `heterogeneous` (grid volumes).
- **Phase functions:** `isotropic`, `hg` (Henyey-Greenstein), `rayleigh`, `sggx`,
  `tabphase` (tabulated), `blendphase`.

So single- or multiple-scattering fog can be expressed as a homogeneous medium with
an HG phase function. The single-scattering forward model in the plan would be an
analytic model compared against, or fitted to, these renders; Mitsuba doesn't
provide it as a mode.

**Spectral range caveat:** Mitsuba's `spectral` variants span **360–830 nm** (per the
variants doc). Common LiDAR wavelengths (e.g. 905 nm or 1550 nm) are outside that
range, so the SP-LiDAR pass would most likely use a `mono` variant with near-infrared
reflectances supplied separately. CARLA/Unreal textures are visible-light RGB.

---

## 4. Existing work in this repo

Search scope: the whole repo including gitignored files, third-party clones and
envs (`--no-ignore --hidden`, `.git` excluded), case-insensitive, across `.py`,
`.ipynb`, `.yaml/.yml`, `.json`, `.md`, `.txt`, `.toml`, `.cfg`, `.xml`, `.sh`, plus
directory names.

| Term | Files matched (whole repo) | Outside third-party envs / package caches |
|---|---|---|
| `carla` (word) | 1 | 1: `PROJECT_STATUS.md` (the pasted plan) |
| `mitsuba` | 1 | 1: `PROJECT_STATUS.md` |
| `mitransient` | 0 | 0 |
| XML scene (`<scene`, `scene….xml`) | 0 | 0 |
| `transient` | 334 | 2: `PROJECT_STATUS.md`, and `FULL_EXPERIMENT.md` ("transient" systemd unit, unrelated) |
| `single-photon` | 1 | 1: `PROJECT_STATUS.md` |
| `SPAD` (word) | 1 | 1: `PROJECT_STATUS.md` |
| `photon` | 12 | 1: `PROJECT_STATUS.md` |
| `beta-binomial` / `betabinom` | 39 | 1: `PROJECT_STATUS.md` (the env hits are statistics-library code) |
| `world sim` | 0 | 0 |
| `scene conversion` | 0 | 0 |

- **Directory names:** no directory named after CARLA, Mitsuba, transient, SPAD,
  photon or scene conversion. The "scene"/"sim"/"xml" hits are nuScenes folders,
  CUDA sample folders and Python's stdlib `xml` package inside env caches.
- **`.gitmodules`:** none anywhere in the repo; no submodules.
- **Requirements / env specs** (`methods/entropy_fusion/requirements.txt`,
  `methods/transfusion/{requirements.txt, research/requirements-pinned.txt,
  research/environment.yml}`, and the feasibility repos' `requirements.txt` /
  `environment.yml`): no reference to CARLA, Mitsuba, Dr.Jit or `mitransient`.
- **Notebooks:** 2 `.ipynb` files outside envs; neither matches any term.
- **Outside the repo:** no other repo under `~/github` mentions CARLA, Mitsuba or
  `mitransient`.

**Conclusion:** the plan in `PROJECT_STATUS.md` is the only trace of this pipeline on
disk. There is no code, config, external repo reference or data for it.

---

## 5. GPU: can this card run CARLA's renderer and Mitsuba's path tracer?

Nothing below is measured, since neither tool is installed.

- **CARLA 0.9.16 alone:** within the documented recommendation (8 GB of 12 GB).
- **CARLA 0.10.0 alone:** below the documented recommendation (16 GB).
- **Mitsuba `cuda_ad_*` alone:** memory depends on the scene and on the transient
  film. The film is a dense (width × height × time bins × channels) float32 buffer.
  Illustrative arithmetic, per stored channel:

  | Resolution | Time bins | Film per channel |
  |---|---|---|
  | 128 × 1024 (scanning-LiDAR-like) | 1,000 | 0.49 GiB |
  | 512 × 512 | 2,048 (`mitransient` default) | 2.0 GiB |
  | 1920 × 1080 | 1,000 | 7.7 GiB |

  The film also carries alpha/weight channels (the source builds channel names
  like `0123` + `AW`), so real use is a multiple of these numbers, before geometry,
  textures and the wavefront working set.
- **CARLA and Mitsuba at the same time:** **flagged as not realistic** at useful
  transient resolutions. CARLA's recommended 8 GB leaves about 4 GB, less than one
  512 × 512 × 2048 film with its extra channels. The pipeline doesn't need them
  concurrent: CARLA produces scene state, then shuts down, then Mitsuba renders
  offline.
- **Alongside a training job:** **flagged.** Heavy BEV training (SAMFusion,
  TransFusion as shipped) used nearly the whole card and would block both. A light
  job like the entropy-fusion model (0.67 GiB) might fit next to Mitsuba memory-wise,
  but compute contention would slow both (not measured).
- **CPU fallback:** Mitsuba's `llvm_ad_*` variants run on the 24-thread CPU (LLVM
  present). That keeps the GPU free for training at a large, unmeasured speed cost.

---

## 6. Disk space

Measured with `df`/`du` on 2026-09-23. Everything is on one ext4 partition.

| | Size |
|---|---|
| **Root partition** | 915 GB total, 692 GB used, **178 GB free (80% used)** |
| `~/data/nuscenes` | 402 GB |
| `~/data/kitti_full` + `~/data/kitti` + `~/data/kitti_tools` | 20 GB + 0.1 GB + 0.2 GB. **About 20 GB total, not ~60 GB.** |
| `~/data/waymo` | 6.4 GB |
| `~/data/seeing_through_fog` (+ toolkit) | 12 KB (+ 0.4 GB) |
| `feasibility/` (envs, clones, package caches) | 46 GB |
| `methods/entropy_fusion/outputs/` (checkpoints, ~1.3 GB `predictions.json` files) | 35 GB |
| `~/venvs/*` | ~20 GB |
| `~/.cache/pip` | 5.0 GB |
| Docker images | 6.4 GB |

Headroom for the new pipeline (per docs, plus the arithmetic above):

| Addition | Size | Free space after |
|---|---|---|
| Mitsuba 3.8.0 + Dr.Jit + mitransient wheels | < 0.1 GB download | ~178 GB |
| CARLA 0.9.16 package | ~20 GB (+ optional AdditionalMaps) | ~158 GB |
| CARLA 0.9.16 source build (UE 4.26 fork + CARLA) | ~130 GB | **~48 GB** |
| CARLA 0.10.0 | 130 GB | ~48 GB |
| Dense transient outputs, 128 × 1024 × 1,000 bins, float32, 1 channel | ~0.52 GB per frame | 178 GB holds **about 340 frames** before anything else is installed |

---

## Open questions before scaffolding

1. **Which CARLA line, and package or source build?** 0.9.16 fits the GPU on paper
   but is untested on Ubuntu 24.04. 0.10.0 recommends 16 GB of VRAM. Reading meshes
   and materials needs a **source build** (~130 GB, Epic-linked GitHub account, 4+ h
   build), which would leave about 48 GB free.
2. **Where does the Unreal editor run?** Mesh export is editor-only, and this box has
   no display server. Options: a remote desktop or virtual display here, or a
   one-time export on another machine.
3. **Docker or native?** The `carlasim/carla` images are Ubuntu 22-based and would
   avoid the 24.04 support gap, but need the NVIDIA container toolkit (sudo).
4. **What "prior Unreal-to-Mitsuba work" does the bridge build on?** I found no
   maintained converter. The nearest are `mitsuba-blender` and `Fbx2Mitsuba`. Knowing
   the intended starting point changes what the "contribution" part of the bridge is.
5. **Material policy.** CARLA exposes no material data through the API, and Unreal
   material graphs don't map one-to-one onto Mitsuba BSDFs. Also, which
   **near-infrared reflectances** will the SP-LiDAR pass use, given that Mitsuba's
   spectral mode stops at 830 nm and CARLA's textures are visible-light RGB?
6. **Fog parameters.** CARLA's fog settings are artistic, not physical. Is CARLA's
   weather used at all, or only as metadata, with the Mitsuba medium (extinction,
   albedo, HG *g*) defined independently? Does CARLA's own camera output feed into the
   dataset, or is CARLA only the scene-state source?
7. **Mitsuba version pin.** Pin `mitsuba==3.8.0` for `mitransient` 1.3.0, or wait
   for a `mitransient` release that supports 3.9?
8. **SP-LiDAR sensor definition.** Wavelength, flash vs scanning, resolution, field of
   view, time-bin width and bin count, and pulse shape. These set the film size, VRAM
   and disk per frame.
9. **Storage format and budget.** Store dense float32 histograms, sampled photon
   counts (e.g. uint16), or sparse/cropped windows? At dense float32, 178 GB holds
   hundreds of frames, not thousands. Is freeing space (feasibility envs 46 GB,
   old run outputs 35 GB) on the table?
10. **GPU scheduling.** Run CARLA and Mitsuba strictly one after the other, with no
    training during rendering? Or render on the CPU (`llvm`) when the GPU is busy?
11. **Dynamic actors in v1.** Rigid vehicles only, or animated pedestrians too? The
    latter needs skeletal-mesh export plus per-frame skinning from `get_bones()`.
