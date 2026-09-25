# CARLA 0.9.16 smoke test (simulator + Python API)

Measured 2026-09-23 on this machine. This is a **run**, not a paper review:
the prebuilt Linux package and the `carla==0.9.16` wheel were installed, the
server was launched headless, and a Python client ticked a small scene. It
does **not** test mesh export, the Unreal editor, AdditionalMaps, other towns,
LiDAR, weather, or a source build.

Replay script: `sim/carla_smoketest/smoketest.py`. JSON + captured frames live
under `sim/carla_smoketest/outputs/` (gitignored). Installs are outside the
repo: server at `~/carla/CARLA_0.9.16`, client venv at `~/venvs/carla`.

---

## Bottom line

| Question | Answer |
|---|---|
| Did the 0.9.16 **package** launch on Ubuntu 24.04.3, which CARLA does not officially support? | **Yes.** `./CarlaUE4.sh -RenderOffScreen -nosound` bound ports 2000/2001/2002 in 3.7 s. No extra packages, no `LD_LIBRARY_PATH`, no Vulkan ICD filter, no Docker, no virtual display. |
| Did ticking and spawning work in synchronous mode? | **Yes.** `synchronous_mode=True`, `fixed_delta_seconds=0.05`, `no_rendering_mode=False`. Five 4-wheel vehicles spawned on the first try (155 spawn points, 33 four-wheel blueprints). 100 ticks advanced simulation time by exactly 5.000 s; vehicles moved 16.9–29.5 m. |
| Did the spawned RGB camera return real image data? | **Yes.** 1280×720 BGRA, 3,686,400 raw bytes, 65,967 unique colors (reload run), mean RGB ≈ 136 / 124 / 117. All 100 frames matched the tick. The saved PNG is a rear view of a red Prius on Town10HD, not a black or constant frame. |
| GPU memory vs the docs' **8 GB** claim | Idle default town: **~5.8–5.9 GB**. After `load_world` of the *same* town: **~9.9 GB**. Sampler peak: **10,034 MiB** of 12,288 MiB (81.7%). The 8 GB figure is a recommended minimum, not measured use. A card with exactly 8 GB would not have completed the reload run. |
| Workarounds needed for 24.04? | **None for the server.** One client-script trap: `ActorAttribute.as_str()` raises on non-string attributes such as `tint` (Float). Dispatch on `a.type` instead. |

---

## What was installed

| Item | Where | Size / identity |
|---|---|---|
| Prebuilt server | `https://downloads.carlasim.com/Linux/CARLA_0.9.16.tar.gz` → `~/carla/CARLA_0.9.16` | 8,346,095,504 bytes compressed (7.8 GiB); **19 G** extracted. `VERSION` = `0.9.16`. Engine banner: `4.26.2-0+++UE4+Release-4.26`. |
| Python client | `~/venvs/carla`, `pip install carla==0.9.16` | `carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl` from PyPI. `import carla` succeeds on system Python 3.12.3. Client and server both report version `0.9.16`. |
| AdditionalMaps | not downloaded | Default town is in the base package. |
| Source / Unreal editor | not installed | Not needed for this test. |

The package also ships a cp312 wheel under `PythonAPI/carla/dist/`. It is not
byte-identical to the PyPI wheel (different `libcarla` SHA, dates 2025-09-15 vs
2025-09-13). The test used the PyPI wheel. `ldd` on
`CarlaUE4-Linux-Shipping` reported no missing libraries.

Disk after the install: 141 GB free of 915 GB (84% used). Before: 177 GB free.

---

## Server launch on Ubuntu 24.04

Command, from `~/carla/CARLA_0.9.16`:

```
./CarlaUE4.sh -RenderOffScreen -nosound
```

Stdout is only:

```
4.26.2-0+++UE4+Release-4.26 522 0
Disabling core dumps.
```

Three launches, all successful. Port 2000 opened 3.7 s after the first start.
`nvidia-smi` listed `CarlaUE4-Linux-Shipping` as type `C+G` on the RTX 3080
(the only GPU; `/dev/dri` is that card). Mesa's lavapipe / Intel Vulkan ICDs
are installed on this box but were not used.

SIGINT (`kill -INT`) always produced UE4's `Exiting abnormally (error code: 130)`
and `SERVER_EXIT=130`. That is how the shipping binary reports Ctrl-C, not a
crash. Device memory returned to 1 MiB after each stop.

No DISPLAY, Xorg, Xwayland, xvfb, or NVIDIA container toolkit was involved.

---

## Client: default town, spawn, tick, camera

```
~/venvs/carla/bin/python sim/carla_smoketest/smoketest.py \
    --vehicles 5 --ticks 100 --dt 0.05
```

`--no-reload` skips `client.load_world` and uses the map the server booted
with. Two complete runs:

| | Run 3 (`load_world` of Town10HD_Opt) | Run 4 (`--no-reload`) |
|---|---|---|
| Map | `Carla/Maps/Town10HD_Opt` (boot and loaded) | same |
| Settings applied | sync on, `fixed_delta_seconds=0.05`, rendering on | same |
| Vehicles | Prius, Charger 2020, Wrangler, MKZ 2017, Fuso Rosa | same set (seed 0) |
| Ticks | 100; frames 5019 → 5119 | 100; frames 99455 → 99555 |
| Sim time advanced | 5.000000 s | 5.000000 s |
| Camera frames matched / missing | 100 / 0 | 100 / 0 |
| Vehicle displacement (m) | 25.904 / 29.523 / 23.731 / 21.675 / 16.888 | identical |
| Median `world.tick()` wall | 1.9 ms | 1.9 ms |
| `load_world` / `get_world` | 5.14 s | 8 µs |
| Image unique colors | 65,967 | 71,217 |
| Image mean RGB | 136.03 / 123.76 / 116.99 | 136.67 / 124.56 / 117.98 |
| Fraction of all-zero pixels | 0.065% | 0.065% |

Maps present without AdditionalMaps: Town01–05 and each `_Opt`, Town10HD and
Town10HD_Opt, plus `AnnotationColorLandscape`.

The camera was `sensor.camera.rgb`, 1280×720, FOV 90, attached 6.5 m behind
and 2.8 m above vehicle 0. `raw_data` is 1280 × 720 × 4 BGRA. Alpha is 255
everywhere. Saved frames:

- `sim/carla_smoketest/outputs/run3_reload/rgb_frame_5119.png`
- `sim/carla_smoketest/outputs/run4_noreload/rgb_frame_99555.png`

Both show the same red hatchback on a Town10HD intersection, with buildings,
sky, road markings and lighting. This is rendered content, not an empty
sensor (`no_rendering_mode` was false).

---

## GPU memory vs the 8 GB claim

All numbers are **device-wide** `nvidia-smi --query-gpu=memory.used` in MiB
on the RTX 3080 (12,288 MiB). A 500 ms sampler ran for the whole session
(`~/carla/logs/gpu_samples.csv`, 1,430 samples). Process rows are from
`nvidia-smi` at the same instants.

| Phase | Device used (MiB) | CARLA process (MiB) |
|---|---|---|
| GPU idle, no CARLA | 1 | — |
| Fresh server, default town, no client (steady) | 5,758–5,951 (typical 5,920–5,934) | 5,826–5,840 |
| After `load_world(Town10HD_Opt)` on that same town | 9,895–9,898 | 9,705–9,751 |
| Mid-100-ticks after reload | 9,818 | 9,719 |
| End of those 100 ticks | 8,720 | 8,621 |
| Sampler **peak** (during reload run) | **10,034** | — |
| No-reload: spawn + 100 ticks | 6,122 | 6,028 |
| After SIGINT | 1 | — |

Against `docs/CARLA_MITSUBA_READINESS.md`'s "docs recommend 8 GB":

- Idle Town10HD_Opt is **under** 8 GB (~5.8 GB).
- `load_world` of Town10HD_Opt, even when it is already the loaded map, jumps
  to **~9.9 GB** and peaked at **10,034 MiB** (9.80 GiB). That is **above**
  8 GB. Headroom on this 12 GB card at the peak was about 2.2 GB.
- Five vehicles plus one 1280×720 RGB camera on top of a *already-booted*
  town stay around **6.1 GB**.

`load_world` is the expensive step, not the five cars. For a smoke test of
the boot map, `--no-reload` is enough and stays under 8 GB.

---

## Errors and non-issues

**Needed a fix in the client script (not the server):**

```
RuntimeError: tint: bad attribute cast: cannot convert to String
```

`sensor.camera.rgb` attribute `tint` is `ActorAttributeType.Float`.
`as_str()` only works on String attributes. The first two client attempts
died after vehicles had already spawned. The server stayed up. The script
now reads Bool/Int/Float/RGBColor/String through the matching `as_*()`.

**Did not need workarounds:**

- Ubuntu 24.04.3 / glibc 2.39 / kernel 7.0.0-31-generic
- NVIDIA driver 580.173.02, Vulkan ICD already present
- Headless / no display server
- Python 3.12 vs the docs' 3.7–3.12 range (wheel exists)

**Observed but not a problem for this test:**

- Shipping-binary stdout is two lines; no `CarlaUE4/Saved/Logs` file appeared.
- SIGINT is logged as "Exiting abnormally (error code: 130)".
- A parallel Mitsuba process briefly used 224 MiB between server 2 and 3.
  It was gone before server 3's measured idle / no-reload numbers.

---

## Follow-up: same camera, three weather presets

Ran 2026-09-23 17:03 CDT after polling until the card was empty (Mitsuba
`analyze.py` had just finished). Script: `sim/carla_smoketest/weather_fog.py`.
No `load_world`. Fixed camera at spawn point 0, 2.2 m up, looking along the
lane. 20 ticks per preset. Frames:

`sim/carla_smoketest/outputs/weather_fog/{clear,fog_mid,fog_max}.png`

| Preset | `fog_density` | mean RGB | unique colors | Device used (MiB) |
|---|---|---|---|---|
| clear | 0 | 139.6 / 128.3 / 113.1 | 98,864 | 5,995 |
| fog_mid | 50 | 155.0 / 154.3 / 159.1 | 20,839 | 6,223 |
| fog_max | 100 | 175.7 / 175.3 / 180.7 | 13,425 | 6,231 |

`set_weather` round-trips: the values written are the values read back
(`fog_density` 0 / 50 / 100, `scattering_intensity` 0 / 1 / 1,
`mie_scattering_scale` 0 / 1 / 1). Fog is a **veil**: mid already hides the
far tower; max whites out the street. Contrast collapses (std RGB 62 → 36)
and the colour count drops. This is Unreal exponential height fog, not a
physical medium — usable as RGB context, not as SP-LiDAR fog ground truth.
VRAM stays ~6.0–6.2 GB without `load_world`, same band as the no-reload
spawn run.

---

## How to repeat

```
# once
python3 -m venv ~/venvs/carla
~/venvs/carla/bin/pip install carla==0.9.16 numpy pillow
# server already extracted at ~/carla/CARLA_0.9.16

cd ~/carla/CARLA_0.9.16
./CarlaUE4.sh -RenderOffScreen -nosound

# other terminal, repo root
~/venvs/carla/bin/python sim/carla_smoketest/smoketest.py \
    --no-reload --vehicles 5 --ticks 100 --dt 0.05
# optional: same camera, clear / fog_mid / fog_max
~/venvs/carla/bin/python sim/carla_smoketest/weather_fog.py
```

Stop the server with Ctrl-C or `kill -INT` on `CarlaUE4-Linux-Shipping`.
The server is not left running; GPU memory after this write-up is 1 MiB.
