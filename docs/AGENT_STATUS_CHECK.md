# Agent status check — disk cleanup, leftover processes, CARLA source clone

Checked 2026-09-23 ~18:02 CDT / 23:02 UTC. Read-only: nothing was deleted, killed,
cloned, or built. The previous version of this file (17:27 CDT) is replaced
by this check. The Mitsuba and CARLA package smoke tests were not re-run.

---

## 1. Disk cleanup — **done**

`feasibility/` was cleaned. It was **46 G** at the earlier readiness check.
It is **104 K** now. `~/.cache/pip` was **5.0 G**; the directory is still
there and is empty (**4.0 K**, the directory itself).

### Result files still present (all seven methods)

| Method | Still on disk |
|---|---|
| seeing-through-fog | `RESULTS.md` |
| raf | `RESULTS.md` |
| samfusion | `RESULTS.md`, `result.json` |
| bevfusion | `RESULTS.md`, `result.json` |
| rcbevdet | `RESULTS.md`, `result.json` |
| transfusion | `RESULTS.md`, `result.json`, `result_as_shipped.json`, `result_spconv256.json` |
| afw-net | `RESULTS.md` |
| (summary) | `feasibility/SUMMARY.md` |

Those files' sizes match the copies that were there before the cleanup
(same byte sizes as in the earlier read). Methods that never had a
`result.json` (seeing-through-fog, raf, afw-net) still don't.

### What is gone

Working tree now: 14 files, all of them the RESULTS / result JSON /
SUMMARY files above. Nothing else.

**Untracked bulk (the 46 G).** Not in git, so `git status` does not list
it. Each method directory previously held micromamba envs (`env` /
`env2`), cloned upstream repos, and package caches (`mamba`,
`mamba_root`, RAF `venv`). Those directories are gone. What remains per
method is only the result write-ups.

**Tracked files deleted from the working tree, still in the git index**
(unstaged `D`, not committed):

- `feasibility/.gitignore`
- harness: `bench_common.py`, `bench_mmengine.py`, `install_cuda.sh`,
  `make_summary.py`, `run_when_free.sh`, `orchestrator.log`
- per-method `bench.sh` / `build_env.sh` / `build_ops.sh` /
  `rebuild_clean.sh` / `prep_mini.*` / `patches/` for bevfusion,
  rcbevdet, samfusion, transfusion

So the cleanup removed more than the envs. The build scripts and patches
that were committed are deleted on disk and would disappear from the repo
if that deletion is committed. This check did not restore them.

### Free space

| When | `df -h /` available |
|---|---|
| Previous check (~17:27 CDT) | **56 G** free (813 G used, 94%) |
| Now | **80 G** free (789 G used, 91%; 79.61 GiB) |

About **+24 G**. That is consistent with removing ~46 G of feasibility
plus ~5 G of pip cache, then the CARLA source tree landing at 27 G
(it was only a partial `.git` when the 56 G figure was taken).

---

## 2. Leftover processes

### `UnrealVersionSelector-Linux-Shipping -register` — **still the same PIDs**

| PID | PPID | State | Elapsed | CPU time | %CPU | wchan |
|---|---|---|---|---|---|---|
| 348692 | 1 | S (sleeping), 87 threads | 1:05:44 | 3:41 | 5.6 | `do_wait` |
| 348889 | 348692 | S (sleeping), 1 thread | 1:05:43 | 0:00 | 0.0 | `poll_schedule_timeout` |

They were **not** killed. Same PIDs as the 16:57 CDT start.

Not a compiler and not the UE `make` (that finished at 17:21 CDT). The
child is idle. The parent is sleeping in `do_wait` but has used 3 min 41 s
of CPU over 66 minutes, so it is not a pure zero-CPU wait either. Nothing
was done because this check was not allowed to kill or restart anything.

### Poll loop for build wrapper pid 345849 — **not found**

Pid 345849 is gone (it was already gone at 17:27, after `make` exited 0).
No process now matches `poll make` or `ue_build.pid`. This check did not
kill one. The loop was written to exit on its next 3-minute wake once
345849 was dead, so the consistent reading is that it exited on its own
after the UE build finished. I did not see that exit.

---

## 3. CARLA source clone — **finished**

Pids 399551, 399552, and 399556 are gone. No `git clone` into
`~/carla/carla-0.9.16-src` is running.

The working tree is checked out, not just `.git`. Top level includes
`LibCarla/`, `PythonAPI/`, `Unreal/`, `Docs/`, `Makefile`, `README.md`,
`CHANGELOG.md`. `git status` is clean.

| | |
|---|---|
| Path | `~/carla/carla-0.9.16-src` |
| HEAD | detached, **tag `0.9.16`** |
| Commit | `294096eb1c38eabf246e4f3a9cdab704e33a7f4c` |
| Subject | `0.9.16 release docs updates (#9272)` (2025-09-16) |
| Size | **27 G** (`du -sh`). `Unreal/` alone is 24 G and already contains `Content/Carla` (Maps, Static, HDMaps, Blueprints, …). |
| Clone log | `~/carla/logs/carla_clone.log` ends with `Updating files: 100% (2725/2725)` and `elapsed_sec=65.84`. The log itself does not contain a `clone_exit=` line. |

Shallow clone (`--depth 1`); git reports the commit as grafted.

---

## 4. Next build step — **not started**

Yes. The UE editor binary is a separate build from compiling CARLA
against it. From the 0.9.16 tree's own `Docs/build_linux.md` ("Build
CARLA with Make"), run from the CARLA repo root, after `UE4_ROOT` points
at the already-built editor (`export UE4_ROOT=~/UnrealEngine_4.26` in
that doc):

1. **Python client** (separate from the editor): `make PythonAPI`
2. **Server / editor project** — the command that compiles CARLA and
   opens the Unreal editor:

```sh
make launch
```

The same page's table: "`make launch` — Launches CARLA server in Editor
window." It also says to run that command each time you want the editor.
`make package` is the packaged-server build, not the editor launch.
`make rebuild` is `make clean` plus `make launch`.

Documented prerequisites that are not the `make` line itself:

- `CARLA_UE4_ROOT` set to this clone (or substitute the path).
- CARLA **content** downloaded into
  `Unreal/CarlaUE4/Content/Carla` (`./Update.sh`, or the git/archive
  options in that section) **before** the editor build.

Content **looks** present (24 G under `Unreal/`, with `Maps/` and
`Static/`). This check did not compare it to `Util/ContentVersions.txt`,
so completeness is not confirmed.

Not started. Disk free is 80 G.

---

## 5. Sanity check — **unchanged**

| Item | Result |
|---|---|
| Xvfb pid 345407 | **Still that PID.** `Xvfb :99 -screen 0 1280x1024x24 -ac -nolisten tcp`, PPID 1, started 16:45 CDT. |
| `UE4Editor` | **Still present.** `/home/soham/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor`, ELF x86-64, **983,888 bytes**, mtime **2026-09-23 17:21:23 -0500** (same size and mtime as when `make` finished). Tree still **93 G**. Cleanup did not touch it. |

---

## Snapshot

| Item | Label |
|---|---|
| feasibility cleanup | **done** — result files kept; envs/repos and also tracked harness/patches removed from disk |
| pip cache | **cleared** (empty dir left) |
| free disk | **80 G** (was 56 G) |
| UnrealVersionSelector 348692 / 348889 | **still running**, sleeping, not a build; not killed |
| poll loop for 345849 | **not found** |
| CARLA 0.9.16 source clone | **finished** — tag `0.9.16`, commit `294096eb`, 27 G |
| `make launch` | **not started** |
| Xvfb :99 and UE4Editor | **still up / untouched** |
