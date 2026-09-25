# CARLA editor / export — restore, content, `make launch`, exporter

Checked 2026-09-23 ~18:10 CDT / 23:10 UTC. This replaces the earlier write-up
in this file. Tracked files under `feasibility/` were restored. `make launch`
was run once and failed. The display was captured. The Carla Exporter plugin
was read. No mesh was exported, and no workaround was applied to the build
failure.

---

## 1–3. Restore tracked `feasibility/` files — **done**

`git status -- feasibility/` before the restore was exactly the unstaged
deletions from the disk cleanup. 31 files, 826 lines, nothing else:

- `feasibility/.gitignore`
- harness: `bench_common.py`, `bench_mmengine.py`, `install_cuda.sh`,
  `make_summary.py`, `run_when_free.sh`, `orchestrator.log`
- bevfusion: `bench.sh`, `build_env.sh`, `build_ops.sh`, `rebuild_clean.sh`,
  `patches/bench_mini.yaml`, `patches/bevfusion-fixes.patch`
- rcbevdet: `bench.sh`, `build_env.sh`, `build_ops.sh`, `prep_mini.py`,
  `patches/bench_mini.py`, `patches/requirements_runtime.txt`
- samfusion: `bench.sh`, `build_env.sh`, `build_env2.sh`, `install_rest.sh`,
  `prep_mini.sh`, `patches/samfusion-fixes.patch`
- transfusion: `bench.sh`, `build_env.sh`, `build_ops.sh`, `rebuild_clean.sh`,
  `patches/bench_mini_voxel_L.py`, `patches/transfusion-fixes.patch`

Restored with `git restore -- feasibility/` only. `RESULTS.md`,
`result*.json`, and `SUMMARY.md` were already on disk and were not touched.
The deleted micromamba envs, cloned repos, and package caches were not in
git, so they stayed deleted.

After the restore, `git status -- feasibility/` is empty. Spot-check of
restored bytes against the HEAD blobs:

| File | Bytes on disk | HEAD blob | Match |
|---|---|---|---|
| `feasibility/bevfusion/patches/bevfusion-fixes.patch` | 4948 | `22247f94fc6c` | yes |
| `feasibility/samfusion/patches/samfusion-fixes.patch` | 1948 | `54661715bb44` | yes |
| `feasibility/bench_common.py` | 6441 | `bc368b285e54` | yes |
| `feasibility/transfusion/patches/transfusion-fixes.patch` | 2456 | `94be5d680d28` | yes |

`git status` outside `feasibility/` is unchanged: untracked
`PROJECT_STATUS.md`, `docs/`, and `sim/` only.

---

## 4. CARLA content — **matches the 0.9.16 stamp**

`Util/ContentVersions.txt` is not a file listing. It is a table of archive
ids. `Update.sh` takes the last non-empty line (`Latest: 20250912_2171890`),
which is the same id as the `0.9.16:` line, and compares it to
`Unreal/CarlaUE4/Content/Carla/.version`.

| Check | Result |
|---|---|
| `.version` | `20250912_2171890` |
| `0.9.16` and `Latest` in `ContentVersions.txt` | `20250912_2171890` |
| `./Update.sh` | `Content is up-to-date.` exit 0. It did not re-download. |
| Git content checkout | No. There is no `Content/Carla/.git`. |
| Size | 24 G |
| Top level | `Blueprints` (378 files), `Config` (11), `HDMaps` (8), `Maps` (817), `road_xodr` (30), `Static` (41944), plus `hooks/`, `HoudiniEngine/`, `LICENSE`. Dates on those directories are 12 Sep 2025, the same day as the archive id. |

By the check CARLA itself uses, the installed content is the 0.9.16 package.
The repo has no checksum or member list of that tarball, so this is not a
byte-for-byte comparison against the archive.

---

## 5. `make launch` — **failed in `setup`, exit 2, 9.59 s**

From `~/carla/carla-0.9.16-src`, with `DISPLAY=:99` and
`UE4_ROOT=/home/soham/UnrealEngine_4.26`:

```
/usr/bin/time -f 'elapsed_sec=%e' make launch
```

| | |
|---|---|
| Start | 2026-09-23T23:09:56Z |
| End | 2026-09-23T23:10:05Z |
| Wall clock | **9.59 s** |
| Exit code | **2** (`make_exit=2`). Make reports the recipe as `setup` **Error 1**; GNU make then exits 2. |
| Log | `~/carla/logs/make_launch.log` |

`make launch` depends on `setup` (`Util/BuildTools/Linux.mk`). Setup died
while building Boost's B2 engine with the UE-bundled compiler
`.../v17_clang-10.0.1-centos7/.../clang++`. That clang 10 linker rejects
this machine's glibc 2.39 libraries:

```
x86_64-unknown-linux-gnu-ld: /lib/x86_64-linux-gnu/libm.so.6: unknown type [0x13] section `.relr.dyn'
x86_64-unknown-linux-gnu-ld: cannot find /lib/x86_64-linux-gnu/libm.so.6
... same for libmvec.so.1 ...
Failed to build B2 build engine
```

The editor binary was never started. `pgrep UE4Editor` found nothing.
LibCarla, the CARLA project, and the editor window were not reached.
No compiler workaround was applied.

---

## 6. Display `:99` after that launch — **not an Unreal Editor**

Capture uses the same tool as the earlier xclock check: `xwd -root` on
`:99`, converted to PNG.

`docs/carla-recon/display99-after-make-launch.png` (1280×1024).

What it shows: a black root window, and one small dialog centered on the
screen. Olive background, yellow text: **"Register Unreal Engine file
types?"** with **Yes** and **No** buttons. Window title `File Types`,
geometry 268×100+506+308, mapped and viewable. Five unique colors in the
whole frame. No editor chrome, no viewport, no menu bar, no 3D scene.

That dialog belongs to the leftover
`UnrealVersionSelector-Linux-Shipping -register` processes (pids 348692
and 348889). Both have `DISPLAY=:99` in their environment and have been
up since 16:57 CDT. Neither button was clicked.

---

## 7. How Carla Exporter is invoked — **editor menu only**

Plugin: `Unreal/CarlaUE4/Plugins/CarlaExporter/`.

`CarlaExporter.uplugin` has one module, `Type: Editor`, `LoadingPhase:
Default`. There is no commandlet module and no Python module.

`FCarlaExporterModule::StartupModule` (`Private/CarlaExporter.cpp`) does
three things:

1. `FCarlaExporterCommands::Register()`
2. Maps `PluginActionExportAll` to `PluginButtonClicked`
3. Adds a Level Editor menu extension on the hook `"FileActors"`,
   `EExtensionHook::After`

`RegisterCommands` (`CarlaExporterCommands.cpp`) defines one UI command:

```
UI_COMMAND(PluginActionExportAll, "Carla Exporter",
  "Export all or selected meshes into an .OBJ file ...",
  EUserInterfaceActionType::Button, FInputGesture());
```

`FInputGesture()` is empty, so there is no default key chord. Nothing in
the plugin calls `IConsoleManager`, `FAutoConsoleCommand`, or a
`UFUNCTION(Exec)`. There is no `*Commandlet*` type. The only caller of
`PluginButtonClicked` is that menu action.

`PluginButtonClicked` needs a live editor world (`GEditor->GetEditorWorldContext().World()`).
With no actors selected it exports every actor except those tagged
`NoExport`, and writes `<mapName>.obj` under the project `Saved/`
directory. That function is not exposed except through the menu button.

CARLA's own doc says the same thing.
`Docs/tuto_M_generate_pedestrian_navigation.md` step 3:

> Press `ctrl + A` to select everything and export the map by selecting
> `File` -> `Carla Exporter`. A `<mapName>.obj` file will be created in
> `Unreal/CarlaUE4/Saved`.

No console command, commandlet, or editor-Python hook exists in this
plugin. It is a File-menu button. Export was not attempted.

---

## Snapshot

| Step | Result |
|---|---|
| Restore `feasibility/` | **Done.** Status clean. Patch and harness sizes match HEAD. Envs/caches left deleted. |
| Content vs `ContentVersions.txt` | **Up to date.** `.version` is `20250912_2171890`, the 0.9.16 / Latest id. `Update.sh` agreed. |
| `make launch` | **Failed.** 9.59 s, exit 2, Boost B2 vs clang 10 / glibc 2.39. Editor not started. |
| Display capture | Black screen plus the UnrealVersionSelector "Register Unreal Engine file types?" dialog. Not the editor. |
| Carla Exporter | **File → Carla Exporter only.** No scriptable entry point. Export not run. |

---

## Linker workaround — 2026-09-23 18:40 CDT / 23:40 UTC

The `make launch` failure is the UE-bundled linker rejecting glibc 2.39
RELR relocations. This pass tried the narrow gold-linker workaround and
stopped at the next, different error. `make launch` was not run again.

### Issue 7991

<https://github.com/carla-simulator/carla/issues/7991> is the same
`.relr.dyn` / `unknown type [0x13]` failure, from the same kind of
UE-bundled CentOS clang. It is not the same binary path. The report is
UE5 `v22_clang-16.0.6-centos7/.../x86_64-unknown-linux-gnu-ld` failing
on `libc.so.6` during CARLA's `Setup.sh` / CMake compiler test. Ours is
UE4 `v17_clang-10.0.1-centos7/.../x86_64-unknown-linux-gnu-ld` failing
on `libm.so.6` while building Boost's B2 engine.

Nobody in that thread posted a working linker workaround.

- The author closed it (2024-07-29): switch to Ubuntu 22.04 or Windows.
  They called replacing the UE clang with the system clang a "crude
  solution" that also means reconfiguring the UE compiler. They did not
  post that change.
- J160KU: use Ubuntu 22.04, the version the docs recommend.
- Gragonfly (2025-01-18): same problem with **ue4.26 and Ubuntu 24.04**.
  No reply.
- Later comments are a Windows CMake/`rc.exe` failure, and another
  "still broken on Ubuntu 24.04" with no fix.
- LukasHaefele (2026-01-27) said an Arch fork built after "minor tweaks"
  and did not paste the patch.

### Where the failing link is

`make setup` / `make launch` both hit `setup` in
`Util/BuildTools/Linux.mk` line 142, which runs `Util/BuildTools/Setup.sh`.

`Setup.sh` puts the UE clang 10 on `PATH` and sets `CC`/`CXX` to it
(lines 55–57). Boost is not yet installed, so it extracts Boost 1.84.0
and runs `./bootstrap.sh --with-toolset=clang` (line 182, before this
edit).

`bootstrap.sh` then builds the B2 engine with this exact line, which
clears `CXX` and `CXXFLAGS`:

```
CXX= CXXFLAGS= "$my_dir/tools/build/src/engine/build.sh" ${TOOLSET}
```

`tools/build/src/engine/build.sh` does not read `LDFLAGS`. The probe it
prints is `clang++ -x c++ -std=c++11 check_cxx11.cpp`, and the real
link is the same `clang++` with the engine sources `-o b2`. That
`clang++` is the UE one, and it calls the bundled
`x86_64-unknown-linux-gnu-ld`, not the system `ld`.

There is no `project-config.jam` yet at that point. Bootstrap writes one
only after B2 exists, and `Setup.sh` then overwrites it with a `using
python` line. The two `./b2 toolset=clang-10.0 ...` lines (the library
build) never ran in the original failure.

### Linker chosen

`/usr/bin/ld.gold` is already on the machine (GNU gold 1.16, from the
`binutils` package, binutils 2.42). The separate `binutils-gold` package
is not installed and was not needed. `sudo` in this shell still asks for
a password, so nothing was installed. `lld` is not installed as a system
package. The UE clang directory does contain its own `ld.lld`; it was
not used.

A one-file probe with that same `clang++` and `-fuse-ld=gold` linked
`/tmp/relr_probe_gold` (7688 bytes) by invoking `/usr/bin/ld.gold`. Gold
is what this pass used, because it was already present and that probe
linked.

### `LDFLAGS` does not reach B2

`make setup` with `LDFLAGS=-fuse-ld=gold` and no file edits:
**9 s, exit 2**, log `~/carla/logs/make_setup_ldflags.log`. The compiler
line was still `clang++ -x c++ -std=c++11 check_cxx11.cpp`, still the
bundled `ld`, still `.relr.dyn`.

### Edit that does reach it

In `~/carla/carla-0.9.16-src/Util/BuildTools/Setup.sh` only:

- After extract, before `./bootstrap.sh`, sed the bootstrap line so the
  engine build is `build.sh --cxxflags="-fuse-ld=gold"`.
- Both `./b2` invocations gained `linkflags="-fuse-ld=gold"`.

### `make setup` after that edit

Isolated target, not `make launch`. Log
`~/carla/logs/make_setup_gold.log`. **23.96 s, exit 2.**

The engine link did use gold:

```
clang++ -x c++ -std=c++11 -fuse-ld=gold ... -o b2
```

Bootstrap finished (`Bootstrapping is done`). `libboost_system.so.1.84.0`
and `libboost_python312.so.1.84.0` linked. Then a different error, still
inside this Boost build, compiling `libboost_numpy` against the user
site-packages NumPy:

```
libs/python/src/numpy/dtype.cpp:101:83: error: no member named 'elsize' in '_PyArray_Descr'
int dtype::get_itemsize() const { return reinterpret_cast<PyArray_Descr*>(ptr())->elsize;}
```

Include path:
`/home/soham/.local/lib/python3.12/site-packages/numpy/_core/include`
plus `/usr/include/python3.12`. `python3` is 3.12.3, NumPy is **2.4.6**.
b2 summary: `...failed updating 2 targets... ...skipped 10 targets...
...updated 266 targets...` The skipped targets are `libboost_numpy312`.
Nothing after Boost (rpclib and the rest of `Setup.sh`) ran. Not patched.
`make launch` not started.

### File-types dialog

The "Register Unreal Engine file types?" window was still mapped on `:99`
(268×100+506+308). `xdotool` is not installed, and `sudo` wanted a
password, so the click was `XTestFakeButtonEvent` from `libXtst` at the
**No** button (root 684, 380). The window is gone (`xwininfo` shows no
children on the root). Pids 348692 and 348889 exited with the dialog.
No kill signal was sent.
