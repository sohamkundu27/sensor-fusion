# CARLA 0.9.16 on this Ubuntu 24.04 machine

The working build uses `/home/soham/UnrealEngine_4.26`, its clang 10 compiler,
and its libc++ for Unreal-facing libraries. Host dependency links use the
installed GNU gold linker. UnrealBuildTool keeps its own bundled LLD linker.

`carla-0.9.16-ubuntu24.patch` contains the build-script changes relative to
CARLA commit `294096e`. `boost_numpy2_dtype.patch` is copied byte-for-byte
from the validated patch in `docs/CARLA_EDITOR_EXPORT.md`.

The build-script patch fixes:

- Boost bootstrap/library linking against the newer host glibc, and applies
  the NumPy 2 dtype patch after each fresh Boost extraction.
- CMake/Autoconf host dependency links through shared `LDFLAGS`.
- The relocated official libpng 1.6.37 archive URL.
- Old libtool dropping linker-selection flags in libpng and SQLite.
- PROJ 7.2.1's missing `<cstdint>` include and position-independent C objects.
- Unreal's sysroot for the editor-facing Xerces, PROJ, and OSM2ODR builds,
  preventing references to glibc 2.38+ C23 symbols unavailable in UE's sysroot.
- Detection and cleanup of incomplete SQLite/Xerces/PROJ installations on retry.

Prerequisites are the installed, built CARLA Unreal Engine 4.26 at commit
`e9d9e60c85f643e10eeb03f42f61554d18dcb30f`, the host development tools
(including GNU gold), Python 3.12 with NumPy 2.4.6, and official CARLA content
version `20250912_2171890`. Content belongs at
`Unreal/CarlaUE4/Content/Carla`; a symlink to that existing asset installation
is sufficient. Launch requires a working X display and Vulkan driver.

On a fresh CARLA source checkout at
`294096eb1c38eabf246e4f3a9cdab704e33a7f4c`:

```sh
export UE4_ROOT=/home/soham/UnrealEngine_4.26
export DISPLAY=:99
./sim/carla_setup/build.sh ~/carla/carla-0.9.16-src setup
./sim/carla_setup/build.sh ~/carla/carla-0.9.16-src launch
```

`apply.sh` checks the CARLA revision, applies the saved script patch (or verifies
it is already applied), and installs the exact NumPy patch at the path used by
Setup.sh. `build.sh` runs `make` in a fresh shell with a cleared environment;
compiler/linker flags come from the committed patch. The patch preserves
dependency versions and the previously validated NumPy fix.

For a clean reproduction, use a separate source checkout with no `Build/`,
project/plugin binaries, intermediates, or `CarlaDependencies/` copied in.
Reuse only the installed engine, official content, and host tools above.
This tests a clean CARLA build, not a new OS or an engine rebuild. Shader/asset
caches may be reused; they do not supply CARLA libraries or project binaries.

Capture the editor using the locally extracted ImageMagick tool:

```sh
xwd -display :99 -root -silent -out /tmp/carla-editor.xwd
~/carla/tools/imagemagick/bin/convert /tmp/carla-editor.xwd /tmp/carla-editor.png
```

Inspect the PNG to confirm the editor is actually visible. Build and launch
findings, exact logs, and screenshots are recorded in
[CARLA_EDITOR_EXPORT.md](../../docs/CARLA_EDITOR_EXPORT.md).
