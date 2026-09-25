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

On a fresh, unmodified CARLA 0.9.16 source tree at the same location:

```sh
mkdir -p ~/carla/patches
cp /home/soham/sensor-fusion/sim/carla_setup/boost_numpy2_dtype.patch ~/carla/patches/
cd ~/carla/carla-0.9.16-src
git apply --check /home/soham/sensor-fusion/sim/carla_setup/carla-0.9.16-ubuntu24.patch
git apply /home/soham/sensor-fusion/sim/carla_setup/carla-0.9.16-ubuntu24.patch
export UE4_ROOT=/home/soham/UnrealEngine_4.26
export DISPLAY=:99
make setup
make launch
```

The live tree is already patched. Do not apply the combined patch again.
The patch preserves dependency versions and the previously validated NumPy fix.

Capture the editor using the locally extracted ImageMagick tool:

```sh
xwd -display :99 -root -silent -out /tmp/carla-editor.xwd
~/carla/tools/imagemagick/bin/convert /tmp/carla-editor.xwd /tmp/carla-editor.png
```

Inspect the PNG to confirm the editor is actually visible. Build and launch
findings, exact logs, and screenshots are recorded in
[CARLA_EDITOR_EXPORT.md](../../docs/CARLA_EDITOR_EXPORT.md).
