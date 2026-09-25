#!/usr/bin/env bash
# Build without shell startup files or ambient compiler/linker flags.
set -euo pipefail

if [[ $# != 2 || ( "$2" != setup && "$2" != launch ) ]]; then
  echo "Usage: $0 CARLA_SOURCE_DIRECTORY setup|launch" >&2
  exit 2
fi
fix_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
carla_dir="$(cd -- "$1" && pwd)"
ue4_dir="${UE4_ROOT:-$HOME/UnrealEngine_4.26}"
if [[ ! -x "$ue4_dir/Engine/Binaries/Linux/UE4Editor" ]]; then
  echo "Set UE4_ROOT to the installed CARLA Unreal Engine 4.26 checkout." >&2
  exit 1
fi
"$fix_dir/apply.sh" "$carla_dir"

exec env -i HOME="$HOME" USER="$(id -un)" LOGNAME="$(id -un)" \
  PATH=/usr/bin:/bin LANG=C.UTF-8 UE4_ROOT="$ue4_dir" \
  DISPLAY="${DISPLAY:-:99}" CARLA_BUILD_NO_COLOR=1 \
  /bin/bash --noprofile --norc -c 'cd -- "$1"; exec make "$2"' \
  carla-build "$carla_dir" "$2"
