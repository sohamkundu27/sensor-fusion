#!/usr/bin/env bash
# Apply the saved fixes to the pinned CARLA source checkout.
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "Usage: $0 CARLA_SOURCE_DIRECTORY" >&2
  exit 2
fi
fix_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
carla_dir="$(cd -- "$1" && pwd)"
expected_commit=294096eb1c38eabf246e4f3a9cdab704e33a7f4c
if [[ $(git -C "$carla_dir" rev-parse HEAD) != "$expected_commit" ]]; then
  echo "Expected CARLA 0.9.16 at $expected_commit; refusing to patch another revision." >&2
  exit 1
fi

patch_file="$fix_dir/carla-0.9.16-ubuntu24.patch"
if git -C "$carla_dir" apply --check "$patch_file" 2>/dev/null; then
  git -C "$carla_dir" apply "$patch_file"
elif git -C "$carla_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "CARLA build-script fixes are already applied."
else
  echo "Build scripts do not match the saved patch or its applied state." >&2
  exit 1
fi

# Setup.sh applies this exact patch after extracting Boost on each fresh build.
install -D -m 644 "$fix_dir/boost_numpy2_dtype.patch" "$HOME/carla/patches/boost_numpy2_dtype.patch"
