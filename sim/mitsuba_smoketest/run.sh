#!/usr/bin/env bash
# Clear + foggy smoke-test renders, then analysis. Logs to outputs/run.log.
# Usage: ./run.sh [sigma_t_fog] [spp]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/venvs/mitsuba/bin/python}
FOG=${1:-0.15}
SPP=${2:-16384}
mkdir -p outputs
exec > >(tee outputs/run.log) 2>&1

gpu_busy() {
  local procs util
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  [[ -z "$util" ]] && return 0
  (( procs > 0 || util > 10 ))
}

snapshot() {
  echo "--- $(date '+%F %T %z') $1"
  uptime
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
  top -bn1 -o %CPU | sed -n '8,11p'
}

if gpu_busy; then VARIANT=llvm_ad_mono; else VARIANT=cuda_ad_mono; fi
echo "variant=$VARIANT fog_sigma_t=$FOG spp=$SPP"

rm -rf "$HOME/.drjit"   # Dr.Jit kernel cache: force a true cold JIT compile
for S in 0.0 "$FOG"; do
  NAME=$([[ "$S" == "0.0" ]] && echo clear || echo fog)
  snapshot "before $NAME render"
  "$PY" render.py --variant "$VARIANT" --sigma-t "$S" --spp "$SPP" --repeats 2 --out "outputs/$NAME.npz"
done
snapshot "after renders"
"$PY" analyze.py
