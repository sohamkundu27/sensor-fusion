#!/bin/bash
# Waits for the gating ablation to finish, then runs each method's bench.sh.
# Detached: survives the Claude session. Never touches the ablation.
F=/home/soham/sensor-fusion/feasibility
ABL=/home/soham/sensor-fusion/methods/entropy_fusion/outputs/nuscenes_gating_pair_seed42/off/status.json
LOG=$F/orchestrator.log
echo "[$(date -u +%FT%TZ)] watcher started, waiting for ablation" >> $LOG

# 1. Wait for the ablation to report completed (poll every 5 min, cap 30h)
for i in $(seq 1 360); do
  STATE=$(python3 -c "import json;print(json.load(open('$ABL'))['state'])" 2>/dev/null)
  if [ "$STATE" = "completed" ]; then
    echo "[$(date -u +%FT%TZ)] ablation completed" >> $LOG; break
  fi
  sleep 300
done

# 2. Confirm GPU actually idle before timing anything
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$USED" -eq 0 ] && break
  echo "[$(date -u +%FT%TZ)] GPU still has $USED process(es), waiting" >> $LOG
  sleep 120
done
echo "[$(date -u +%FT%TZ)] GPU clear, starting benchmarks" >> $LOG

# 3. Run each method's bench.sh sequentially (never in parallel: timings must not contend)
for m in bevfusion transfusion samfusion rcbevdet; do
  if [ -x "$F/$m/bench.sh" ]; then
    echo "[$(date -u +%FT%TZ)] === $m START ===" >> $LOG
    timeout 7200 "$F/$m/bench.sh" > "$F/$m/bench.log" 2>&1
    echo "[$(date -u +%FT%TZ)] === $m END rc=$? ===" >> $LOG
  else
    echo "[$(date -u +%FT%TZ)] $m SKIPPED (no bench.sh - env build did not complete)" >> $LOG
  fi
done
echo "[$(date -u +%FT%TZ)] all benchmarks done" >> $LOG

# 4. Regenerate the results file from whatever produced a result.json
python3 /home/soham/sensor-fusion/feasibility/make_summary.py >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] SUMMARY.md regenerated" >> $LOG
