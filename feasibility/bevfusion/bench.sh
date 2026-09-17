#!/bin/bash
# BEVFusion timing benchmark: batch 1, 30 iterations, nuScenes-mini.
# Called by run_when_free.sh once the GPU is idle.
set -x
D=/home/soham/sensor-fusion/feasibility/bevfusion
E=$D/env2
export CUDA_HOME=$E
export LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$D/repo:$PYTHONPATH
cd $D/repo
$E/bin/python /home/soham/sensor-fusion/feasibility/bench_common.py \
  --method bevfusion \
  --config $D/repo/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bench_mini.yaml \
  --loader torchpack --unwrap-cbgs --drop-steps ObjectPaste --iters 30 --warmup 5 \
  --out $D/result.json
