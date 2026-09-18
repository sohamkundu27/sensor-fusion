#!/bin/bash
# RCBEVDet timing: batch 1, 30 iterations, nuScenes-mini.
set -x
D=/home/soham/sensor-fusion/feasibility/rcbevdet
E=$D/env
C=$D/code/rcbevdet-master
export CUDA_HOME=$E LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH PYTHONPATH=$C:$PYTHONPATH
cd $C
$E/bin/python /home/soham/sensor-fusion/feasibility/bench_common.py \
  --method rcbevdet --config $C/configs/rcbevdet/bench_mini.py \
  --unwrap-cbgs --iters 30 --warmup 5 --out $D/result.json
