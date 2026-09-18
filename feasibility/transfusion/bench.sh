#!/bin/bash
# TransFusion timing: batch 1, 30 iterations, nuScenes-mini. VARIANT=L (default) or LC.
set -x
D=/home/soham/sensor-fusion/feasibility/transfusion
E=$D/env2
V=${VARIANT:-L}
export CUDA_HOME=$E LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH PYTHONPATH=$D/repo:$PYTHONPATH
cd $D/repo
OUT=$D/result.json; [ "$V" != "L" ] && OUT=$D/result_$V.json
$E/bin/python /home/soham/sensor-fusion/feasibility/bench_common.py \
  --method transfusion-$V --config $D/repo/configs/bench_mini_voxel_$V.py \
  --unwrap-cbgs --drop-steps ObjectSample --iters 30 --warmup 5 --out $OUT
