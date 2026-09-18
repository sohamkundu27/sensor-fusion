#!/bin/bash
# SAMFusion timing: batch 1, 30 iterations, nuScenes-mini (mmengine harness).
set -x
D=/home/soham/sensor-fusion/feasibility/samfusion
E=$D/env2
S=$D/repo
export CUDA_HOME=$E PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib PYTHONPATH=$S:$PYTHONPATH TORCH_CUDA_ARCH_LIST=8.6
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
cd $S
$E/bin/python /home/soham/sensor-fusion/feasibility/bench_mmengine.py \
  --method samfusion --config $S/projects/SAMFusion/configs/nuscenes/SAMFusion.py \
  --ann-file nuscenes_mini_radar_mmdet_v2_infos_train.pkl \
  --unwrap-cbgs --iters 30 --warmup 5 --out $D/result.json
