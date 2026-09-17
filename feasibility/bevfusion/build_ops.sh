#!/bin/bash
set -x
E=/home/soham/sensor-fusion/feasibility/bevfusion/env2
cd /home/soham/sensor-fusion/feasibility/bevfusion/repo
# keep build cache
export CUDA_HOME=$E
export PATH=$E/bin:$PATH
export LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export CC=$E/bin/x86_64-conda-linux-gnu-gcc
export CXX=$E/bin/x86_64-conda-linux-gnu-g++
export MAX_JOBS=8
$E/bin/python setup.py develop
$E/bin/python -c "import mmdet3d; from mmdet3d.ops import bev_pool; print('MMDET3D OPS OK')"
