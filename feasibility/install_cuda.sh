#!/bin/bash
set -x
export MAMBA_ROOT_PREFIX=/home/soham/sensor-fusion/feasibility/mamba_root
MM=/home/soham/.local/bin/micromamba
# nvcc 11.3 to match torch 1.10.0+cu113; needed to compile CUDA extensions.
$MM install -y -q -p /home/soham/sensor-fusion/feasibility/bevfusion/env \
  -c nvidia/label/cuda-11.3.1 -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-cccl libcusparse-dev libcublas-dev libcusolver-dev 2>&1
ls -la /home/soham/sensor-fusion/feasibility/bevfusion/env/bin/nvcc
/home/soham/sensor-fusion/feasibility/bevfusion/env/bin/nvcc --version
