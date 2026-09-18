#!/bin/bash
set -x
E=/home/soham/sensor-fusion/feasibility/samfusion/env2
export CUDA_HOME=$E PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib PYTHONPATH=/home/soham/sensor-fusion/feasibility/samfusion/repo:/home/soham/sensor-fusion/feasibility/samfusion/repo/tools:$PYTHONPATH
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++ TORCH_CUDA_ARCH_LIST=8.6
cd /home/soham/sensor-fusion/feasibility/samfusion/repo && $E/bin/python -u tools/create_data.py nuscenes --root-path ./data/nuscenes --version v1.0-mini --out-dir ./data/nuscenes --extra-tag nuscenes_mini
ls -la /home/soham/data/nuscenes/nuscenes_mini_radar*
