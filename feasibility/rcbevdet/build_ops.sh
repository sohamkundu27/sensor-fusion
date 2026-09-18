#!/bin/bash
set -x
E=/home/soham/sensor-fusion/feasibility/rcbevdet/env
C=/home/soham/sensor-fusion/feasibility/rcbevdet/code/rcbevdet-master
export CUDA_HOME=$E PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
# spconv-cu113 already installed
cd $C && $E/bin/python setup.py develop
cd $C/mmdet3d/ops/csrc && $E/bin/python setup.py build_ext --inplace
cd $C/mmdet3d/ops/deformattn && $E/bin/python setup.py build install
cd $C && $E/bin/python -c "import mmdet3d, spconv; print('RCBEV OPS OK', mmdet3d.__version__, spconv.__version__)"
