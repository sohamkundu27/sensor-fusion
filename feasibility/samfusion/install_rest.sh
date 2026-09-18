#!/bin/bash
set -x
E=/home/soham/sensor-fusion/feasibility/samfusion/env2
S=/home/soham/sensor-fusion/feasibility/samfusion/repo
export CUDA_HOME=$E PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
$E/bin/pip install mmdet3d==1.2.0
$E/bin/pip install spconv-cu113
$E/bin/pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
cd $S && $E/bin/pip install -e .
cd $S && $E/bin/python -c "import mmdet3d, spconv, detectron2, mmengine; import projects.SAMFusion.mmdet3d_plugin; print('SAMF DEPS OK mmdet3d', mmdet3d.__version__, 'spconv', spconv.__version__)"
