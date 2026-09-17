#!/bin/bash
set -x
export MAMBA_ROOT_PREFIX=/home/soham/sensor-fusion/feasibility/mamba_root
MM=/home/soham/.local/bin/micromamba
D=/home/soham/sensor-fusion/feasibility/bevfusion
cd $D
rm -rf env2
# Self-consistent CUDA 11.3 toolkit FIRST, from the nvidia 11.3.1 label only,
# so conda-forge cannot pull a newer cudart over it.
$MM create -y -p ./env2 -c nvidia/label/cuda-11.3.1 -c conda-forge python=3.8 cuda gxx_linux-64=10
./env2/bin/python -c "import sys;print('py',sys.version)"
grep -m1 CUDART_VERSION ./env2/targets/x86_64-linux/include/cuda_runtime_api.h
./env2/bin/nvcc --version | tail -2
./env2/bin/pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
./env2/bin/pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
./env2/bin/pip install cython==0.29.36 numpy==1.23.5
./env2/bin/pip install mmdet==2.20.0
./env2/bin/pip install numba==0.48.0 nuscenes-devkit pyquaternion Pillow==8.4.0 tqdm setuptools==59.5.0 ninja
./env2/bin/python -c "import torch,mmcv,mmdet;print('ENV2 OK torch',torch.__version__,'mmcv',mmcv.__version__,'mmdet',mmdet.__version__,'cuda',torch.cuda.is_available())"
