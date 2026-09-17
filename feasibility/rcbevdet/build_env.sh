#!/bin/bash
set -x
export MAMBA_ROOT_PREFIX=/home/soham/sensor-fusion/feasibility/mamba_root
MM=/home/soham/.local/bin/micromamba
D=/home/soham/sensor-fusion/feasibility/rcbevdet
cd $D
# Same proven pattern as bevfusion/env2: self-consistent CUDA first, nvidia label
# pinned ahead of conda-forge so a newer cudart cannot be pulled over it.
$MM create -y -p ./env -c nvidia/label/cuda-11.3.1 -c conda-forge python=3.8 cuda gxx_linux-64=10
grep -m1 "define CUDART_VERSION" ./env/include/cuda_runtime_api.h
./env/bin/pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
./env/bin/pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
./env/bin/pip install cython==0.29.36 numpy==1.23.5
./env/bin/pip install mmdet==2.24.0 mmsegmentation==0.24.0
./env/bin/pip install numba==0.53.1 nuscenes-devkit pyquaternion Pillow==8.4.0 tqdm setuptools==59.5.0 ninja
./env/bin/python -c "import torch,mmcv,mmdet;print('RCBEV ENV OK torch',torch.__version__,'mmcv',mmcv.__version__,'mmdet',mmdet.__version__,'cuda',torch.cuda.is_available())"
