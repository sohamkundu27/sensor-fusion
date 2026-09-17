#!/bin/bash
set -x
export MAMBA_ROOT_PREFIX=/home/soham/sensor-fusion/feasibility/mamba_root
MM=/home/soham/.local/bin/micromamba
D=/home/soham/sensor-fusion/feasibility/samfusion
cd $D
rm -rf env2
$MM create -y -p ./env2 -c nvidia/label/cuda-11.3.1 -c conda-forge python=3.8 cuda gxx_linux-64=10 && $MM install -y -q -p ./env2 -c conda-forge "sysroot_linux-64=2.17"
grep -m1 "define CUDART_VERSION" ./env2/include/cuda_runtime_api.h
./env2/bin/pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# mmcv 2.0.1 IS published for cu113/torch1.10.0 -- upstream pins are consistent.
./env2/bin/pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
./env2/bin/pip install cython==0.29.36 numpy==1.23.5
./env2/bin/pip install mmdet==3.1.0 mmengine
./env2/bin/pip install numba==0.53.1 nuscenes-devkit pyquaternion Pillow==8.4.0 tqdm setuptools==59.5.0 ninja
./env2/bin/python -c "import torch,mmcv,mmdet;print('SAMF ENV OK torch',torch.__version__,'mmcv',mmcv.__version__,'mmdet',mmdet.__version__,'cuda',torch.cuda.is_available())"
