#!/bin/bash
set -x
export MAMBA_ROOT_PREFIX=/home/soham/sensor-fusion/feasibility/mamba_root
MM=/home/soham/.local/bin/micromamba
D=/home/soham/sensor-fusion/feasibility/bevfusion
cd $D
$MM create -y -q -p ./env python=3.8 -c conda-forge
./env/bin/pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
./env/bin/pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
./env/bin/pip install mmdet==2.20.0
./env/bin/pip install numpy==1.23.5 numba==0.48.0 nuscenes-devkit pyquaternion Pillow==8.4.0 tqdm
./env/bin/pip install torchpack
./env/bin/python -c "import torch,mmcv,mmdet;print('OK torch',torch.__version__,'mmcv',mmcv.__version__,'mmdet',mmdet.__version__,'cuda',torch.cuda.is_available())"
