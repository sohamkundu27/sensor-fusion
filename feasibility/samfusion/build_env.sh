#!/bin/bash
set -x
export MAMBA_ROOT_PREFIX=/home/soham/sensor-fusion/feasibility/samfusion/mamba
MM=/home/soham/.local/bin/micromamba
$MM create -y -p ./env python=3.7.11 -c conda-forge 2>&1
$MM run -p ./env pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 2>&1
$MM run -p ./env python -c "import torch;print('TORCH',torch.__version__,torch.cuda.is_available())" 2>&1
