#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TRANSFUSION_ENV="${TRANSFUSION_ENV:-$HOME/venvs/transfusion}"
MAMBA_BIN="${MAMBA_BIN:-$HOME/.local/bin/micromamba}"
"$MAMBA_BIN" create -y -p "$TRANSFUSION_ENV" -f research/environment.yml
"$TRANSFUSION_ENV/bin/python" scripts/install_cuda.py --prefix "$TRANSFUSION_ENV"
source scripts/activate_transfusion.sh
python -m pip install numpy==1.19.5 Cython==0.29.36
python -m pip install --no-build-isolation -r research/python-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python scripts/check_environment.py
