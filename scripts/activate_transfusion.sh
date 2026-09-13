# Source this file from bash: source scripts/activate_transfusion.sh
export TRANSFUSION_ENV="${TRANSFUSION_ENV:-$HOME/venvs/transfusion}"
export PATH="$TRANSFUSION_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CUDA_HOME="$TRANSFUSION_ENV/cuda-debs/usr/local/cuda-11.0"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$TRANSFUSION_ENV/bin/x86_64-conda-linux-gnu-cc"
export CXX="$TRANSFUSION_ENV/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
# CUDA 11.0 supports sm_80; Ampere 8.6 runs this code with forward compatibility.
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export MAX_JOBS="${MAX_JOBS:-4}"
export CPATH="$TRANSFUSION_ENV/include${CPATH:+:$CPATH}"
# Older conda Python embeds --sysroot=/ in its linker flags; override it for this build.
export LDFLAGS="--sysroot=$TRANSFUSION_ENV/x86_64-conda-linux-gnu/sysroot -Wl,--sysroot=$TRANSFUSION_ENV/x86_64-conda-linux-gnu/sysroot"
