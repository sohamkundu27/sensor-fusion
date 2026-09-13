# TransFusion environment

Activate from the project root:

```bash
source scripts/activate_transfusion.sh
python scripts/check_environment.py
```

The environment lives at `~/venvs/transfusion`, outside Git. It does not replace
the system Python or the separate nuScenes-download environment. Python user-site
packages are disabled on activation to avoid importing the machine's newer stack.

## Version choices

| Component | Version / source |
| --- | --- |
| Python | 3.7.12 |
| PyTorch | 1.7.0+cu110 |
| torchvision | 0.8.1+cu110 |
| MMCV | **mmcv-full 1.2.4**, official CUDA 11.0 / torch 1.7.0 wheel |
| MMDetection | 2.10.0 |
| MMDetection3D | This checkout, version 0.11.0, editable installation |
| spconv | Bundled legacy `mmdet3d.ops.spconv`, compiled from this checkout |
| CUDA compiler | 11.0.221, NVIDIA Ubuntu 20.04 development packages extracted locally |
| Host compiler | GCC/G++ 9.5, conda-forge; glibc 2.17 build sysroot |
| NumPy / Numba / llvmlite | 1.19.5 / 0.48.0 / 0.31.0 |
| nuScenes devkit | 1.1.9 |

The [upstream README](TRANSFUSION_UPSTREAM_README.md) selects mmdet 2.10.0 and
MMCV 1.2.4; [upstream experiment notes](../configs/nuscenes.md) use torch 1.7.0
with CUDA 10.1 on V100. Here CUDA 11.0 enables the Ampere GPU while keeping the
same torch/MMCV/mmdet generation. CUDA 11.0 compiles `8.0+PTX`; the RTX 3080 is
compute capability 8.6 and uses Ampere forward compatibility. The installed
NVIDIA driver is retained. The CUDA version printed by `nvidia-smi` is the driver's
maximum supported CUDA version, not the selected toolkit/runtime.

## Conflicts with current defaults

- Ubuntu 24.04's Python 3.12 and GCC 13 cannot be used as-is for these dependencies.
  The old Numba/llvmlite and MMCV wheels require an older Python, and CUDA 11.0
  needs an older host compiler. A compatible sysroot avoids new glibc header issues.
- Modern PyTorch 2.x, MMCV 2.x, MMDetection 3.x, and MMDetection3D 1.x are **not**
  drop-in replacements. Do not install a separate modern `mmdet3d` package over
  this checkout. `mmcv` without the `-full` compiled ops is insufficient.
- Do not install modern `spconv-cu11x`/`spconv-cu12x`: the code imports its bundled
  legacy spconv, built by `setup.py` together with voxelization and other ops.
- Upstream pins `numpy<1.20`, `numba==0.48.0`, `networkx<2.3`, and
  `trimesh==2.35.39`. New unpinned scientific packages can conflict with these.
- Install NumPy and Cython first, then use `--no-build-isolation` for old source
  packages. Pinning `yapf==0.32.0` avoids the removed `verify` argument used by
  old MMCV config formatting. No modern framework upgrade is hidden in this setup.
- Optional Waymo/TensorFlow and Open3D dependencies are not needed for nuScenes
  training and are not installed here.

## Recreate the environment

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
at `~/.local/bin/micromamba` (or set `MAMBA_BIN`), then run on Linux x86-64:

```bash
bash scripts/install_environment.sh
```

This uses [official PyTorch wheels](https://pytorch.org/get-started/previous-versions/),
the [matching OpenMMLab wheel index](https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html),
and checksum-pinned NVIDIA packages in `research/cuda-packages.json`.
`dpkg-deb -x` extracts compiler/development files without sudo or driver changes;
PyTorch supplies its CUDA math-library runtimes. The full multi-gigabyte CUDA
installer is unnecessary. Package archives remain in the user's cache.

The build targets this machine's Ampere GPU. It can take several minutes and
uses up to four compiler jobs by default (`MAX_JOBS` is configurable). A working
CUDA-visible NVIDIA GPU is needed for this checkout's extension selection.

## Validation scope

`scripts/check_environment.py` verifies pinned versions, CUDA availability,
MMCV NMS, voxelization, bundled sparse-convolution output and gradients on tiny
synthetic tensors, both voxel TransFusion model constructors, and imports of the
nuScenes converters. It does not load nuScenes, run model training, validate
full-sample memory use, or prove that the missing fusion checkpoint is available.

## Recorded validation result

The setup checks passed on September 12, 2026; see
[validation output](../research/validation.txt). `pip check` reported no broken
requirements. All ten custom extension modules compiled. Training and conversion
CLI help loaded successfully; no data preparation or training was executed.

The upstream optional DLA module prints `import DCN failed` because its separate
`dcn_v2` package is absent. The supplied voxel configurations use SECOND and
ResNet50, and both constructed successfully. Selecting the DLA alternative would
require its additional DCNv2 build; this setup does not claim that variant works.

[Python lock](../research/python-lock.txt) records all resolved Python packages
with public, hash-qualified URLs for the three GPU wheels. The
[conda lock](../research/conda-linux-64.lock) records the exact Linux packages;
`research/environment.yml` records the toolchain constraints. To reproduce the
exact conda packages instead of solving the YAML, use:

```bash
~/.local/bin/micromamba create -y -p ~/venvs/transfusion \
  -f research/conda-linux-64.lock
```

The activation script also exposes libxcrypt headers and overrides old Python's
host-root linker flag with the conda sysroot; both were necessary for extension
compilation on this Ubuntu release. These changes affect only the sourced shell.
