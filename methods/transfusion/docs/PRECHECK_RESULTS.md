# Initial training precheck — September 13, 2026

The pinned environment passed its import, model-construction and CUDA operator
checks. The full trainval presence check passed for 1,515,676 LiDAR/camera files,
850 scenes and 34,149 keyframes, including required map files. This checks presence
and nonzero size, not file contents. Download logs report archive CRC success but
published MD5 mismatches remain unresolved; publisher-checksum verification must
not be described as passed.

A bounded mini benchmark completed 12 real forward/backward/AdamW updates with
finite loss and gradient norms on the RTX 3080 12 GB. Settings: LiDAR-only voxel
config, FP32, batch 1, two data workers, ten sweeps, original voxel limits and
object sampling, LR 0.00000625, gradient clipping at 0.1. No class-balanced dataset
wrapper or LR schedule was used for this short timing run. Weights were randomly
initialized and discarded afterward; this is not an accuracy or convergence test.
Original model configs were not edited.

The last ten iterations averaged 0.976 seconds (median 0.979, range 0.387–1.371).
Peak PyTorch allocation was 2.563 GiB and reservation was 3.102 GiB. These numbers
exclude CUDA context/driver memory and do not establish worst-case full-data or
validation memory requirements. Small, cached mini samples are not representative
of all scene densities or sustained disk throughput.

The supplied object sampler has `data_root=None`, whereas the converter writes
relative database paths. The benchmark sets its `data_root` to `data/nuscenes`.
Apply the same correction in the eventual training configuration. Mini artifacts
use the separate `benchmark_mini` prefix; full trainval infos/database have not
been generated. The benchmark can be rerun from the method directory with the
prepared environment using `python scripts/benchmark_training.py`.
Raw timings are in `work_dirs/mini_precheck/report.json`.

Full metadata contains 28,130 training keyframes and 128,106 sample/category pairs.
The upstream CBGS wrapper therefore yields approximately 128,100 samples per epoch
(rounding can differ slightly). At the measured batch-1 mean, this extrapolates to
34.7 hours per epoch and 29 days for 20 epochs, excluding validation, preparation,
checkpoints and interruptions. Without CBGS the arithmetic would be 7.6 hours per
pass / 6.4 days for 20 passes, but that changes the training recipe and class balance.
Neither estimate is a measured full-data runtime or guarantees useful accuracy.

Next: resolve archive provenance/checksum discrepancies, prepare full trainval
infos and GT database, then benchmark more diverse full-data samples and validation.
Batch 2 is worth measuring given the initial headroom, but has not been tested.
Keep long training opt-in; the short benchmark has exited and the GPU is idle.
