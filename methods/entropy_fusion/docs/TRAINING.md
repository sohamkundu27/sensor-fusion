# Running the 3D baseline

Use the existing environment from this method's directory:

```bash
cd ~/sensor-fusion/methods/entropy_fusion
source ~/venvs/entropy-fusion/bin/activate
```

## Verification and a bounded benchmark

```bash
python -m pytest -q
python -m scripts.verify_batch --workers 2
python -m scripts.check_learning
python train.py --max-steps 12 --output outputs/my_benchmark
python eval.py --checkpoint outputs/my_benchmark/last.pt --output outputs/my_evaluation
```

Training defaults to all six mini training camera views, batch size 1, FP32 and
two workers. The first run downloads the official ImageNet ResNet18 weights into
the PyTorch cache (about 45 MB). `--no-pretrained` is an explicit random-camera
ablation. Generated predictions, logs, reports and checkpoints stay in ignored
`outputs/`. Choose a new output directory for each experiment; existing checkpoints
are protected from accidental replacement without `--resume`.

The verified benchmark from setup is `outputs/benchmark_3d/last.pt`, with evaluation
under `outputs/eval_3d/`. It has only 12 updates and is not a useful detector yet.
Earlier `benchmark_fp32` artifacts predate the final coordinate encoding and are
rejected by the current checkpoint format check.

## Mini training and resume

```bash
python train.py --epochs 20 --output outputs/mini_experiment
python train.py --epochs 20 --resume outputs/mini_experiment/last.pt --output outputs/mini_experiment
python eval.py --checkpoint outputs/mini_experiment/last.pt --output outputs/mini_experiment_eval
```

These longer training commands were not run during implementation. There is no
background training job. Inspect the mini learning curves and held-out metrics
before investing in full training; loss reduction on one repeated training view
only verifies learning mechanics.

Checkpoints are written at epoch completion and at an explicit `--max-steps` stop.
A crash or interrupt mid-epoch can lose updates since the preceding checkpoint.
`--max-steps` counts absolute microbatches, including after resume, not optimizer
updates. It flushes any partial accumulation window before saving. Resuming a run
that was deliberately stopped inside a window therefore preserves its flushed
updates rather than matching an uninterrupted window exactly. Resume restores
weights, optimizer, scaler, RNG and the position in that epoch's shuffle; keep the
original batch size, cameras, resolution, split, seed, precision, accumulation,
LR and dropout settings. Extending `--epochs` or `--max-steps` is supported.

## Scheduled mini experiment

To train for 20 epochs with official validation at epochs 1, 5, 10, 15 and 20:

```bash
python -m scripts.run_mini_experiment --output outputs/my_mini_run --epochs 20 --validate-every 5
python -m scripts.summarize_experiment outputs/my_mini_run
```

The runner starts fresh from ImageNet camera weights and the configured seed,
resuming optimizer/RNG state between validation stages. It writes stage logs,
`status.json`, numbered evaluated checkpoints, and `best.pt` selected by validation
mAP. The second command produces `summary.json` and `learning_curves.png`; it also
works while training, in which case the latest epoch may be incomplete. Use an
empty directory for the runner. If the runner fails, preserve its output and use
`train.py --resume` with `training/last.pt` to continue training explicitly.

## Precision and accumulation

```bash
python train.py --amp --accumulation-steps 2 --max-steps 5 --output outputs/my_amp_check
```

This five-microbatch case performs three optimizer updates, testing the last
partial window. FP16 with dynamic scaling is supported on this GPU; initial scale
is 1024. Overflowed optimizer updates are skipped and logged. FP32 remains the
baseline, and AMP is not guaranteed to improve this small model's throughput.
There is no automatic learning-rate rescaling when batch size or accumulation
changes. The starting LR is a research hyperparameter, not a tuned recipe.

## Full train/validation

The same loader reads the original devkit metadata without TransFusion info files
or an object-sampling database. Before using the full split, finish integrity checks
for raw LiDAR, radar and camera files: the earlier full-data check covered LiDAR and
camera presence only, and archive MD5 discrepancies remain unresolved.

```bash
python train.py --version v1.0-trainval --split train --epochs 20 --output outputs/full_experiment
python eval.py --split val --checkpoint outputs/full_experiment/last.pt --output outputs/full_experiment_eval
```

Full-data training and validation have not been exercised. The devkit holds full
metadata in CPU memory; monitor host RAM and loader throughput. Mini timings do not
establish full-data runtime. Each complete six-view pass visits 168,780 training
items (28,130 keyframes); there is no CBGS expansion in this method.

Evaluation always uses all six views. A limited `--max-batches` invocation writes
`partial_predictions.json` and reports no official metrics unless it actually
covered every view. Complete evaluation writes `predictions.json`, `run.json`,
`metrics_summary.json` and devkit metric details. Global same-class distance merging
and score thresholds need tuning on validation data, without using the test split.
