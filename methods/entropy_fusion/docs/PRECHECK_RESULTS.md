# 3D implementation checks — September 13, 2026

The entropy-fusion model, training losses, checkpoint/resume path and official
nuScenes evaluator are implemented. The earlier data-only milestone is complete. The later [20-epoch mini experiment](MINI_EXPERIMENT.md) records sustained training and validation results; the measurements below describe the initial short checks.

## Measured FP32 benchmark

RTX 3080 12 GB, batch size 1, 640 x 384, two loader workers, all six camera channels
in the training dataset, p=0.5 modality dropout, ImageNet-pretrained ResNet18 and
randomly initialized remaining layers. There are 13,205,670 model parameters.

Twelve real mini training microbatches completed forward/backward/AdamW updates
with finite losses and gradients. Excluding the first two iterations, mean wall
time including loader wait and input transfer was **0.0409 seconds per camera view**.
Peak PyTorch allocation was **0.341 GiB**, with **0.389 GiB reserved**. Driver/context
memory is additional; these are not whole-device memory readings or worst-case
full-data measurements. Raw logs are in `outputs/benchmark_3d/metrics.jsonl` and
summary in `outputs/benchmark_3d/report.json`.

One TransFusion step processes a LiDAR scene with ten sweeps; this model processes
one image view with current-frame projected sensors. Six views at the measured
rate are approximately 0.245 seconds per keyframe, compared with the earlier
0.976-second TransFusion batch-1 check. Different inputs, sampling, architectures
and objectives make this only a preliminary compute comparison, not an accuracy
comparison. No full epoch was timed for either method here.

## Correctness checks

- Ten unit tests cover sensor geometry, masked entropy, stream dropout/gates,
  box-coordinate roundtrips, invalid velocity masking, matching conflicts,
  empty scenes, valid result schema and cross-view duplicate suppression.
- Real data-only mini batches passed on the front camera and all six validation
  views, including independent devkit projection comparison and CUDA transfer.
- A repeated training-view check confirmed nonzero finite gradients in all three
  backbones, every entropy gate at all resolutions, and the head. Over 40 steps,
  mean loss fell from **8.72** (first five) to **1.06** (last five). Dropout was disabled
  for this controlled learning check; these weights were discarded.
- FP16 and two-step accumulation passed five microbatches / three optimizer updates,
  including the final one-microbatch window, with no skipped updates. Resume then
  completed microbatches six and seven and preserved the expected four total
  optimizer updates. The scaler remained at 1024. Initial experiments with the
  larger default scale overflowed; the runner now reduces scale/skips overflowed
  updates and raises an error if overflow persists.
- The final 12-step checkpoint was evaluated on **all 486 camera views / 81 samples**
  in mini_val. Official nuScenes evaluation completed with mAP **0.00073956** and
  NDS **0.00725848** (both expressed on the 0–1 scale). Near-zero accuracy is expected
  after 12 updates and establishes only that the evaluation path works.

The current benchmark checkpoint is `outputs/benchmark_3d/last.pt`; final evaluation
outputs are in `outputs/eval_3d/`. A compact record is committed in
[benchmark_3d.json](benchmark_3d.json). Raw data, generated predictions and model
weights are not committed. No long-running training process is active.

## Remaining experimental work

Train on mini long enough to inspect learning and held-out predictions, then
validate full-data integrity (including radar and unresolved archive MD5 issues)
before full experiments. Tune depth regression, class balance, merging and learning
rate. Evaluate modality ablations and actual corrupted/adverse conditions before
making robustness claims. This is a new 3D baseline inspired by the paper, not a
validated reproduction or an established competitor to TransFusion.
