# Full nuScenes experiment

Started September 13, 2026 at 15:52 CDT using training code commit
`af8830de0b48832e74f5fc2147ec3d5795898b2e`. This is a fresh entropy-fusion
run with ImageNet camera initialization, rather than a continuation of the mini
checkpoint. The experiment targets 20 epochs on the official train split, with
all six cameras, batch size 1, FP32, AdamW LR 1e-4 and seed 42.

There are 168,780 camera-view training items per epoch (28,130 keyframes), for
3,375,600 optimizer steps over 20 epochs. Validation processes all 36,114 views
of 6,019 val keyframes after epochs 1, 5, 10, 15 and 20. The runner retains these
numbered checkpoints and selects `best.pt` by validation mAP.

## Input check and startup

All 409,788 sensor files referenced by the 34,149 train/val keyframes exist and
are nonempty. LiDAR file sizes pass the five-float record-size check. Forty-eight
representative train/val camera views decoded to finite RGB/LiDAR/radar inputs.
The local report is `outputs/full_data_check.json`. This does not establish that
all file contents match the publisher: the earlier archive MD5 discrepancies
remain unresolved. Archive CRC checks were reported successful by the downloader.

The full-data model completed its first training steps with finite losses and
gradients. The setup did not change model architecture or training hyperparameters
from the mini baseline. Successful startup does not establish final accuracy.

## Job and artifacts

The job runs in the user service `entropy-fusion-full.service`. User lingering was
enabled so logging out does not stop the service. The machine must remain powered
on. The original transient service did not resume after reboot; the persistent
recovery unit added below is now configured to do so. Automatic AC idle
sleep timeout was already set to zero at launch. Manual suspend or shutdown will
interrupt execution.

All run artifacts are under `outputs/full_20epoch_20260913/`. `status.json` records
stage transitions; `training/metrics.jsonl` records live training steps. Each stage
has its own training/evaluation log. `training/last.pt` is written every 1,000
microbatches at optimizer boundaries and at epoch completion. Numbered evaluated
checkpoints and `best.pt` are separate. Logs/checkpoints/data stay outside Git.

From this method's directory:

```bash
systemctl --user status entropy-fusion-full.service --no-pager
tail -n 1 outputs/full_20epoch_20260913/training/metrics.jsonl
journalctl --user -u entropy-fusion-full.service -n 20 --no-pager
```

To stop the run deliberately:

```bash
systemctl --user stop entropy-fusion-full.service
```

If interrupted, the latest checkpoint preserves optimizer/RNG state and the
position within that epoch. Up to 999 steps since its save may be lost. After
confirming that the original process has stopped, resume the staged runner:

```bash
python -m scripts.run_full_experiment --epochs 20 --validate-every 5 \
  --output outputs/full_20epoch_20260913 --resume
```

This retains scheduled validation and archives the previous status and metrics.
Metrics after the restored checkpoint are removed from the active metrics file;
the original log remains in the archive. Stage logs are appended to. The local
service now includes `--resume`, so restarting that existing service uses this
recovery path. If systemd has collected the stopped transient unit, it must be
created again; the Python command above also works in the prepared environment. Loading metadata and replaying the shuffled loader up to the saved
position can leave the GPU idle briefly before updates resume.

## September 13 recovery

The first attempt crashed at 15:56 CDT while loading step 5,478: a projected
LiDAR coordinate rounded to x=640 for a 640-pixel image. Commit `c3bc5bc` filters
points that round outside the image after projection and resizing. The exact
failing CAM_BACK sample now passes loading and a finite forward/backward update;
boundary regression tests cover this case. The rasterizer still rejects invalid
inputs rather than silently dropping entire samples.

Commit `328cb53` adds staged-run recovery with log preservation. All 12 method
tests passed. The service restarted at 18:12 CDT from step 5,000, replaying the
477 successful updates that had not been checkpointed. This recovery preserves
the model, optimizer and validation schedule; it does not reset training.

The first recovery attempt encountered CPU thread contention during loader replay.
The service was restarted with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`; PyTorch's existing explicit training thread setting remains.
Training then passed step 8,234 with no skipped updates since recovery. The new
step-8,000 checkpoint was reopened and all model tensors were finite. GPU
utilization was sampled at 70%. These are observations at recovery verification,
not a continuously updated health report; inspect the service and live metrics
for current status.

The initial planning estimate is 2–3 days including validation. Re-estimate from
sustained full-data throughput; no completion or accuracy guarantee is implied.
The earlier mini run achieved only 4.76% mAP, so this remains an exploratory run.

## September 13 kernel stall — reboot pending

At 20:25:28 CDT, training stopped advancing at step 341,110 in epoch 3.
The kernel journal records a supervisor-mode page fault in `folio_mark_dirty`
while a `pt_data_worker` was unmapping memory, at that same timestamp. Later
messages report CPU soft lockups during worker cleanup. The cause of the kernel
fault is not established; it should not be attributed to a specific driver,
package or hardware component without further investigation. Restarting the
service did not restore training. A running service alone is not evidence of
progress. The saved local journal is `outputs/full_20epoch_20260913/kernel_stall_20260913.log`.

The step-341,000 checkpoint was reopened successfully and its model tensors were
finite. Recovery will replay 110 updates. Epoch 1 full-validation results are
10.77% mAP and 18.77% NDS. The experiment has not finished.

A persistent user unit has now been written to
`~/.config/systemd/user/entropy-fusion-full.service` and linked into
`default.target.wants`, replacing reliance on a transient unit after reboot.
With user lingering enabled, it is configured to resume this checkpoint and the
scheduled validations when the user service manager starts after reboot. This
post-reboot recovery has not yet been verified. Save open work and reboot the
machine, then verify fresh step records beyond 341,110 and a new checkpoint.
Commit `2f94216` adds SIGUSR1 thread dumps to the training log for future diagnosis.
