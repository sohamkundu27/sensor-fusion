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
on; the service does not automatically resume after a reboot. Automatic AC idle
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
position within that epoch. Up to 999 steps since its save may be lost. The staged
runner requires an empty directory for a new experiment; do not blindly restart
it on existing artifacts. Training can be resumed explicitly in the prepared
environment after confirming that the original process has stopped:

```bash
python train.py --version v1.0-trainval --split train --epochs 20 \
  --checkpoint-every 1000 --output outputs/full_20epoch_20260913/training \
  --resume outputs/full_20epoch_20260913/training/last.pt
```

That manual command resumes training only; schedule evaluation separately if
recovering outside the staged runner. See `eval.py --help` and use `--split val`.
The original staged run performs its scheduled evaluations automatically.

The initial planning estimate is 2–3 days including validation. Re-estimate from
sustained full-data throughput; no completion or accuracy guarantee is implied.
The earlier mini run achieved only 4.76% mAP, so this remains an exploratory run.
