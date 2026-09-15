# KITTI camera–LiDAR car experiment

KITTI is the interim dataset while Seeing Through Fog registration is pending.
This remains **3D detection**, with a car-only classifier and two available
modalities. It tests our entropy-fusion adaptation; KITTI does not supply radar
or gated NIR, so it cannot reproduce the paper's four-stream adverse-weather study.

## Data and access

The [KITTI object benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
provides 7,481 labeled training frames and an unlabeled test set. We use the common
3,712/3,769 labeled train/validation split from
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/233f849829b6ac19afb8af8837a0246890908755/data/kitti/ImageSets).
The source revision and split hashes are recorded in each download manifest.
The website's login links coexist with accessible public `avg-kitti` S3 archives;
these public archives were checked and used directly. There is no test-server submission.

- `~/data/kitti`: verified 32-training/8-validation-frame subset, 111,233,681 bytes.
- `~/data/kitti_full`: full labeled set downloading in the background.
- `~/data/kitti_tools`: checksum-pinned official evaluation source and locally
  extracted Ubuntu Boost headers. The nuScenes CUDA environment is preserved.

The downloader fetches selected ZIP members using persistent HTTP connections and
validated byte ranges. It preserves zero-padded frame IDs, verifies ZIP CRCs,
records SHA-256 hashes and checks existing files before resuming. Images, LiDAR,
calibration and labels are included. Small annotation/calibration archives are
cached during download; complete large archives are not retained. Unlabeled test
images/point clouds are not extracted. Partial files have a temporary suffix.

## Geometry and model

The adapter uses rectification and Velodyne-to-camera calibration, including P2's
nonzero camera baseline. KITTI bottom-centered boxes become geometric centers
internally and are converted back for evaluation. Internal X/Y/Z axes are
forward/left/up in a local frame, not geographic coordinates. Tests independently
check projection, bottom-center conversion, dimensions, yaw and prediction export.
LiDAR channels encode camera depth, sensor-frame height and reflectance; KITTI
reflectance is already in [0,1] and is not divided by 255. Inputs use no labels.

The existing revised network has one foreground class. Radar inputs and masks
are always absent; they are not treated as measured zero returns. Available-sensor
dropout chooses between camera and LiDAR with probability 0.5, retaining at least
one available stream. DontCare and Van regions suppress negative-anchor training;
KITTI's labeled 3D boxes remain supervised without invented point-count or velocity
labels. NuScenes keeps its original defaults and checkpoint shapes.

The full recipe uses near-native **384x1248** inputs, batch two, ImageNet camera
initialization, random sensor/fusion/head weights, selective AMP with FP32 camera
and exchange layers, AdamW at 2e-4, 100-step warmup and 20-epoch cosine decay.
The added metric-center weight ramps over 1,000 updates. This is our KITTI baseline
recipe, not a claim of the original paper's training settings. Full training has
37,120 batches. Validation runs at epochs 1, 5, 10, 15 and 20; `best.pt` is selected
by moderate **car 3D AP_R40 at IoU 0.7**, not nuScenes mAP/NDS.

## Verification

All 40 downloaded subset frames loaded with finite inputs. There were 77 valid
car training boxes and 20 validation boxes. RGB/LiDAR overlays were inspected;
the 3D coordinate roundtrip and ignored-region behavior have regression tests.
The initial 384x640 repeated-two-image check reduced loss from 7.599 to 1.391 and
mean center error from 4.734m to 1.099m over 80 FP32 updates. All active branches
received gradients; peak allocated VRAM was 0.605 GiB. This measures learning
mechanics on those two training images, not generalization.

A five-epoch execution check on the 32/8 subset completed training, checkpoint
resume and both scheduled evaluations. Its 80 training updates produced **0% AP**
on that tiny validation set; it is not evidence of useful detection accuracy.
The full recipe and larger validation split still need to establish that.
Artifacts are under `outputs/kitti_smoke_20260915` and `outputs/kitti_precheck`.

The same five-epoch execution check also passed at the selected 384x1248
resolution, including automatic checkpoint continuation and both evaluations.
It peaked at 1.044 GiB allocated VRAM and averaged 0.122 seconds per batch in the
final stage. Its tiny-set AP was also zero. This is preserved separately under
`outputs/kitti_native_smoke_20260915`. All **45 regression tests** passed, including
the nuScenes resume tests, KITTI metric selection, and the verified-download gate.

The evaluator uses the official KITTI devkit archive with SHA-256
`ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1`.
Local changes set the frame count, replace the server entry point/logging and
disable plotting. The matching and difficulty logic remain upstream. The wrapper
averages precision entries 1 through 40 for R40, reporting 2D, BEV and 3D car AP
separately. Synthetic perfect predictions scored 100% for every metric/difficulty;
displacing their 3D centers scored 0% BEV/3D while leaving 2D at 100%. These are
evaluator tests, not model scores. The readme and original source are retained
alongside the patched source; no email or remote submission is performed.

## Background services

The downloader is `kitti-download.service`. The experiment service is
`kitti-experiment.service`, which waits for the full manifest to be verified,
checks all expected file sizes and split membership, then starts the resumable
20-epoch runner. A failed download stops that queue rather than launching on
partial data. The full run starts from fresh initialization, not the tiny pilot.

```bash
systemctl --user status kitti-download.service kitti-experiment.service --no-pager
cat ~/data/kitti_full/velodyne_progress.json
cat outputs/kitti_full_20260915.wait.json
# Available once the training runner starts:
cat outputs/kitti_full_20260915/status.json
```

The services survive disconnect/logout with user lingering enabled. They are not
enabled to run automatically at boot. After a reboot or resolved download error,
start the downloader and experiment services again; existing verified files and
training checkpoints are reused. Keep the computer awake and powered on.

Reproduce the initial bounded checks from the method directory:

```bash
python -m scripts.download_kitti_subset
python -m scripts.check_kitti
python -m scripts.check_kitti_evaluator
```

The local evaluator requires a C++ compiler and Boost headers. On this machine,
Ubuntu's `libboost1.83-dev` package was downloaded with apt and extracted under
`~/data/kitti_tools/deps`; no system-wide installation was needed.
